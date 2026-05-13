"""
embed_checkpoint.py - Save/load OpenAI embedding checkpoints for NAICS Classifier.

A .naics_embed file is a ZIP archive containing:

  meta.json         — checkpoint type, config snapshot, model metadata

  Training checkpoints (ckpt_type="training"):
    emb_general.npy / lbl_general.npy   — general model embeddings + encoded labels
    emb_cat_N.npy   / lbl_cat_N.npy     — per-category embeddings + labels (multi only)

  Prediction checkpoints (ckpt_type="prediction"):
    emb_pred.npy    — all-row embeddings shape (n, 3072), original row order
    df.parquet      — DataFrame in Parquet format (includes _desc and _supplier columns)
"""

import json
import os
import tempfile
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Training checkpoints
# ---------------------------------------------------------------------------

def save_training_ckpt(
    path: str,
    models: Dict[str, dict],
    cfg_snapshot: dict,
    categories: List[str],
) -> None:
    """Save training-embedding checkpoint.

    models must contain a "__general__" key plus one key per category.
    Each entry: {"X": ndarray, "y": ndarray, "label_map": dict,
                 "naics_descs": dict, "prompt_template": str, "n_rows": int}
    """
    bundle_type = "multi" if categories else "single"

    meta: dict = {
        "ckpt_type":   "training",
        "bundle_type": bundle_type,
        "categories":  categories,
        "created_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config":      cfg_snapshot,
        "models":      {},
    }
    for key, m in models.items():
        meta["models"][key] = {
            "n_rows":          m["n_rows"],
            "n_classes":       len(m["label_map"]),
            "prompt_template": m["prompt_template"],
            "label_map":       {str(k): v for k, v in m["label_map"].items()},
            "naics_descs":     m["naics_descs"],
        }

    with tempfile.TemporaryDirectory() as tmp:
        meta_path = os.path.join(tmp, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(meta_path, "meta.json")

            # General model arrays
            gen = models["__general__"]
            p_emb = os.path.join(tmp, "emb_general.npy")
            p_lbl = os.path.join(tmp, "lbl_general.npy")
            np.save(p_emb, gen["X"])
            np.save(p_lbl, gen["y"])
            zf.write(p_emb, "emb_general.npy")
            zf.write(p_lbl, "lbl_general.npy")

            # Per-category arrays
            for i, cat in enumerate(categories):
                cm = models[cat]
                p_ce = os.path.join(tmp, f"emb_cat_{i}.npy")
                p_cl = os.path.join(tmp, f"lbl_cat_{i}.npy")
                np.save(p_ce, cm["X"])
                np.save(p_cl, cm["y"])
                zf.write(p_ce, f"emb_cat_{i}.npy")
                zf.write(p_cl, f"lbl_cat_{i}.npy")


def save_training_ckpt_from_npzs(
    path: str,
    models_meta: Dict[str, dict],
    npz_paths: Dict[str, str],
    cfg_snapshot: dict,
    categories: List[str],
) -> None:
    """Save training-embedding checkpoint by loading npz files one at a time.

    Unlike save_training_ckpt (which requires all arrays in memory), this
    function loads each npz file, writes its arrays to the ZIP, and frees them
    before moving to the next model.  Peak memory = one model's arrays at a time.

    models_meta: {key: {"label_map", "naics_descs", "prompt_template", "n_rows"}}
    npz_paths:   {key: "/path/to/emb_<key>.npz"}  — each npz has {"X": ndarray, "y": ndarray}
    """
    bundle_type = "multi" if categories else "single"
    meta: dict = {
        "ckpt_type":   "training",
        "bundle_type": bundle_type,
        "categories":  categories,
        "created_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config":      cfg_snapshot,
        "models":      {},
    }
    for key, m in models_meta.items():
        meta["models"][key] = {
            "n_rows":          m["n_rows"],
            "n_classes":       len(m["label_map"]),
            "prompt_template": m["prompt_template"],
            "label_map":       {str(k): v for k, v in m["label_map"].items()},
            "naics_descs":     m["naics_descs"],
        }

    with tempfile.TemporaryDirectory() as tmp:
        meta_path = os.path.join(tmp, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(meta_path, "meta.json")

            # General model — load npz, write arrays, free immediately
            gen_data = _load_npz(npz_paths["__general__"])
            _write_npy_to_zip(zf, tmp, "emb_general.npy", gen_data["X"])
            _write_npy_to_zip(zf, tmp, "lbl_general.npy", gen_data["y"])
            del gen_data

            # Per-category — one at a time
            for i, cat in enumerate(categories):
                cat_data = _load_npz(npz_paths[cat])
                _write_npy_to_zip(zf, tmp, f"emb_cat_{i}.npy", cat_data["X"])
                _write_npy_to_zip(zf, tmp, f"lbl_cat_{i}.npy", cat_data["y"])
                del cat_data


def _load_npz(npz_path: str) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        return {"X": data["X"].copy(), "y": data["y"].copy()}


def _write_npy_to_zip(zf: zipfile.ZipFile, tmp: str, name: str, arr: np.ndarray) -> None:
    p = os.path.join(tmp, name)
    np.save(p, arr)
    zf.write(p, name)
    os.remove(p)  # free temp file immediately after adding to ZIP


def load_training_ckpt_meta(path: str) -> dict:
    """Load training checkpoint metadata only — no numpy arrays loaded.

    Use load_training_ckpt_arrays() to fetch X/y arrays for one model at a time.
    """
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("meta.json") as f:
            meta = json.load(f)

    categories  = meta.get("categories", [])
    models_meta = meta.get("models", {})
    models: Dict[str, dict] = {}

    for key, mm in models_meta.items():
        models[key] = {
            "label_map":       {int(k): v for k, v in mm.get("label_map", {}).items()},
            "naics_descs":     mm.get("naics_descs", {}),
            "prompt_template": mm.get("prompt_template", ""),
            "n_rows":          mm.get("n_rows", 0),
        }

    return {
        "bundle_type":    meta.get("bundle_type", "single"),
        "categories":     categories,
        "config":         meta.get("config", {}),
        "n_rows_general": models.get("__general__", {}).get("n_rows", 0),
        "models":         models,
    }


def load_training_ckpt_arrays(
    path: str,
    key: str,
    cat_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the numpy arrays (X, y) for a single model from a training checkpoint.

    key:       "__general__" or a category string
    cat_index: integer index for category models (position in the categories list);
               ignored for "__general__"
    """
    if key == "__general__":
        emb_fname = "emb_general.npy"
        lbl_fname = "lbl_general.npy"
    else:
        if cat_index is None:
            raise ValueError(f"cat_index required for category key '{key}'")
        emb_fname = f"emb_cat_{cat_index}.npy"
        lbl_fname = f"lbl_cat_{cat_index}.npy"

    with zipfile.ZipFile(path, "r") as zf:
        with tempfile.TemporaryDirectory() as td:
            zf.extract(emb_fname, td)
            zf.extract(lbl_fname, td)
            X = np.load(os.path.join(td, emb_fname))
            y = np.load(os.path.join(td, lbl_fname))

    return X, y


# ---------------------------------------------------------------------------
# Prediction checkpoints
# ---------------------------------------------------------------------------

def save_prediction_ckpt(
    path: str,
    df: pd.DataFrame,
    X_flat: np.ndarray,
    cfg_snapshot: dict,
) -> None:
    """Save prediction-embedding checkpoint (all rows, original order)."""
    meta = {
        "ckpt_type":  "prediction",
        "n_rows":     len(df),
        "n_dims":     int(X_flat.shape[1]) if X_flat.ndim == 2 else 3072,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config":     cfg_snapshot,
    }
    with tempfile.TemporaryDirectory() as tmp:
        meta_path = os.path.join(tmp, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        emb_path = os.path.join(tmp, "emb_pred.npy")
        np.save(emb_path, X_flat)

        df_path = os.path.join(tmp, "df.parquet")
        df.to_parquet(df_path, index=True)

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(meta_path, "meta.json")
            zf.write(emb_path,  "emb_pred.npy")
            zf.write(df_path,   "df.parquet")


def load_prediction_ckpt(path: str) -> dict:
    """Load prediction checkpoint.

    Returns::

        {"df": DataFrame, "X_flat": ndarray, "config": dict, "n_rows": int}
    """
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("meta.json") as f:
            meta = json.load(f)

        namelist = zf.namelist()
        with tempfile.TemporaryDirectory() as td:
            zf.extract("emb_pred.npy", td)
            X_flat = np.load(os.path.join(td, "emb_pred.npy"))

            zf.extract("df.parquet", td)
            df = pd.read_parquet(os.path.join(td, "df.parquet"))

    return {
        "df":     df,
        "X_flat": X_flat,
        "config": meta.get("config", {}),
        "n_rows": meta.get("n_rows", len(df)),
    }


# ---------------------------------------------------------------------------
# Metadata peek (no arrays loaded)
# ---------------------------------------------------------------------------

def peek_ckpt(path: str) -> dict:
    """Read metadata only — does NOT load embeddings or dataframe."""
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("meta.json") as f:
            return json.load(f)
