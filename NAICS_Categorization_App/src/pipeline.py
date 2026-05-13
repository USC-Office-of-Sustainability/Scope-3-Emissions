"""
pipeline.py - Training and prediction workers for NAICS Classifier.

Both workers are plain Python classes run inside QThread wrappers in app.py.
Progress is reported via a callback dict:

  {"type": "log",      "message": "...", "level": "info|success|warning|error"}
  {"type": "progress", "stage": "step|embedding|training|predicting|done",
                        "pct": 0-100, ...extra}

Multi-model training (use_category=True in config)
---------------------------------------------------
When a category column is supplied, one model is trained per valid category
plus a general model trained on all rows.  All models are packed into a single
.naics_model file (bundle_type="multi").

Multi-model prediction (bundle.is_multi + category_col in config)
-----------------------------------------------------------------
Each row is routed to its matching category model; rows with an unrecognised
or absent category fall back to the general model.  A "ML_model_used" column
is appended to the output to indicate which model handled each row.
"""

import os
import re
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import gc

import numpy as np
import pandas as pd
import xgboost as xgb
from openai import (
    OpenAI,
    AuthenticationError as OAIAuthError,
    PermissionDeniedError,
    APIConnectionError as OAIConnectionError,
    RateLimitError as OAIRateLimitError,
    APITimeoutError as OAITimeoutError,
    InternalServerError as OAIServerError,
)
from sklearn.preprocessing import LabelEncoder

from model_bundle import ModelBundle
from embed_checkpoint import (
    save_training_ckpt, save_training_ckpt_from_npzs,
    load_training_ckpt_meta, load_training_ckpt_arrays,
    save_prediction_ckpt, load_prediction_ckpt,
)

_EMBED_DIM = 3072  # text-embedding-3-large output dimension
_EMBED_RETRY_DELAYS = (2, 5, 15, 30, 60)  # seconds between attempts
_EMBED_MAX_RETRIES = 5


def _make_dtrain(X: np.ndarray, y: np.ndarray) -> xgb.DMatrix:
    """Build a training DMatrix with minimum persistent memory.

    QuantileDMatrix (XGBoost ≥ 1.7) stores only uint8 bin indices (~307 MB for
    100k×3072) instead of raw float32 data (~1.2 GB), saving ~900 MB that would
    otherwise persist for the entire training run.  Falls back to regular DMatrix
    on older XGBoost versions.
    """
    try:
        return xgb.QuantileDMatrix(X, label=y, max_bin=256, nthread=-1)
    except AttributeError:
        return xgb.DMatrix(X, label=y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _norm_code(v) -> str:
    """Normalise a NAICS code value to a clean string (337214.0 → '337214')."""
    s = str(v).strip()
    if "." in s:
        try:
            s = str(int(float(s)))
        except ValueError:
            pass
    return s


def clean_text(text) -> str:
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_prompt(template: str, supplier: str, description: str) -> str:
    """Fill template slots; gracefully omit supplier if missing."""
    supplier = str(supplier).strip() if supplier else ""
    if supplier.lower() in {"", "nan", "n/a", "na", "none", "null"}:
        supplier = ""
    description = str(description).strip()

    if supplier:
        return template.replace("{Supplier}", supplier).replace("{Description}", description)

    # Supplier absent — remove its placeholder and tidy punctuation
    prompt = template.replace("{Supplier}", "").replace("{Description}", description)
    prompt = re.sub(r"Supplier:\s*,?", "", prompt)
    prompt = re.sub(r",\s*,", ",", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip().strip(",").strip()
    return prompt


def _make_cat_prompt_template(category_key: str) -> str:
    """Auto-generate a prompt template by injecting a data category value."""
    key = str(category_key).strip()
    if key:
        return f"NAICS code for {key} item. Supplier: {{Supplier}}, Description: {{Description}}"
    return "NAICS code for item. Supplier: {Supplier}, Description: {Description}"


def detect_columns(df: pd.DataFrame) -> dict:
    """Heuristically suggest column roles from header names."""
    def _norm(s: str) -> str:
        return re.sub(r"[\s_\-\.]+", "", s.lower())

    normalized = {_norm(c): c for c in df.columns}

    def find(*keywords):
        for kw in keywords:
            kw_n = _norm(kw)
            for n, orig in normalized.items():
                if kw_n in n:
                    return orig
        return None

    suggestions = {}
    suggestions["desc_col"]      = find("lineitem", "line item", "line_item", "description", "item desc", "product", "name")
    suggestions["supplier_col"]  = find("supplier", "vendor", "company")
    suggestions["label_col"]     = find("naics code", "naicscode", "naics", "code", "label", "class")
    suggestions["naics_desc_col"] = find("code description", "code desc", "codedescription")
    suggestions["category_col"]  = find("spend category", "procurement category", "product category", "category", "spend type", "type")
    return suggestions


# ---------------------------------------------------------------------------
# Shared embedding helper
# ---------------------------------------------------------------------------

def _do_embed(
    prompts: List[str],
    api_key: str,
    batch_size: int,
    progress_cb: Callable,
    stop_checker: Callable[[], bool],
    embed_offset: int = 0,
    embed_total: Optional[int] = None,
    model_idx: int = 0,
    model_total: int = 1,
    model_label: str = "General",
    tok_offset: int = 0,
    overall_batch_offset: int = 0,
    overall_total_batches: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Embed *prompts* via OpenAI text-embedding-3-large.

    embed_offset / embed_total allow cumulative progress reporting across
    multiple sequential embedding passes (multi-model training/prediction).
    model_idx / model_total / model_label identify which model is being embedded.
    tok_offset carries cumulative tokens from previous models for cost tracking.

    Memory: pre-allocates a single float32 array — avoids the ~8× overhead of
    accumulating a Python list of Python lists and converting at the end.

    Reliability: transient network / rate-limit errors are retried with
    exponential back-off; auth errors abort immediately.
    """
    client    = OpenAI(api_key=api_key)
    emb_model = "text-embedding-3-large"
    tok_in  = 0
    n       = len(prompts)
    if embed_total is None:
        embed_total = n
    t0 = time.time()

    def _log(msg, level="info"):
        progress_cb({"type": "log", "message": msg, "level": level})

    def _prog(pct, **kw):
        progress_cb({"type": "progress", "stage": "embedding", "pct": pct, **kw})

    # Pre-allocate output array up front.
    # A Python list of Python lists for 100 k rows × 3072 dims costs ~9–10 GB;
    # this float32 array costs ~1.2 GB and requires no conversion at the end.
    X = np.empty((n, _EMBED_DIM), dtype=np.float32)

    for i in range(0, n, batch_size):
        if stop_checker():
            return None, 0
        batch = prompts[i: i + batch_size]

        last_exc: Optional[Exception] = None
        for attempt in range(_EMBED_MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=emb_model, input=batch)
                for item in resp.data:
                    X[i + item.index] = item.embedding
                tok_in += resp.usage.prompt_tokens
                last_exc = None
                break
            except (OAIAuthError, PermissionDeniedError) as e:
                _log(f"Invalid API key: {e}", "error")
                progress_cb({"type": "auth_error", "message": str(e)})
                return None, 0
            except (OAIConnectionError, OAIRateLimitError, OAITimeoutError, OAIServerError) as e:
                delay = _EMBED_RETRY_DELAYS[min(attempt, len(_EMBED_RETRY_DELAYS) - 1)]
                _log(
                    f"Embedding API transient error (batch {i // batch_size + 1}, "
                    f"attempt {attempt + 1}/{_EMBED_MAX_RETRIES}): {e}. "
                    f"Retrying in {delay}s…",
                    "warning",
                )
                last_exc = e
                for _ in range(delay):
                    if stop_checker():
                        return None, 0
                    time.sleep(1)
            except Exception as e:
                delay = _EMBED_RETRY_DELAYS[min(attempt, len(_EMBED_RETRY_DELAYS) - 1)]
                _log(
                    f"Embedding API error (batch {i // batch_size + 1}, "
                    f"attempt {attempt + 1}/{_EMBED_MAX_RETRIES}): {e}. "
                    f"Retrying in {delay}s…",
                    "warning",
                )
                last_exc = e
                for _ in range(delay):
                    if stop_checker():
                        return None, 0
                    time.sleep(1)

        if last_exc is not None:
            _log(
                f"Embedding batch {i // batch_size + 1} failed after "
                f"{_EMBED_MAX_RETRIES} attempts: {last_exc}",
                "error",
            )
            return None, 0

        local_done    = min(i + batch_size, n)
        global_done   = embed_offset + local_done
        pct           = global_done / embed_total * 100
        model_pct     = local_done / n * 100
        elapsed       = time.time() - t0
        rate          = local_done / elapsed if elapsed > 0 else 0
        eta_model     = _format_eta((n - local_done) / rate) if rate > 0 else "—"
        eta_overall   = _format_eta((embed_total - global_done) / rate) if rate > 0 else "—"
        batch_done    = i // batch_size + 1
        total_batches = (n + batch_size - 1) // batch_size
        ovr_total_b   = overall_total_batches if overall_total_batches is not None else total_batches
        ovr_batch     = overall_batch_offset + batch_done
        tokens_total  = tok_offset + tok_in
        _prog(pct,
              done=global_done, total=embed_total,
              tokens_total=tokens_total,
              eta_model=eta_model, eta_overall=eta_overall,
              batch_done=batch_done, total_batches=total_batches,
              overall_batch_done=ovr_batch, overall_total_batches=ovr_total_b,
              model_idx=model_idx, model_total=model_total, model_label=model_label,
              model_done=local_done, model_total_rows=n, model_pct=model_pct)
        _log(
            f"[{model_label} {model_idx + 1}/{model_total}] "
            f"Rows: {global_done}/{embed_total} ({pct:.1f}%)  |  "
            f"Tokens: {tokens_total:,}  |  "
            f"Model ETA: {eta_model}  |  Overall ETA: {eta_overall}"
        )

    return X, tok_in


# ---------------------------------------------------------------------------
# Training Worker
# ---------------------------------------------------------------------------

class TrainWorker:
    def __init__(self, config: dict, progress_cb: Callable, stop_checker: Callable[[], bool],
                 skip_checker: Optional[Callable[[], bool]] = None):
        self.config = config
        self.cb = progress_cb
        self.should_stop = stop_checker
        self._skip_checker = skip_checker or (lambda: False)

    def should_skip(self) -> bool:
        return self._skip_checker()

    def _log(self, msg: str, level: str = "info"):
        self.cb({"type": "log", "message": msg, "level": level})

    def _prog(self, stage: str, pct: float, **kw):
        self.cb({"type": "progress", "stage": stage, "pct": pct, **kw})

    # ------------------------------------------------------------------
    def run(self) -> Optional[str]:
        cfg = self.config
        model_dir  = cfg["model_dir"]
        model_name = cfg["model_name"]
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        _src       = cfg.get("input_file") or cfg.get("load_ckpt_path") or "training"
        dataset_id = Path(_src).stem
        temp_dir   = os.path.join(model_dir, "temp", f"{dataset_id}_{ts}")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        try:
            # ── Checkpoint bypass: skip steps 1-3 ────────────────────────
            if cfg.get("load_ckpt_path"):
                try:
                    return self._run_from_ckpt(cfg, model_dir, model_name, ts, temp_dir)
                except Exception as e:
                    self._log(f"Unexpected error: {e}", "error")
                    self._log(traceback.format_exc(), "error")
                    return None

            use_category = cfg.get("use_category", False)
            category_col = cfg.get("category_col", "")

            try:
                # ── Step 1: Load & clean ──────────────────────────────────
                self._log("Step 1/5 — Loading and cleaning data...")
                self._prog("step", 0, step=1, step_name="Load & Clean")

                df = self._load_data(cfg)
                if df is None:
                    return None
                self._log(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

                label_col     = cfg["label_col"]
                desc_col      = cfg["desc_col"]
                supplier_col  = cfg.get("supplier_col", "")
                naics_desc_col = cfg.get("naics_desc_col", "")

                # Extract NAICS descriptions (full dataset)
                naics_descriptions: Dict[str, str] = {}
                if naics_desc_col and naics_desc_col in df.columns:
                    pairs = df[[label_col, naics_desc_col]].dropna().drop_duplicates()
                    naics_descriptions = {
                        _norm_code(row[label_col]): str(row[naics_desc_col])
                        for _, row in pairs.iterrows()
                    }
                    self._log(f"Extracted {len(naics_descriptions)} NAICS descriptions.")
                else:
                    self._log("No NAICS description column — descriptions will be empty.", "warning")

                df["_desc"]     = df[desc_col].apply(clean_text)
                df["_supplier"] = (
                    df[supplier_col].apply(clean_text)
                    if supplier_col and supplier_col in df.columns else ""
                )

                self._prog("step", 20, step=1, step_name="Load & Clean")
                if self.should_stop():
                    return None

                # ── Decide: single or multi-model ────────────────────────
                do_multi = (
                    use_category
                    and bool(category_col)
                    and category_col in df.columns
                )

                if do_multi:
                    return self._run_multi(df, naics_descriptions, cfg, temp_dir, model_dir, model_name, ts, label_col, supplier_col)
                else:
                    return self._run_single(df, naics_descriptions, cfg, temp_dir, model_dir, model_name, ts, label_col, supplier_col)

            except Exception as e:
                self._log(f"Unexpected error: {e}", "error")
                self._log(traceback.format_exc(), "error")
                return None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def _run_from_ckpt(self, cfg, model_dir, model_name, ts, temp_dir) -> Optional[str]:
        """Train directly from a .naics_embed checkpoint, skipping steps 1–3."""
        load_path = cfg["load_ckpt_path"]
        self._log(f"Loading embedding checkpoint: {Path(load_path).name}")

        # Load metadata only first (no numpy arrays) to log the plan without
        # pulling all category embeddings into RAM simultaneously.
        ckpt_meta = load_training_ckpt_meta(load_path)
        self._prog("step", 60, step=4, step_name="Train XGBoost")
        bundle_type = ckpt_meta["bundle_type"]
        categories  = ckpt_meta["categories"]
        cfg_snap    = ckpt_meta.get("config", {})
        label_col    = cfg_snap.get("label_col", "")
        supplier_col = cfg_snap.get("supplier_col", "")
        category_col = cfg_snap.get("category_col", "")

        n_gen = ckpt_meta["n_rows_general"]
        summary = f"{bundle_type} | {n_gen:,} general rows"
        if categories:
            summary += f" | {len(categories)} categories: {', '.join(categories)}"
        self._log(f"Checkpoint loaded: {summary}", "success")

        if bundle_type == "single":
            gen_meta = ckpt_meta["models"]["__general__"]
            label_map          = gen_meta["label_map"]
            naics_descriptions = gen_meta["naics_descs"]
            template           = gen_meta["prompt_template"]
            self._log(f"General model: {n_gen:,} rows, {len(label_map)} classes.")

            if self.should_stop():
                return None

            self._log("Step 4/5 — Training XGBoost model...")
            X, y = load_training_ckpt_arrays(load_path, "__general__")

            actual_n_rows_gen = len(y)
            dtrain = _make_dtrain(X, y)
            y_eval = y
            del X, y
            gc.collect()
            booster = self._train(dtrain, len(label_map), cfg,
                                  model_idx=0, model_total=1, model_label="General")
            if booster is None:
                return None
            self._prog("step", 80, step=4, step_name="Train XGBoost")

            self._log("Step 5/5 — Evaluating on training set...")
            self._prog("step", 80, step=5, step_name="Evaluate")
            preds    = booster.predict(dtrain)
            pred_lbl = np.argmax(preds, axis=1) if preds.ndim == 2 else preds.astype(int)
            acc = float(np.mean(pred_lbl == y_eval))
            del dtrain, y_eval, preds, pred_lbl
            self._log(f"Training accuracy: {acc * 100:.2f}%", "success")

            _bundle_ts = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            bundle_path = os.path.join(model_dir, f"{model_name}_{_bundle_ts}.naics_model")
            training_cfg = {
                "desc_col":          cfg_snap.get("desc_col", ""),
                "supplier_col":      supplier_col,
                "label_col":         label_col,
                "batch_size":        cfg_snap.get("batch_size", cfg.get("batch_size", 200)),
                "max_depth":         cfg["max_depth"],
                "num_boost_round":   cfg["num_boost_round"],
                "training_accuracy": acc,
                "num_training_rows": actual_n_rows_gen,
                "trained_at":        ts,
            }
            ModelBundle(
                model_name=model_name, prompt_template=template,
                label_map=label_map, naics_descriptions=naics_descriptions,
                training_config=training_cfg, booster=booster, bundle_type="single",
            ).save(bundle_path)
            self._log(f"Model saved → {bundle_path}", "success")
            self._prog("step", 100, step=5, step_name="Done")
            self._prog("done", 100, model_path=bundle_path, accuracy=acc)
            return bundle_path

        # ── multi ─────────────────────────────────────────────────────
        # Use metadata only (already loaded above) to log the plan.
        n_models = 1 + len(categories)
        gen_meta = ckpt_meta["models"]["__general__"]
        self._log(f"Will train {n_models} models total:")
        self._log(f"  [1/{n_models}] General  ({gen_meta['n_rows']:,} rows, {len(gen_meta['label_map'])} classes)")
        for i, cat in enumerate(categories, 2):
            cd = ckpt_meta["models"][cat]
            self._log(f"  [{i}/{n_models}] Category '{cat}'  ({cd['n_rows']:,} rows, {len(cd['label_map'])} classes)")

        if self.should_stop():
            return None

        self._log("Step 4/5 — Training XGBoost models...")

        total_rounds = cfg["num_boost_round"] * n_models
        round_offset = 0
        trained: Dict[str, dict] = {}

        # Stream arrays one model at a time — never hold all category embeddings in RAM.
        train_order_keys = [("__general__", None)] + [(cat, i) for i, cat in enumerate(categories)]

        trained_n_rows_ckpt: Dict[str, int] = {}

        for idx, (key, cat_index) in enumerate(train_order_keys):
            model_label = "General" if key == "__general__" else key
            lm = ckpt_meta["models"][key]["label_map"]
            self._log(f"  Training {model_label} ({idx + 1}/{n_models})...")

            X, y = load_training_ckpt_arrays(load_path, key, cat_index)

            trained_n_rows_ckpt[key] = len(y)
            dtrain = _make_dtrain(X, y)
            y_eval = y
            del X, y
            gc.collect()

            booster = self._train(dtrain, len(lm), cfg,
                                  train_offset=round_offset, train_total=total_rounds,
                                  model_idx=idx, model_total=n_models, model_label=model_label)
            if booster is None:
                return None

            preds    = booster.predict(dtrain)
            pred_lbl = np.argmax(preds, axis=1) if preds.ndim == 2 else preds.astype(int)
            acc = float(np.mean(pred_lbl == y_eval))
            self._log(f"    {model_label} training accuracy: {acc * 100:.2f}%", "success")
            trained[key] = {"booster": booster, "label_map": lm, "acc": acc}
            round_offset += cfg["num_boost_round"]
            del dtrain, y_eval, preds, pred_lbl

        self._prog("step", 80, step=4, step_name="Train XGBoost")

        self._log("Step 5/5 — Assembling multi-model bundle...")
        self._prog("step", 80, step=5, step_name="Evaluate")

        gen  = trained["__general__"]
        batch_size_snap = cfg_snap.get("batch_size", cfg.get("batch_size", 200))
        gen_training_cfg = {
            "desc_col":              cfg_snap.get("desc_col", ""),
            "supplier_col":          supplier_col,
            "label_col":             label_col,
            "category_col":          category_col,
            "batch_size":            batch_size_snap,
            "max_depth":             cfg["max_depth"],
            "num_boost_round":       cfg["num_boost_round"],
            "training_accuracy":     gen["acc"],
            "num_training_rows":     trained_n_rows_ckpt.get("__general__", gen_meta["n_rows"]),
            "trained_at":            ts,
            "num_category_models":   len(categories),
        }
        category_bundles: Dict[str, ModelBundle] = {}
        for cat in categories:
            td_cat = trained[cat]
            cd     = ckpt_meta["models"][cat]
            category_bundles[cat] = ModelBundle(
                model_name=f"{model_name}_{cat}",
                prompt_template=cd["prompt_template"],
                label_map=td_cat["label_map"],
                naics_descriptions=cd["naics_descs"],
                training_config={
                    "desc_col": cfg_snap.get("desc_col", ""), "supplier_col": supplier_col,
                    "label_col": label_col, "category_col": category_col, "category": cat,
                    "batch_size": batch_size_snap, "max_depth": cfg["max_depth"],
                    "num_boost_round": cfg["num_boost_round"],
                    "training_accuracy": td_cat["acc"],
                    "num_training_rows": trained_n_rows_ckpt.get(cat, cd["n_rows"]),
                    "trained_at": ts,
                },
                booster=td_cat["booster"], bundle_type="single",
            )

        _bundle_ts  = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        bundle_path = os.path.join(model_dir, f"{model_name}_{_bundle_ts}.naics_model")
        ModelBundle(
            model_name=model_name,
            prompt_template=gen_meta["prompt_template"],
            label_map=gen["label_map"],
            naics_descriptions=gen_meta["naics_descs"],
            training_config=gen_training_cfg,
            booster=gen["booster"],
            bundle_type="multi",
            category_bundles=category_bundles,
        ).save(bundle_path)
        self._log(
            f"Multi-model bundle saved → {bundle_path}  "
            f"(1 general + {len(categories)} category models)", "success"
        )
        self._prog("step", 100, step=5, step_name="Done")
        self._prog("done", 100, model_path=bundle_path, accuracy=gen["acc"])
        return bundle_path

    # ------------------------------------------------------------------
    def _run_single(self, df, naics_descriptions, cfg, temp_dir, model_dir, model_name, ts, label_col, supplier_col) -> Optional[str]:
        """Train one general model on all rows (original behaviour)."""
        template = cfg["prompt_template"]

        # Step 2: Encode labels
        self._log("Step 2/5 — Encoding NAICS labels...")
        self._prog("step", 20, step=2, step_name="Encode Labels")
        le = LabelEncoder()
        y  = le.fit_transform(df[label_col].apply(_norm_code))
        label_map: Dict[int, str] = {int(i): _norm_code(c) for i, c in enumerate(le.classes_)}
        n_rows = len(df)
        self._log(f"Found {len(label_map)} unique NAICS classes.")
        self._prog("step", 40, step=2, step_name="Encode Labels")
        if self.should_stop():
            return None

        # Step 3: Embeddings
        self._log("Step 3/5 — Generating OpenAI embeddings...")
        self._prog("step", 40, step=3, step_name="Embeddings")
        prompts = [
            build_prompt(template, s, d)
            for s, d in zip(df["_supplier"].tolist(), df["_desc"].tolist())
        ]
        # Free the DataFrame now — prompts are built, df is no longer needed.
        del df
        gc.collect()

        X, tok_in = _do_embed(prompts, cfg["api_key"], cfg["batch_size"],
                               self.cb, self.should_stop,
                               model_idx=0, model_total=1, model_label="General")
        del prompts
        if X is None:
            return None
        self._log(f"Embeddings complete. Tokens used: {tok_in:,}", "success")
        self._prog("step", 60, step=3, step_name="Embeddings")
        if self.should_stop():
            return None

        # Save embedding checkpoint if requested.
        save_ckpt_path = cfg.get("save_ckpt_path", "")
        if save_ckpt_path:
            try:
                save_training_ckpt(
                    save_ckpt_path,
                    {"__general__": {
                        "X": X, "y": y, "label_map": label_map,
                        "naics_descs": naics_descriptions,
                        "prompt_template": template, "n_rows": len(X),
                    }},
                    {
                        "desc_col": cfg["desc_col"], "supplier_col": supplier_col,
                        "label_col": label_col, "naics_desc_col": cfg.get("naics_desc_col", ""),
                        "batch_size": cfg["batch_size"], "prompt_template": template,
                        "use_category": False, "category_col": "",
                    },
                    categories=[],
                )
                self._log(f"Embeddings checkpoint saved → {save_ckpt_path}", "success")
            except Exception as e:
                self._log(f"Could not save checkpoint: {e}", "warning")

        # Step 4: Train XGBoost
        # Build DMatrix first so we can immediately free the numpy arrays — XGBoost
        # copies the data internally, so X and y are no longer needed after this.
        self._log("Step 4/5 — Training XGBoost model...")
        self._prog("step", 60, step=4, step_name="Train XGBoost")
        dtrain = _make_dtrain(X, y)
        y_eval = y
        del X, y
        gc.collect()
        booster = self._train(dtrain, len(label_map), cfg,
                               model_idx=0, model_total=1, model_label="General")
        if booster is None:
            return None
        self._prog("step", 80, step=4, step_name="Train XGBoost")

        # Step 5: Evaluate — reuse the same DMatrix (no second copy of embeddings)
        self._log("Step 5/5 — Evaluating on training set...")
        self._prog("step", 80, step=5, step_name="Evaluate")
        preds  = booster.predict(dtrain)
        pred_lbl = np.argmax(preds, axis=1) if preds.ndim == 2 else preds.astype(int)
        acc = float(np.mean(pred_lbl == y_eval))
        del dtrain, y_eval, preds, pred_lbl
        self._log(f"Training accuracy: {acc * 100:.2f}%", "success")

        # Save
        _bundle_ts = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        bundle_path = os.path.join(model_dir, f"{model_name}_{_bundle_ts}.naics_model")
        training_cfg = {
            "desc_col":          cfg["desc_col"],
            "supplier_col":      supplier_col,
            "label_col":         label_col,
            "batch_size":        cfg["batch_size"],
            "max_depth":         cfg["max_depth"],
            "num_boost_round":   cfg["num_boost_round"],
            "training_accuracy": acc,
            "num_training_rows": n_rows,
            "trained_at":        ts,
        }
        bundle = ModelBundle(
            model_name=model_name,
            prompt_template=template,
            label_map=label_map,
            naics_descriptions=naics_descriptions,
            training_config=training_cfg,
            booster=booster,
            bundle_type="single",
        )
        bundle.save(bundle_path)
        self._log(f"Model saved → {bundle_path}", "success")
        self._prog("step", 100, step=5, step_name="Done")
        self._prog("done", 100, model_path=bundle_path, accuracy=acc)
        return bundle_path

    # ------------------------------------------------------------------
    def _run_multi(self, df, naics_descriptions, cfg, temp_dir, model_dir, model_name, ts, label_col, supplier_col) -> Optional[str]:
        """Train a general model + per-category models; save as multi-model bundle."""
        category_col     = cfg["category_col"]
        general_template = cfg["prompt_template"]
        n_total_rows     = len(df)  # capture before any deletion

        # ── Step 2: Encode labels (general + per-category) ────────────
        self._log("Step 2/5 — Encoding labels for all models...")
        self._prog("step", 20, step=2, step_name="Encode Labels")

        # Identify and validate categories FIRST (using the full df)
        raw_cats = [
            str(c).strip()
            for c in df[category_col].dropna().unique()
            if str(c).strip()
        ]

        cat_info: Dict[str, dict] = {}   # cat -> {df, le, y, label_map, template, naics_descs}
        valid_cats: List[str] = []

        for cat in sorted(raw_cats):
            mask   = df[category_col].astype(str).str.strip() == cat
            cat_df = df[mask]
            if len(cat_df) < 2:
                self._log(f"  Category '{cat}': only {len(cat_df)} row(s) — skipping.", "warning")
                continue
            unique_codes = cat_df[label_col].apply(_norm_code).unique()
            if len(unique_codes) < 2:
                self._log(
                    f"  Category '{cat}': only 1 unique NAICS class — "
                    "skipping (need ≥2 for classification).", "warning"
                )
                continue

            cat_le = LabelEncoder()
            y_cat  = cat_le.fit_transform(cat_df[label_col].apply(_norm_code))
            cat_lm = {int(i): _norm_code(c) for i, c in enumerate(cat_le.classes_)}

            # Extract per-category NAICS descriptions
            naics_desc_col = cfg.get("naics_desc_col", "")
            cat_naics_descs: Dict[str, str] = {}
            if naics_desc_col and naics_desc_col in cat_df.columns:
                pairs = cat_df[[label_col, naics_desc_col]].dropna().drop_duplicates()
                cat_naics_descs = {
                    _norm_code(r[label_col]): str(r[naics_desc_col])
                    for _, r in pairs.iterrows()
                }

            cat_info[cat] = {
                "df":          cat_df,
                "le":          cat_le,
                "y":           y_cat,
                "label_map":   cat_lm,
                "template":    _make_cat_prompt_template(cat),
                "naics_descs": cat_naics_descs,
            }
            valid_cats.append(cat)
            self._log(f"  Category '{cat}': {len(cat_df)} rows, {len(cat_lm)} classes.")

        if not valid_cats:
            self._log(
                "No valid categories found (each needs ≥2 rows and ≥2 NAICS classes). "
                "Training general model only.", "warning"
            )
            return self._run_single(df, naics_descriptions, cfg, temp_dir, model_dir, model_name, ts, label_col, supplier_col)

        # ── General model uses all rows ──
        df_general = df

        # Encode general model labels
        general_le = LabelEncoder()
        y_general  = general_le.fit_transform(df_general[label_col].apply(_norm_code))
        general_label_map: Dict[int, str] = {
            int(i): _norm_code(c) for i, c in enumerate(general_le.classes_)
        }
        self._log(
            f"General model: {len(general_label_map)} unique NAICS classes"
            f"  ({len(df_general):,} rows)."
        )

        n_models = 1 + len(valid_cats)
        self._log(f"Will train {n_models} models total:")
        self._log(f"  [1/{n_models}] General  ({len(df_general):,} rows, {len(general_label_map)} classes)")
        for i, cat in enumerate(valid_cats, 2):
            ci = cat_info[cat]
            self._log(f"  [{i}/{n_models}] Category '{cat}'  ({len(ci['df'])} rows, {len(ci['label_map'])} classes)")
        self._prog("step", 40, step=2, step_name="Encode Labels")
        if self.should_stop():
            return None

        # ── Step 3: Embeddings (general then each category) ───────────
        self._log("Step 3/5 — Generating embeddings for all models...")
        self._prog("step", 40, step=3, step_name="Embeddings")

        total_embed  = len(df_general) + sum(len(ci["df"]) for ci in cat_info.values())
        embed_offset = 0
        tok_total    = 0

        # Ordered list: (key, sub_df, template) — general first, then each category
        embed_order = [("__general__", df_general, general_template)] + [
            (cat, cat_info[cat]["df"], cat_info[cat]["template"])
            for cat in valid_cats
        ]

        # Precompute per-model batch counts for overall-batch tracking
        batch_size = cfg["batch_size"]
        model_batch_counts = [
            (len(sub_df) + batch_size - 1) // batch_size
            for _, sub_df, _ in embed_order
        ]
        overall_total_batches = sum(model_batch_counts)
        overall_batch_offset  = 0

        emb_paths: Dict[str, str] = {}       # key -> npz path
        emb_paths_n_rows: Dict[str, int] = {}  # key -> row count

        for idx, (key, sub_df, sub_tpl) in enumerate(embed_order):
            model_label = "General" if key == "__general__" else key
            self._log(f"  Embedding {model_label} ({len(sub_df)} rows)...")
            prompts = [
                build_prompt(sub_tpl, s, d)
                for s, d in zip(sub_df["_supplier"].tolist(), sub_df["_desc"].tolist())
            ]
            X, tok = _do_embed(
                prompts, cfg["api_key"], cfg["batch_size"],
                self.cb, self.should_stop,
                embed_offset=embed_offset,
                embed_total=total_embed,
                model_idx=idx,
                model_total=n_models,
                model_label=model_label,
                tok_offset=tok_total,
                overall_batch_offset=overall_batch_offset,
                overall_total_batches=overall_total_batches,
            )
            del prompts
            if X is None:
                return None
            tok_total   += tok
            embed_offset += len(sub_df)

            # Save to temp file (avoids holding all embeddings in RAM simultaneously).
            y_data = y_general if key == "__general__" else cat_info[key]["y"]
            npz_path = os.path.join(temp_dir, f"emb_{re.sub(r'[^a-z0-9]', '_', key.lower())}.npz")
            np.savez(npz_path, X=X, y=y_data)
            emb_paths[key] = npz_path
            emb_paths_n_rows[key] = len(y_data)
            overall_batch_offset += model_batch_counts[idx]
            del X  # free embedding array; it's now safely on disk

        # embed_order holds tuple references to df_general and all cat_info["df"]
        # sub-DataFrames.  Deleting it is what actually releases those references.
        del embed_order

        # Free the full DataFrames — embeddings are on disk, DataFrames are done.
        # Also strip the "df" key from cat_info so the sub-DataFrames are released
        # even if a reference to cat_info itself outlives this block.
        for _ci in cat_info.values():
            _ci.pop("df", None)
        del df, df_general
        gc.collect()

        self._log(f"All embeddings complete. Total tokens: {tok_total:,}", "success")
        self._prog("step", 60, step=3, step_name="Embeddings")
        if self.should_stop():
            return None

        # Save embedding checkpoint if requested — stream each pkl one at a time
        # to avoid loading all model arrays into memory simultaneously.
        save_ckpt_path = cfg.get("save_ckpt_path", "")
        if save_ckpt_path:
            try:
                models_meta_for_ckpt: Dict[str, dict] = {
                    "__general__": {
                        "label_map": general_label_map,
                        "naics_descs": naics_descriptions,
                        "prompt_template": general_template,
                        "n_rows": emb_paths_n_rows["__general__"],
                    },
                }
                for key in valid_cats:
                    ci = cat_info[key]
                    models_meta_for_ckpt[key] = {
                        "label_map": ci["label_map"],
                        "naics_descs": ci["naics_descs"],
                        "prompt_template": ci["template"],
                        "n_rows": emb_paths_n_rows[key],
                    }
                save_training_ckpt_from_npzs(
                    save_ckpt_path,
                    models_meta_for_ckpt,
                    emb_paths,
                    {
                        "desc_col": cfg["desc_col"], "supplier_col": supplier_col,
                        "label_col": label_col, "naics_desc_col": cfg.get("naics_desc_col", ""),
                        "batch_size": cfg["batch_size"], "prompt_template": general_template,
                        "use_category": True, "category_col": category_col,
                    },
                    categories=valid_cats,
                )
                self._log(f"Embeddings checkpoint saved → {save_ckpt_path}", "success")
            except Exception as e:
                self._log(f"Could not save checkpoint: {e}", "warning")

        # ── Step 4: Train XGBoost for all models ─────────────────────
        self._log("Step 4/5 — Training XGBoost models...")
        self._prog("step", 60, step=4, step_name="Train XGBoost")

        total_rounds = cfg["num_boost_round"] * n_models
        round_offset = 0
        trained: Dict[str, dict] = {}       # key -> {booster, label_map, acc}
        trained_n_rows: Dict[str, int] = {} # key -> actual training row count

        train_order = [("__general__", general_label_map)] + [
            (cat, cat_info[cat]["label_map"]) for cat in valid_cats
        ]

        for idx, (key, lm) in enumerate(train_order):
            model_label = "General" if key == "__general__" else key
            self._log(f"  Training {model_label} ({idx + 1}/{n_models})...")
            with np.load(emb_paths[key], allow_pickle=False) as data:
                X = data["X"].copy()
                y = data["y"].copy()

            trained_n_rows[key] = len(y)

            # Build QuantileDMatrix then immediately free the numpy arrays.
            # QuantileDMatrix stores uint8 bin indices (~307 MB for 100k×3072) instead
            # of raw float32 data (~1.2 GB), saving ~900 MB for the duration of training.
            dtrain = _make_dtrain(X, y)
            y_eval = y
            del X, y
            gc.collect()

            booster = self._train(dtrain, len(lm), cfg,
                                  train_offset=round_offset,
                                  train_total=total_rounds,
                                  model_idx=idx,
                                  model_total=n_models,
                                  model_label=model_label)
            if booster is None:
                return None

            preds    = booster.predict(dtrain)
            pred_lbl = np.argmax(preds, axis=1) if preds.ndim == 2 else preds.astype(int)
            acc = float(np.mean(pred_lbl == y_eval))
            self._log(f"    {model_label} training accuracy: {acc * 100:.2f}%", "success")
            trained[key] = {"booster": booster, "label_map": lm, "acc": acc}
            round_offset += cfg["num_boost_round"]
            del dtrain, y_eval, preds, pred_lbl

        self._prog("step", 80, step=4, step_name="Train XGBoost")

        # ── Step 5: Assemble & save ───────────────────────────────────
        self._log("Step 5/5 — Assembling multi-model bundle...")
        self._prog("step", 80, step=5, step_name="Evaluate")

        gen = trained["__general__"]
        gen_training_cfg = {
            "desc_col":            cfg["desc_col"],
            "supplier_col":        supplier_col,
            "label_col":           label_col,
            "category_col":        category_col,
            "batch_size":          cfg["batch_size"],
            "max_depth":           cfg["max_depth"],
            "num_boost_round":     cfg["num_boost_round"],
            "training_accuracy":   gen["acc"],
            "num_training_rows":   trained_n_rows.get("__general__", emb_paths_n_rows["__general__"]),
            "num_total_rows":      n_total_rows,
            "trained_at":          ts,
            "num_category_models": len(valid_cats),
        }

        category_bundles: Dict[str, ModelBundle] = {}
        for cat in valid_cats:
            td = trained[cat]
            ci = cat_info[cat]
            cat_training_cfg = {
                "desc_col":          cfg["desc_col"],
                "supplier_col":      supplier_col,
                "label_col":         label_col,
                "category_col":      category_col,
                "category":          cat,
                "batch_size":        cfg["batch_size"],
                "max_depth":         cfg["max_depth"],
                "num_boost_round":   cfg["num_boost_round"],
                "training_accuracy": td["acc"],
                "num_training_rows": trained_n_rows.get(cat, emb_paths_n_rows.get(cat, 0)),
                "trained_at":        ts,
            }
            category_bundles[cat] = ModelBundle(
                model_name=f"{model_name}_{cat}",
                prompt_template=ci["template"],
                label_map=td["label_map"],
                naics_descriptions=ci["naics_descs"],
                training_config=cat_training_cfg,
                booster=td["booster"],
                bundle_type="single",
            )

        _bundle_ts  = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        bundle_path = os.path.join(model_dir, f"{model_name}_{_bundle_ts}.naics_model")
        bundle = ModelBundle(
            model_name=model_name,
            prompt_template=general_template,
            label_map=gen["label_map"],
            naics_descriptions=naics_descriptions,
            training_config=gen_training_cfg,
            booster=gen["booster"],
            bundle_type="multi",
            category_bundles=category_bundles,
        )
        bundle.save(bundle_path)
        self._log(
            f"Multi-model bundle saved → {bundle_path}  "
            f"(1 general + {len(valid_cats)} category models)", "success"
        )
        self._prog("step", 100, step=5, step_name="Done")
        self._prog("done", 100, model_path=bundle_path, accuracy=gen["acc"])
        return bundle_path

    # ------------------------------------------------------------------
    def _load_data(self, cfg) -> Optional[pd.DataFrame]:
        path = cfg["input_file"]
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path, sheet_name=cfg.get("sheet_name", 0))
            else:
                try:
                    df = pd.read_csv(path, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(path, encoding="latin-1")
        except Exception as e:
            self._log(f"Cannot load file: {e}", "error")
            return None

        for key, human in [("desc_col", "Description"), ("label_col", "NAICS Code")]:
            col = cfg.get(key, "")
            if not col:
                self._log(f"{human} column not set.", "error")
                return None
            if col not in df.columns:
                self._log(f"Column '{col}' not found in file.", "error")
                return None

        # Drop rows with missing NAICS labels to avoid a spurious "nan" class
        label_col = cfg["label_col"]
        before = len(df)
        df = df.dropna(subset=[label_col]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            self._log(
                f"Dropped {dropped:,} row(s) with missing NAICS label — "
                f"{len(df):,} rows remain.", "warning"
            )
        if len(df) == 0:
            self._log("No rows with a valid NAICS label. Cannot train.", "error")
            return None

        return df

    def _train(
        self, dtrain: xgb.DMatrix, num_classes: int, cfg: dict,
        train_offset: int = 0, train_total: Optional[int] = None,
        model_idx: int = 0, model_total: int = 1, model_label: str = "General",
    ) -> Optional[xgb.Booster]:
        params = {
            "objective":   "multi:softprob",
            "num_class":   num_classes,
            "eval_metric": ["mlogloss", "merror"],
            "tree_method": "hist",
            "eta":         0.05,
            "max_depth":   cfg["max_depth"],
            "verbosity":   0,
        }
        num_rounds = cfg["num_boost_round"]
        if train_total is None:
            train_total = num_rounds

        # Explain expected memory to the user so the numbers aren't surprising.
        # Gradient+hessian buffer (n_rows × n_classes × 2 × 4 B) is allocated once
        # and held for the entire run.  Tree storage grows ~(n_classes × ~2 KB) per round.
        n_rows_train = dtrain.num_row()
        grad_mb      = n_rows_train * num_classes * 2 * 4 / 1024 ** 2
        tree_mb_rnd  = num_classes * 2 / 1024          # ~2 KB per tree × n_classes
        total_tree_mb = tree_mb_rnd * num_rounds
        self._log(
            f"  [{model_label}] {n_rows_train:,} rows × {num_classes} classes — "
            f"gradient buffer ~{grad_mb:.0f} MB (constant); "
            f"tree storage ~{tree_mb_rnd:.1f} MB/round × {num_rounds} rounds "
            f"= ~{total_tree_mb:.0f} MB growth over full run."
        )

        _self = self

        class _Callback(xgb.callback.TrainingCallback):
            def before_training(self, model):
                self._t0 = time.time()
                return model

            def after_iteration(self, model, epoch, evals_log):
                if _self.should_stop():
                    return True
                if _self.should_skip():
                    _self._log(
                        f"Skipping remaining rounds for {model_label} "
                        f"({model_idx + 1}/{model_total}) — saving partial model.",
                        "warning",
                    )
                    return True
                global_round = train_offset + epoch + 1
                pct  = global_round / train_total * 100
                logs = {}
                for ds, ds_m in evals_log.items():
                    for mname, vals in ds_m.items():
                        logs[f"{ds}-{mname}"] = vals[-1] if vals else 0.0

                loss  = logs.get("train-mlogloss", logs.get("mlogloss", 0.0))
                error = logs.get("train-merror",   logs.get("merror",   0.0))
                acc   = (1.0 - error) * 100

                model_round = epoch + 1
                model_pct   = model_round / num_rounds * 100
                elapsed     = time.time() - self._t0
                rate        = model_round / elapsed if elapsed > 0 else 0
                eta_model   = _format_eta((num_rounds - model_round) / rate) if rate > 0 else "—"
                eta_overall = _format_eta((train_total - global_round) / rate) if rate > 0 else "—"

                _self._prog(
                    "training", pct,
                    round=global_round, total_rounds=train_total,
                    model_round=model_round, model_total_rounds=num_rounds, model_pct=model_pct,
                    loss=loss, error=error, accuracy=acc,
                    eta_model=eta_model, eta_overall=eta_overall,
                    model_idx=model_idx, model_total=model_total, model_label=model_label,
                )
                _self._log(
                    f"[{model_label} {model_idx + 1}/{model_total}] "
                    f"Round {model_round}/{num_rounds}  (Overall: {global_round}/{train_total}) | "
                    f"Loss: {loss:.4f} | Acc: {acc:.2f}% | "
                    f"Model ETA: {eta_model} | Overall ETA: {eta_overall}"
                )
                return False

        try:
            booster = xgb.train(
                params, dtrain,
                num_boost_round=num_rounds,
                evals=[(dtrain, "train")],
                callbacks=[_Callback()],
                verbose_eval=False,
            )
            # Full stop → abort the entire run; skip → save the partial model
            if self.should_stop():
                return None
            return booster
        except Exception as e:
            self._log(f"XGBoost training error: {e}", "error")
            return None


# ---------------------------------------------------------------------------
# Prediction Worker
# ---------------------------------------------------------------------------

class PredictWorker:
    def __init__(self, config: dict, progress_cb: Callable, stop_checker: Callable[[], bool],
                 skip_checker=None):
        self.config = config
        self.cb = progress_cb
        self.should_stop = stop_checker

    def _log(self, msg: str, level: str = "info"):
        self.cb({"type": "log", "message": msg, "level": level})

    def _prog(self, stage: str, pct: float, **kw):
        self.cb({"type": "progress", "stage": stage, "pct": pct, **kw})

    def run(self) -> Optional[str]:
        cfg = self.config

        # Load model
        self._log(f"Loading model: {Path(cfg['model_path']).name}")
        try:
            bundle = ModelBundle.load(cfg["model_path"])
        except Exception as e:
            self._log(f"Cannot load model: {e}", "error")
            return None

        self._log(
            f"Model ready — {bundle.model_name} | "
            f"{'multi-model (' + str(len(bundle.categories)) + ' categories + general)' if bundle.is_multi else str(bundle.num_classes) + ' classes'} | "
            f"created {bundle.created_at}",
            "success",
        )

        try:
            # ── Checkpoint bypass: skip input file loading and embedding ──
            if cfg.get("load_ckpt_path"):
                return self._predict_from_ckpt(bundle, cfg)

            # Load input file
            path = cfg["input_file"]
            try:
                if path.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(path, sheet_name=cfg.get("sheet_name", 0))
                else:
                    try:
                        df = pd.read_csv(path, encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        df = pd.read_csv(path, encoding="latin-1")
            except Exception as e:
                self._log(f"Cannot load input file: {e}", "error")
                return None

            self._log(f"Loaded {len(df):,} rows.")
            if len(df) == 0:
                self._log("Input file is empty — no rows to predict.", "error")
                return None

            desc_col     = cfg["desc_col"]
            supplier_col = cfg.get("supplier_col", "")
            category_col = cfg.get("category_col", "")

            if desc_col not in df.columns:
                self._log(
                    f"Description column '{desc_col}' not found in file. "
                    f"Available: {', '.join(df.columns.tolist())}", "error"
                )
                return None

            df["_desc"]     = df[desc_col].apply(clean_text)
            df["_supplier"] = (
                df[supplier_col].apply(clean_text)
                if supplier_col and supplier_col in df.columns else ""
            )

            topk       = cfg.get("topk", 3)
            batch_size = cfg.get("batch_size", 200)
            api_key    = cfg["api_key"]

            # Route to multi or single prediction
            use_cat_routing = (
                bundle.is_multi
                and bool(category_col)
                and category_col in df.columns
            )

            if bundle.is_multi and not use_cat_routing:
                self._log(
                    "Multi-model bundle loaded but no category column selected — "
                    "using general model for all rows.", "warning"
                )

            if use_cat_routing:
                result = self._predict_multi(df, bundle, category_col, api_key, batch_size, topk)
            else:
                result = self._predict_single(df, bundle, api_key, batch_size, topk, add_model_used=bundle.is_multi)
            if result is None:
                return None

            df = result

            # Clean up helper columns
            df.drop(columns=["_desc", "_supplier"], inplace=True, errors="ignore")

            out_path = cfg.get("output_file", "")
            if not out_path:
                base = Path(cfg["input_file"])
                ts_pred = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                out_path = str(base.parent / f"{base.stem}_with_LLM_NAICS_Prediction_{ts_pred}.csv")

            df.to_csv(out_path, index=False)
            self._log(f"Output saved → {out_path}", "success")
            self._prog("done", 100, output_path=out_path)
            return out_path

        except Exception as e:
            self._log(f"Unexpected error: {e}", "error")
            self._log(traceback.format_exc(), "error")
            return None

    # ------------------------------------------------------------------
    def _predict_from_ckpt(self, bundle: ModelBundle, cfg: dict) -> Optional[str]:
        """Predict using pre-computed embeddings from a .naics_embed checkpoint."""
        load_path = cfg["load_ckpt_path"]
        self._log(f"Loading prediction checkpoint: {Path(load_path).name}")
        try:
            ckpt = load_prediction_ckpt(load_path)
        except Exception as e:
            self._log(f"Cannot load checkpoint: {e}", "error")
            return None

        df     = ckpt["df"]
        X_flat = ckpt["X_flat"]
        # Drop prediction columns from the prior run so stale ML_pred* columns
        # don't survive into the output if topk or model changed between runs.
        _stale = [c for c in df.columns if c.startswith("ML_pred") or c == "ML_model_used"]
        if _stale:
            df = df.drop(columns=_stale)
        n      = len(df)
        self._log(f"Checkpoint: {n:,} rows, {X_flat.shape[1]} dims", "success")
        self._prog("predicting", 5)

        topk         = cfg.get("topk", 3)
        category_col = cfg.get("category_col", "")
        use_cat = bundle.is_multi and bool(category_col) and category_col in df.columns

        if bundle.is_multi and not use_cat:
            self._log(
                "Multi-model bundle loaded but no category column in checkpoint — "
                "using general model for all rows.", "warning"
            )

        if use_cat:
            result = self._predict_multi_from_X(df, bundle, category_col, X_flat, topk)
        else:
            result = self._predict_single_from_X(df, bundle, X_flat, topk,
                                                  add_model_used=bundle.is_multi)
        if result is None:
            return None

        df = result
        df.drop(columns=["_desc", "_supplier"], inplace=True, errors="ignore")

        out_path = cfg.get("output_file", "")
        if not out_path:
            _ts = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            out_path = str(
                Path(load_path).parent /
                f"{Path(load_path).stem}_with_LLM_NAICS_Prediction_{_ts}.csv"
            )

        df.to_csv(out_path, index=False)
        self._log(f"Output saved → {out_path}", "success")
        self._prog("done", 100, output_path=out_path)
        return out_path

    # ------------------------------------------------------------------
    def _predict_single_from_X(
        self, df: pd.DataFrame, bundle: ModelBundle, X: np.ndarray, topk: int,
        add_model_used: bool = False, model_used_label: str = "general",
    ) -> Optional[pd.DataFrame]:
        """Predict from pre-computed embeddings (no API calls)."""
        n = len(df)
        self._log("Running XGBoost predictions...")
        self._prog("predicting", 50)
        dtest = xgb.DMatrix(X)
        proba = bundle.booster.predict(dtest)
        self._prog("predicting", 90)

        actual_topk = min(topk, proba.shape[1])
        top_indices = np.argsort(proba, axis=1)[:, ::-1]
        df = df.copy()
        for k in range(1, actual_topk + 1):
            col_idx    = top_indices[:, k - 1]
            confidence = proba[np.arange(n), col_idx]
            codes = [bundle.label_map.get(int(idx), str(idx)) for idx in col_idx]
            descs = [bundle.naics_descriptions.get(str(c), "") for c in codes]
            df[f"ML_pred{k}_NAICS"]       = codes
            df[f"ML_pred{k}_confidence"]  = np.round(confidence, 4)
            df[f"ML_pred{k}_description"] = descs

        if add_model_used:
            df["ML_model_used"] = model_used_label
        return df

    # ------------------------------------------------------------------
    def _predict_multi_from_X(
        self, df: pd.DataFrame, bundle: ModelBundle,
        category_col: str, X_flat: np.ndarray, topk: int,
    ) -> Optional[pd.DataFrame]:
        """Predict multi-model routing from pre-computed embeddings."""
        n          = len(df)
        cat_series = df[category_col].fillna("").astype(str).str.strip()

        _cat_lookup   = {k.lower(): k for k in bundle.categories}
        routing_key   = [""] * n
        model_used_lbl = [""] * n
        for i, cat_val in enumerate(cat_series):
            matched = _cat_lookup.get(cat_val.lower())
            if matched is not None:
                routing_key[i]    = matched
                model_used_lbl[i] = f"category:{matched}"
            elif cat_val == "":
                model_used_lbl[i] = "general (fallback — no category)"
            else:
                routing_key[i] = ""
                model_used_lbl[i] = f"general (fallback — unknown category: {cat_val})"

        groups: Dict[str, List[int]] = {}
        for i, rkey in enumerate(routing_key):
            groups.setdefault(rkey, []).append(i)

        self._log(
            f"Category routing: {len(groups)} group(s) "
            f"({', '.join(str(len(v)) + ' rows → ' + ('general' if k == '' else k) for k, v in sorted(groups.items()))})"
        )

        for k in range(1, topk + 1):
            df[f"ML_pred{k}_NAICS"]       = ""
            df[f"ML_pred{k}_confidence"]  = 0.0
            df[f"ML_pred{k}_description"] = ""
        df["ML_model_used"] = model_used_lbl

        sorted_groups = sorted(groups.items(), key=lambda x: (x[0] == "", x[0]))
        n_groups = len(sorted_groups)

        for g_idx, (rkey, row_idxs) in enumerate(sorted_groups):
            sub_bundle  = bundle._category_bundles[rkey] if rkey else bundle
            group_label = f"category '{rkey}'" if rkey else "general model"
            self._log(f"Predicting {group_label} ({len(row_idxs)} rows)...")
            self._prog("predicting", int(g_idx / n_groups * 90))

            X_group = X_flat[np.array(row_idxs)]
            dtest   = xgb.DMatrix(X_group)
            proba   = sub_bundle.booster.predict(dtest)
            top_idx = np.argsort(proba, axis=1)[:, ::-1]
            actual_topk = min(topk, proba.shape[1])

            for j, orig_i in enumerate(row_idxs):
                for k in range(1, actual_topk + 1):
                    ci   = top_idx[j, k - 1]
                    conf = float(proba[j, ci])
                    code = sub_bundle.label_map.get(int(ci), str(ci))
                    desc = sub_bundle.naics_descriptions.get(str(code), "")
                    df.at[df.index[orig_i], f"ML_pred{k}_NAICS"]       = code
                    df.at[df.index[orig_i], f"ML_pred{k}_confidence"]  = round(conf, 4)
                    df.at[df.index[orig_i], f"ML_pred{k}_description"] = desc

        self._prog("predicting", 100)
        return df

    # ------------------------------------------------------------------
    def _predict_single(
        self, df: pd.DataFrame, bundle: ModelBundle,
        api_key: str, batch_size: int, topk: int,
        add_model_used: bool = False,
        model_used_label: str = "general",
        embed_offset: int = 0,
        embed_total: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Embed and predict using a single (or general) bundle."""
        template = bundle.prompt_template
        self._log(f"Prompt template: {template}")

        n = len(df)
        if embed_total is None:
            embed_total = n

        prompts = [
            build_prompt(template, s, d)
            for s, d in zip(df["_supplier"].tolist(), df["_desc"].tolist())
        ]

        self._log(f"Generating embeddings ({n} rows)...")
        X, tok_in = _do_embed(
            prompts, api_key, batch_size, self.cb, self.should_stop,
            embed_offset=embed_offset, embed_total=embed_total,
        )
        if X is None:
            return None
        self._log(f"Embeddings complete. Tokens: {tok_in:,}", "success")

        # Save prediction checkpoint if requested
        save_ckpt_path = self.config.get("save_ckpt_path", "")
        if save_ckpt_path:
            try:
                cfg_snap = {
                    "desc_col":    self.config.get("desc_col", ""),
                    "supplier_col": self.config.get("supplier_col", ""),
                    "batch_size":  self.config.get("batch_size", 200),
                    "bundle_type": "single",
                }
                save_prediction_ckpt(save_ckpt_path, df, X, cfg_snap)
                self._log(f"Prediction checkpoint saved → {save_ckpt_path}", "success")
            except Exception as e:
                self._log(f"Could not save checkpoint: {e}", "warning")

        return self._predict_single_from_X(df, bundle, X, topk, add_model_used, model_used_label)

    # ------------------------------------------------------------------
    def _predict_multi(
        self, df: pd.DataFrame, bundle: ModelBundle,
        category_col: str, api_key: str, batch_size: int, topk: int,
    ) -> Optional[pd.DataFrame]:
        """
        Route each row to its category sub-model (or general as fallback),
        embed each group separately (different prompt templates), predict,
        reassemble in original order, and append ML_model_used column.
        """
        n         = len(df)
        cat_series = df[category_col].fillna("").astype(str).str.strip()

        # Build routing: original row index → (routing_key, model_used_label)
        _cat_lookup   = {k.lower(): k for k in bundle.categories}
        routing_key   = [""] * n   # "" = general
        model_used_lbl = [""] * n

        for i, cat_val in enumerate(cat_series):
            matched = _cat_lookup.get(cat_val.lower())
            if matched is not None:
                routing_key[i]    = matched
                model_used_lbl[i] = f"category:{matched}"
            else:
                routing_key[i] = ""  # general fallback
                if cat_val == "":
                    model_used_lbl[i] = "general (fallback — no category)"
                else:
                    model_used_lbl[i] = f"general (fallback — unknown category: {cat_val})"

        # Group row indices by routing key
        groups: Dict[str, List[int]] = {}
        for i, rkey in enumerate(routing_key):
            groups.setdefault(rkey, []).append(i)

        self._log(
            f"Category routing: {len(groups)} group(s) "
            f"({', '.join(str(len(v)) + ' rows → ' + ('general' if k == '' else k) for k, v in sorted(groups.items()))})"
        )

        # Total embedding rows = n (each row embedded exactly once)
        embed_offset = 0

        # Pre-allocate X_flat if checkpoint saving is requested
        save_ckpt_path = self.config.get("save_ckpt_path", "")
        X_flat = np.empty((n, _EMBED_DIM), dtype=np.float32) if save_ckpt_path else None

        # Initialise result columns
        for k in range(1, topk + 1):
            df[f"ML_pred{k}_NAICS"]       = ""
            df[f"ML_pred{k}_confidence"]  = 0.0
            df[f"ML_pred{k}_description"] = ""
        df["ML_model_used"] = model_used_lbl

        # Process groups (general last so progress ends at 100 %)
        sorted_groups = sorted(groups.items(), key=lambda x: (x[0] == "", x[0]))
        n_groups = len(sorted_groups)

        # Precompute per-group batch counts for overall-batch tracking
        overall_total_batches = sum(
            (len(groups[rkey]) + batch_size - 1) // batch_size
            for rkey in (g[0] for g in sorted_groups)
        )
        overall_batch_offset = 0

        tok_total = 0

        for g_idx, (rkey, row_idxs) in enumerate(sorted_groups):
            sub_bundle = bundle._category_bundles[rkey] if rkey else bundle
            group_label = f"category '{rkey}'" if rkey else "general model"
            model_label = group_label
            self._log(f"Processing {group_label} ({len(row_idxs)} rows)...")

            sub_desc     = [df.iloc[i]["_desc"]     for i in row_idxs]
            sub_supplier = [df.iloc[i]["_supplier"] for i in row_idxs]
            prompts = [
                build_prompt(sub_bundle.prompt_template, s, d)
                for s, d in zip(sub_supplier, sub_desc)
            ]
            del sub_desc, sub_supplier

            group_batch_count = (len(row_idxs) + batch_size - 1) // batch_size

            X, tok = _do_embed(
                prompts, api_key, batch_size, self.cb, self.should_stop,
                embed_offset=embed_offset, embed_total=n,
                model_idx=g_idx, model_total=n_groups, model_label=model_label,
                tok_offset=tok_total,
                overall_batch_offset=overall_batch_offset,
                overall_total_batches=overall_total_batches,
            )
            del prompts
            if X is None:
                return None

            tok_total += tok

            if X_flat is not None:
                X_flat[np.array(row_idxs)] = X

            embed_offset += len(row_idxs)
            overall_batch_offset += group_batch_count

            self._log(f"Predicting for {group_label}...")
            dtest = xgb.DMatrix(X)
            del X  # DMatrix has its own copy
            proba = sub_bundle.booster.predict(dtest)
            del dtest
            top_idx = np.argsort(proba, axis=1)[:, ::-1]
            actual_topk = min(topk, proba.shape[1])

            for j, orig_i in enumerate(row_idxs):
                for k in range(1, actual_topk + 1):
                    ci   = top_idx[j, k - 1]
                    conf = float(proba[j, ci])
                    code = sub_bundle.label_map.get(int(ci), str(ci))
                    desc = sub_bundle.naics_descriptions.get(str(code), "")
                    df.at[df.index[orig_i], f"ML_pred{k}_NAICS"]       = code
                    df.at[df.index[orig_i], f"ML_pred{k}_confidence"]  = round(conf, 4)
                    df.at[df.index[orig_i], f"ML_pred{k}_description"] = desc
            del proba, top_idx

            self._prog("predicting", embed_offset / n * 80)

        if X_flat is not None:
            try:
                cfg_snap = {
                    "desc_col":    self.config.get("desc_col", ""),
                    "supplier_col": self.config.get("supplier_col", ""),
                    "batch_size":  self.config.get("batch_size", 200),
                    "bundle_type": "multi",
                    "category_col": category_col,
                }
                save_prediction_ckpt(save_ckpt_path, df, X_flat, cfg_snap)
                self._log(f"Prediction checkpoint saved → {save_ckpt_path}", "success")
            except Exception as e:
                self._log(f"Could not save checkpoint: {e}", "warning")

        self._prog("predicting", 100)
        return df
