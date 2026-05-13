"""
model_bundle.py - Portable single-file model format for NAICS Classifier.

A .naics_model file is a ZIP archive containing:
  bundle.json         — metadata, label maps, prompt templates, training configs
  model.ubj           — XGBoost general/single model (UBJ binary)
  model_cat_N.ubj     — XGBoost category-specific models (multi bundles only)

Bundle types
------------
"single"  Original format: one general model.  Fully backward-compatible.
"multi"   One general model + N category-specific sub-models.
          The general model fields are the same as "single".
          _category_bundles holds the per-category ModelBundle instances.
          Use get_bundle_for_category() to route to the right sub-model.
"""

import json
import zipfile
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Optional

import xgboost as xgb


def _cat_filename(index: int) -> str:
    return f"model_cat_{index}.ubj"


class ModelBundle:
    """
    Encapsulates everything needed to embed and predict NAICS codes.

    For bundle_type="single" (default, backward-compatible):
        prompt_template, label_map, naics_descriptions, training_config, booster
        are used directly for all predictions.

    For bundle_type="multi":
        The same fields hold the GENERAL model (trained on all data).
        _category_bundles maps category strings to per-category ModelBundle
        instances (each is bundle_type="single" internally).
        Use get_bundle_for_category(cat) to dispatch predictions.
    """

    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        label_map: Dict[int, str],
        naics_descriptions: Dict[str, str],
        training_config: dict,
        created_at: Optional[str] = None,
        booster: Optional[xgb.Booster] = None,
        bundle_type: str = "single",
        category_bundles: Optional[Dict[str, "ModelBundle"]] = None,
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.label_map = {int(k): str(v) for k, v in label_map.items()}
        self.naics_descriptions = {str(k): str(v) for k, v in naics_descriptions.items()}
        self.training_config = training_config
        self.created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._booster = booster
        self.bundle_type = bundle_type
        self._category_bundles: Dict[str, "ModelBundle"] = category_bundles or {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def booster(self) -> Optional[xgb.Booster]:
        return self._booster

    @booster.setter
    def booster(self, b: xgb.Booster):
        self._booster = b

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def is_multi(self) -> bool:
        return self.bundle_type == "multi"

    @property
    def categories(self) -> List[str]:
        return list(self._category_bundles.keys()) if self.is_multi else []

    def get_bundle_for_category(self, category: str) -> "ModelBundle":
        """Return the sub-model for *category*, or self (general) if not found."""
        if not self.is_multi:
            return self
        cat_str = str(category).strip()
        if cat_str in self._category_bundles:
            return self._category_bundles[cat_str]
        # Case-insensitive fallback
        cat_lower = cat_str.lower()
        for k, v in self._category_bundles.items():
            if k.lower() == cat_lower:
                return v
        return self  # general model fallback

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save bundle to a .naics_model ZIP archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # ── Build metadata ────────────────────────────────────────
            meta: dict = {
                "bundle_type": self.bundle_type,
                "model_name":  self.model_name,
                "created_at":  self.created_at,
            }

            def _model_meta(b: "ModelBundle") -> dict:
                return {
                    "prompt_template":   b.prompt_template,
                    "label_map":         {str(k): v for k, v in b.label_map.items()},
                    "naics_descriptions": b.naics_descriptions,
                    "training_config":   b.training_config,
                }

            if self.bundle_type == "single":
                meta.update(_model_meta(self))
            else:  # multi
                meta["general_model"] = _model_meta(self)
                cats = list(self._category_bundles.keys())
                meta["categories"] = cats
                meta["category_models"] = {
                    cat: _model_meta(cb)
                    for cat, cb in self._category_bundles.items()
                }
                meta["category_file_map"] = {
                    cat: _cat_filename(i)
                    for i, cat in enumerate(cats)
                }

            # ── Write bundle.json ─────────────────────────────────────
            meta_path = os.path.join(tmpdir, "bundle.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # ── Write model files ─────────────────────────────────────
            gen_path = os.path.join(tmpdir, "model.ubj")
            if self._booster is not None:
                self._booster.save_model(gen_path)

            if self.bundle_type == "multi":
                for i, (cat, cb) in enumerate(self._category_bundles.items()):
                    if cb._booster is not None:
                        cp = os.path.join(tmpdir, _cat_filename(i))
                        cb._booster.save_model(cp)

            # ── Pack ZIP ──────────────────────────────────────────────
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(meta_path, "bundle.json")
                if os.path.exists(gen_path):
                    zf.write(gen_path, "model.ubj")
                if self.bundle_type == "multi":
                    for i, cat in enumerate(self._category_bundles.keys()):
                        cp = os.path.join(tmpdir, _cat_filename(i))
                        if os.path.exists(cp):
                            zf.write(cp, _cat_filename(i))

    @classmethod
    def load(cls, path: str) -> "ModelBundle":
        """Load a bundle (single or multi) from a .naics_model ZIP."""
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("bundle.json") as f:
                meta = json.load(f)

            bundle_type = meta.get("bundle_type", "single")
            namelist = zf.namelist()

            def _load_booster(fname: str) -> Optional[xgb.Booster]:
                if fname not in namelist:
                    return None
                # Write to a fixed safe path to avoid any path-traversal in fname.
                with tempfile.TemporaryDirectory() as td:
                    safe_path = os.path.join(td, "model.ubj")
                    with zf.open(fname) as src, open(safe_path, "wb") as dst:
                        dst.write(src.read())
                    b = xgb.Booster()
                    b.load_model(safe_path)
                    return b

            if bundle_type == "single":
                label_map = {int(k): v for k, v in meta["label_map"].items()}
                return cls(
                    model_name=meta["model_name"],
                    prompt_template=meta["prompt_template"],
                    label_map=label_map,
                    naics_descriptions=meta.get("naics_descriptions", {}),
                    training_config=meta.get("training_config", {}),
                    created_at=meta.get("created_at", ""),
                    booster=_load_booster("model.ubj"),
                    bundle_type="single",
                )

            # multi bundle
            general = meta["general_model"]
            general_label_map = {int(k): v for k, v in general["label_map"].items()}

            file_map = meta.get("category_file_map", {})
            cat_models_meta = meta.get("category_models", {})
            category_bundles: Dict[str, "ModelBundle"] = {}

            for cat, fname in file_map.items():
                cm = cat_models_meta.get(cat, {})
                cat_lm = {int(k): v for k, v in cm.get("label_map", {}).items()}
                category_bundles[cat] = cls(
                    model_name=f"{meta['model_name']}_cat_{cat}",
                    prompt_template=cm.get("prompt_template", ""),
                    label_map=cat_lm,
                    naics_descriptions=cm.get("naics_descriptions", {}),
                    training_config=cm.get("training_config", {}),
                    created_at=meta.get("created_at", ""),
                    booster=_load_booster(fname),
                    bundle_type="single",
                )

            return cls(
                model_name=meta["model_name"],
                prompt_template=general.get("prompt_template", ""),
                label_map=general_label_map,
                naics_descriptions=general.get("naics_descriptions", {}),
                training_config=general.get("training_config", {}),
                created_at=meta.get("created_at", ""),
                booster=_load_booster("model.ubj"),
                bundle_type="multi",
                category_bundles=category_bundles,
            )

    @staticmethod
    def peek(path: str) -> dict:
        """Read metadata only — does NOT load XGBoost model(s)."""
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("bundle.json") as f:
                meta = json.load(f)

        bundle_type = meta.get("bundle_type", "single")

        if bundle_type == "single":
            return {
                "bundle_type":    "single",
                "model_name":     meta["model_name"],
                "created_at":     meta.get("created_at", ""),
                "num_classes":    len(meta["label_map"]),
                "prompt_template": meta["prompt_template"],
                "training_config": meta.get("training_config", {}),
                "categories":     [],
                "category_models": {},
            }

        # multi
        general = meta.get("general_model", {})
        categories = meta.get("categories", [])
        cat_models_meta = meta.get("category_models", {})
        return {
            "bundle_type":    "multi",
            "model_name":     meta["model_name"],
            "created_at":     meta.get("created_at", ""),
            "num_classes":    len(general.get("label_map", {})),
            "prompt_template": general.get("prompt_template", ""),
            "training_config": general.get("training_config", {}),
            "categories":     categories,
            "category_models": {
                cat: {
                    "num_classes":    len(cat_models_meta.get(cat, {}).get("label_map", {})),
                    "prompt_template": cat_models_meta.get(cat, {}).get("prompt_template", ""),
                    "training_config": cat_models_meta.get(cat, {}).get("training_config", {}),
                }
                for cat in categories
                if cat in cat_models_meta
            },
        }
