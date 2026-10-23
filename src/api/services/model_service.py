"""
services/model_service.py
==========================
Loads the trained regression pipeline (preprocessing + model bundled
together by notebooks/05_Regression_Model.ipynb) and the metadata that
describes it. This is the ONLY place the model artifact is touched —
no preprocessing is ever re-implemented here, per project rules.
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.api.core.config import Settings, get_settings
from src.api.schemas import MeasurementFeatures

logger = logging.getLogger("api.model_service")


class ModelNotAvailableError(RuntimeError):
    """Raised when the model artifact or its metadata cannot be loaded."""


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline: Any = None
        self.metadata: dict = {}
        self._load()

    def _load(self) -> None:
        if not self.settings.model_path.exists():
            raise ModelNotAvailableError(
                f"Model artifact not found at {self.settings.model_path}. "
                f"Run notebooks/05_Regression_Model.ipynb to train and save it first."
            )
        if not self.settings.model_metadata_path.exists():
            raise ModelNotAvailableError(
                f"Model metadata not found at {self.settings.model_metadata_path}."
            )

        self.pipeline = joblib.load(self.settings.model_path)
        import json

        with open(self.settings.model_metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(
            "Loaded model '%s' version %s from %s",
            self.metadata.get("model_name"),
            self.metadata.get("model_version"),
            self.settings.model_path,
        )

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def feature_order(self) -> list[str]:
        """Source of truth for column order at inference time.

        IMPORTANT (bug fixed 2026): this used to return metadata.json's
        "feature_order" list directly. That list is documentation, written/
        regenerated separately from the trained artifact, and can silently
        drift out of sync with the order the pipeline was actually fit on
        (this happened in practice: metadata.json listed
        [..., moving_average_5, rolling_median_3, min, ...] while the fitted
        imputer's feature_names_in_ was [..., min, moving_average_5,
        rolling_median_3, ...]). scikit-learn raises when column order at
        predict() time doesn't match fit() time, so every prediction failed.

        The fitted pipeline itself always knows its true training column
        order (imputer.feature_names_in_), so we read it from there and use
        metadata.json only as a fallback if that attribute is ever absent.
        """
        try:
            first_step = self.pipeline.steps[0][1]
            return list(first_step.feature_names_in_)
        except (AttributeError, IndexError):
            logger.warning(
                "Could not read feature order from the fitted pipeline; "
                "falling back to metadata.json (may be stale)."
            )
            return self.metadata["feature_order"]

    @property
    def prediction_unit(self) -> str:
        return self.metadata.get("prediction_unit", "UNKNOWN / REQUIRES CONFIRMATION")

    @property
    def prediction_mode(self) -> str:
        """'delta_normalized_v2' (predicts drift from 5 EMT-band-normalized
        features, reconstructed as current + delta) or 'absolute_v1_legacy'
        (predicts the absolute next value directly from the 12 raw features).
        Read from metadata so swapping the deployed artifact never requires
        an API code change."""
        return self.metadata.get("prediction_mode", "absolute_v1_legacy")

    def _derive_normalized_features(self, features: MeasurementFeatures) -> dict:
        """Build the 5 EMT-band-normalized features used by the v2 model from
        the same raw physical inputs the metrologist already enters — no UI
        or input-schema change needed. EMT band width is a known, non-leaking,
        instrument-level scale constant used purely to remove the
        between-instrument scale confound identified in the audit (see
        AUDIT_REPORT.md section 7)."""
        emt_width = None
        if features.EMT_haute is not None and features.EMT_basse is not None:
            width = features.EMT_haute - features.EMT_basse
            emt_width = width if width != 0 else None

        def safe_div(numerator):
            if numerator is None or emt_width is None:
                return None
            return numerator / emt_width

        vref, consigne, ewm = features.valeur_reference, features.consigne, features.ewm_mean
        rmean3, mavg5 = features.rolling_mean_3, features.moving_average_5
        vmax, vmin = features.max, features.min

        return {
            "pos_in_band": safe_div(vref - consigne) if (vref is not None and consigne is not None) else None,
            "ewm_pos_in_band": safe_div(ewm - consigne) if (ewm is not None and consigne is not None) else None,
            "recent_shift": safe_div(vref - ewm) if (vref is not None and ewm is not None) else None,
            "rolling_trend": safe_div(rmean3 - mavg5) if (rmean3 is not None and mavg5 is not None) else None,
            "uncertainty_norm": safe_div(vmax - vmin) if (vmax is not None and vmin is not None) else None,
        }

    def predict(self, features: MeasurementFeatures) -> float:
        if not self.is_loaded:
            raise ModelNotAvailableError("Model is not loaded.")

        if self.prediction_mode == "delta_normalized_v2":
            if features.valeur_reference is None:
                raise ValueError(
                    "valeur_reference is required for this model: the prediction is "
                    "reconstructed as valeur_reference + predicted_delta."
                )
            derived = self._derive_normalized_features(features)
            X = pd.DataFrame([{name: derived.get(name) for name in self.feature_order}], columns=self.feature_order)
            predicted_delta = float(self.pipeline.predict(X)[0])
            return features.valeur_reference + predicted_delta

        row = {name: getattr(features, name) for name in self.feature_order}
        X = pd.DataFrame([row], columns=self.feature_order)
        prediction = self.pipeline.predict(X)[0]
        return float(prediction)


@lru_cache
def get_model_service() -> ModelService:
    return ModelService(get_settings())
