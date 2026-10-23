"""
services/history_service.py
============================
The active model (HuberRegressor, models/regression_active_model.joblib) uses
5 EMT-normalized ratio features (see models/metadata.json -> feature_order),
2 of which are derived from rolling/exponential HISTORY statistics computed
from an instrument's past measurements (ewm_pos_in_band, rolling_trend). A
metrologist entering a single new reading cannot know these off the top of
their head.

This service looks up the most recent row already computed for a given
instrument in whatever file config.yaml's api.training_features_file points
to (currently data/processed/ILA_train_augmente_features.csv - the exact
table the active model was trained on, produced by notebooks/05_Regression_
Model.ipynb section 16) and returns it as a suggestion the metrologist can
review and override — nothing here is invented or recomputed with different
logic than the training pipeline used.
"""
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd

from src.api.core.config import Settings, get_settings

logger = logging.getLogger("api.history_service")

HISTORICAL_FEATURE_COLS = [
    "rolling_mean_3",
    "moving_average_5",
    "ewm_mean",
]
CONTEXT_COLS = ["EMT_basse", "EMT_haute", "min", "max", "consigne", "valeur_reference"]


class HistoryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self) -> None:
        path = self.settings.training_features_path
        if not path.exists():
            logger.warning("Training features file not found at %s; last-known-features endpoint will be empty.", path)
            self._df = pd.DataFrame()
            return
        self._df = pd.read_csv(path, encoding="utf-8-sig")

    def list_instruments(self) -> list[str]:
        if self._df is None or self._df.empty:
            return []
        return sorted(self._df["id_instrument"].dropna().unique().tolist())

    def observation_count(self, instrument_id: str, before_year: int | None = None) -> int:
        """Count of prior observations for this instrument (optionally, only
        those strictly before a given year). Used to gate confident
        predictions: audit_v3 found instruments with <=1 prior observation
        have ~30x higher baseline error than those with more history
        (mean abs error 21.24 vs 0.6-0.9 across all other bins) - see
        AUDIT_REPORT.md section 9 / experiments/audit_v3/FINAL_REPORT.md."""
        if self._df is None or self._df.empty:
            return 0
        subset = self._df[self._df["id_instrument"] == instrument_id]
        if before_year is not None:
            subset = subset[subset["annee"] < before_year]
        return len(subset)

    def earliest_year(self, instrument_id: str) -> Optional[int]:
        if self._df is None or self._df.empty:
            return None
        subset = self._df[self._df["id_instrument"] == instrument_id]
        if subset.empty:
            return None
        return int(subset["annee"].min())

    def last_known_features(self, instrument_id: str, as_of_year: Optional[int] = None) -> Optional[dict]:
        """Return the most recent historical row for this instrument that is
        STRICTLY BEFORE as_of_year (if given). This matters: if the
        metrologist is predicting for year Y, prefilling with data from a
        year AFTER Y would leak future information into a "past" prediction.
        If as_of_year is omitted, falls back to the instrument's absolute
        latest known row."""
        if self._df is None or self._df.empty:
            return None
        subset = self._df[self._df["id_instrument"] == instrument_id]
        if as_of_year is not None:
            subset = subset[subset["annee"] < as_of_year]
        if subset.empty:
            return None
        subset = subset.sort_values("annee")
        last_row = subset.iloc[-1]
        cols = HISTORICAL_FEATURE_COLS + CONTEXT_COLS
        result = {c: (None if pd.isna(last_row[c]) else float(last_row[c])) for c in cols if c in last_row}
        result["source_year"] = int(last_row["annee"])
        return result


    def full_history(self, instrument_id: str) -> list[dict]:
        """Full year-by-year series for an instrument, for charting.
        An instrument typically has several measurement points per year, so
        rows are aggregated (mean) per year — otherwise a line chart would
        zigzag between points within the same year instead of showing a
        clean year-over-year trend. Returns the observed reference value
        plus its EMT tolerance band, min/max interval, and consigne."""
        if self._df is None or self._df.empty:
            return []
        subset = self._df[self._df["id_instrument"] == instrument_id]
        if subset.empty:
            return []
        cols = ["valeur_reference", "EMT_basse", "EMT_haute", "min", "max", "consigne"]
        grouped = subset.groupby("annee")[cols].mean().reset_index().sort_values("annee")
        records = []
        for _, row in grouped.iterrows():
            rec = {c: (None if pd.isna(row[c]) else float(row[c])) for c in cols}
            rec["annee"] = int(row["annee"])
            records.append(rec)
        return records


@lru_cache
def get_history_service() -> HistoryService:
    return HistoryService(get_settings())
