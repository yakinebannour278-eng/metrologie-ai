"""
services/prediction_service.py
===============================
Business logic for a single prediction request.

IMPORTANT: whether the ML model or the persistence baseline is recommended
is NEVER hardcoded here - it is read live from models/metadata.json's
evaluation_metrics_groupkfold.beats_baseline flag.

As of the most recent decision (2026-09-18, see
models/metadata.json:primary_validation_chronological and
notebooks/06_Audit_v3_Full_Benchmark.ipynb section 3B), the PRIMARY
validation protocol for this flag is a leakage-safe CHRONOLOGICAL split
(per instrument: older real observations = history/train, newest real
observations = held-out future/test) - because this project's real use case
is predicting a KNOWN instrument's future drift, not generalizing to a
brand-new instrument. Under that protocol, 0 of 14 tested algorithms
(including HuberRegressor, previously selected by a single instrument-
disjoint GroupShuffleSplit with MAE=0.894 - a different question: unseen-
instrument generalization, not future-in-time prediction) beat the
persistence baseline, so beats_baseline=False. GroupKFold (unseen
instruments) and repeated-GroupShuffleSplit (robustness) results are kept in
metadata.json for reference but are secondary, not the selection criterion.
Because this flag is read from metadata rather than hardcoded, a future
retrain that finds a model which DOES beat persistence chronologically flips
this automatically, with no code change needed here. Per this design, this
service:

  1. Always computes AND returns the baseline prediction alongside the ML
     model's prediction - never hides the comparison.
  2. Sets `recommended_source` from metadata's beats_baseline flag (see above).
  3. Gates confident predictions on a minimum-history rule: instruments with
     <=1 prior observation had ~30x higher baseline error than better-
     documented instruments (mean abs error 21.24 vs 0.6-0.9 - see
     experiments/audit_v3/FINAL_REPORT.md, "How much history is required?").
     Below that threshold, status is INSUFFICIENT_HISTORY, not a confident
     numeric prediction. This gate applies regardless of which model is
     active - it is about data sufficiency, not model choice.
"""
import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.api.database import repository
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.services.history_service import HistoryService
from src.api.services.model_service import ModelService

logger = logging.getLogger("api.prediction_service")

# Empirically justified in experiments/audit_v3/FINAL_REPORT.md ("How much
# history is required?" table): the 0-1-prior-observation bin has a mean
# absolute baseline error ~30x every other bin. 2 is the smallest threshold
# that excludes that catastrophic bin.
MINIMUM_HISTORY_REQUIRED = 2


def build_interpretation(
    request: PredictionRequest,
    ml_prediction: float,
    baseline_prediction: float,
    recommended_source: str,
    unit: str,
    history_status: str,
    n_prior: int | None,
) -> str:
    """Neutral, factual, and HONEST about the baseline comparison - no
    'safe'/'critical'/'dangerous' labels, and no pretending the ML model is
    the recommended answer when it has not been shown to beat a trivial
    baseline (see experiments/audit_v3/FINAL_REPORT.md)."""
    if history_status == "INSUFFICIENT_HISTORY":
        return (
            f"INSUFFICIENT_HISTORY: instrument '{request.instrument_id}' has "
            f"{n_prior if n_prior is not None else 'fewer than'} prior recorded "
            f"observation(s), below the minimum of {MINIMUM_HISTORY_REQUIRED} required "
            "for a meaningful prediction. Instruments with this little history showed "
            "roughly 30x higher error than better-documented instruments in validation "
            "testing. No confident prediction is provided; a metrologist's direct "
            "judgement is required for this instrument."
        )

    parts = [
        f"Prediction requested for instrument '{request.instrument_id}', "
        f"targeting the campaign year {request.year}."
    ]
    unit_str = (" " + unit) if unit and "UNKNOWN" not in unit else ""

    if recommended_source == "baseline_persistence":
        parts.append(
            f"RECOMMENDED VALUE: {baseline_prediction:.4f}{unit_str} (persistence baseline: "
            "assumes no change from the current reading). This is recommended because, under "
            "PRIMARY chronological validation (predicting each instrument's own genuinely "
            "future, held-out-in-time observations - see "
            "models/metadata.json:primary_validation_chronological), 0 of 14 tested regression "
            "algorithms (Linear/Ridge/Lasso/ElasticNet/Huber/KNN/SVR/DecisionTree/RandomForest/"
            "ExtraTrees/GradientBoosting/HistGradientBoosting/AdaBoost/Bagging) reliably beat "
            "this simple baseline - see experiments/audit_v3/model_comparison_PRIMARY_CHRONOLOGICAL.csv."
        )
        parts.append(
            f"For reference only, the trained ML model predicts "
            f"{ml_prediction:.4f}{unit_str} - shown for transparency, NOT recommended as the "
            "primary estimate."
        )
    else:
        parts.append(
            f"RECOMMENDED VALUE: {ml_prediction:.4f}{unit_str} (this model version has been "
            "verified to beat the persistence baseline - see models/metadata.json)."
        )
        parts.append(f"Persistence baseline, for reference: {baseline_prediction:.4f}{unit_str}.")

    parts.append(
        "This is a decision-support estimate only. It does not by itself constitute an "
        "ISO/IEC 17025 conformity decision and does not replace a metrologist's review."
    )
    return " ".join(parts)


def run_prediction(
    request: PredictionRequest,
    model_service: ModelService,
    history_service: HistoryService,
    db: Session,
) -> PredictionResponse:
    n_prior = history_service.observation_count(request.instrument_id, before_year=request.year)
    history_status = "INSUFFICIENT_HISTORY" if n_prior < MINIMUM_HISTORY_REQUIRED else "SUFFICIENT_HISTORY"

    ml_prediction = model_service.predict(request.features)
    unit = model_service.prediction_unit

    baseline_prediction = request.features.valeur_reference
    beats_baseline = bool(
        model_service.metadata.get("evaluation_metrics_groupkfold", {}).get("beats_baseline", False)
    )
    recommended_source = "ml_model" if beats_baseline else "baseline_persistence"
    recommended_prediction = ml_prediction if beats_baseline else baseline_prediction

    interpretation = build_interpretation(
        request, ml_prediction, baseline_prediction, recommended_source, unit, history_status, n_prior
    )
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")

    entry = repository.save_prediction(
        db,
        instrument_id=request.instrument_id,
        year=request.year,
        point=request.point,
        input_features=request.features.model_dump(),
        prediction=ml_prediction,
        model_version=model_service.metadata.get("model_version", "unknown"),
    )

    logger.info(
        "Prediction: instrument=%s year=%s n_prior=%d history_status=%s ml_pred=%.4f "
        "baseline=%s recommended_source=%s model=%s",
        request.instrument_id, request.year, n_prior, history_status, ml_prediction,
        f"{baseline_prediction:.4f}" if baseline_prediction is not None else "N/A",
        recommended_source, model_service.metadata.get("model_version"),
    )

    return PredictionResponse(
        success=True,
        model_name=model_service.metadata.get("model_name", "unknown"),
        model_version=model_service.metadata.get("model_version", "unknown"),
        prediction=ml_prediction,
        baseline_prediction=baseline_prediction if baseline_prediction is not None else ml_prediction,
        recommended_prediction=recommended_prediction if recommended_prediction is not None else ml_prediction,
        recommended_source=recommended_source,
        prediction_unit=unit,
        instrument_id=request.instrument_id,
        year=request.year,
        point=request.point,
        input_features=request.features,
        timestamp=timestamp,
        status="insufficient_history" if history_status == "INSUFFICIENT_HISTORY" else "prediction_generated",
        history_status=history_status,
        n_prior_observations=n_prior,
        interpretation=interpretation,
        prediction_id=entry.prediction_id,
    )
