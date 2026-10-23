"""
main.py
=======
FastAPI microservice exposing the trained drift-prediction regression
model to the metrologist interface (frontend/app.py).

Architecture (current phase — regression only, per project scope):

    METROLOGIST INTERFACE (Streamlit)
            |  HTTP
            v
        FASTAPI  <- this file
            |
      PREPROCESSING + REGRESSION  (single joblib Pipeline, see model_service.py)
            |
        PREDICTION
            |
      RESULT DISPLAY + local SQLite history

Anomaly detection, classification, and the future SQL Server / .NET
architecture are OUT OF SCOPE for this service — see README.md.
"""
import logging
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.core.config import Settings, get_settings
from src.api.database.database import get_db, init_db
from src.api.database import repository
from src.api.schemas import (
    HealthResponse,
    HistoryEntry,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services.history_service import HistoryService, get_history_service
from src.api.services.model_service import ModelNotAvailableError, ModelService, get_model_service
from src.api.services.prediction_service import run_prediction
from src.api.utils.validation import validate_year

settings = get_settings()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(settings.logs_dir / "api.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("api.main")

_model_load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_load_error
    logger.info("API startup: initializing database at %s", settings.database_path)
    init_db()
    try:
        get_model_service()
        logger.info("API startup: model loaded successfully.")
    except ModelNotAvailableError as e:
        _model_load_error = str(e)
        logger.error("API startup: model failed to load: %s", e)
    yield
    logger.info("API shutdown.")


app = FastAPI(
    title="Metrology Drift Prediction API",
    description=(
        "Regression-only microservice predicting an instrument's next reference value "
        "from its metrology history. Decision-support tool; does not replace metrologist review."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model_service_or_503() -> ModelService:
    try:
        return get_model_service()
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    model_loaded = False
    try:
        model_loaded = get_model_service().is_loaded
    except ModelNotAvailableError:
        model_loaded = False
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info(model_service: ModelService = Depends(get_model_service_or_503)) -> ModelInfoResponse:
    md = model_service.metadata
    return ModelInfoResponse(
        model_name=md["model_name"],
        model_version=md["model_version"],
        model_type=md["model_type"],
        target_variable=md["target_variable"],
        target_description=md["target_description"],
        prediction_unit=md["prediction_unit"],
        feature_order=md["feature_order"],
        training_date=md["training_date"],
        n_training_rows=md["n_training_rows"],
        n_instruments=md["n_instruments"],
        evaluation_metrics_groupkfold=md["evaluation_metrics_groupkfold"],
        known_limitations=md["known_limitations"],
    )


@app.get("/model/instruments", tags=["model"])
def list_instruments(history_service: HistoryService = Depends(get_history_service)) -> list[str]:
    return history_service.list_instruments()


@app.get("/model/instruments/{instrument_id}/last-known-features", tags=["model"])
def last_known_features(
    instrument_id: str,
    year: int | None = None,
    history_service: HistoryService = Depends(get_history_service),
) -> dict:
    result = history_service.last_known_features(instrument_id, as_of_year=year)
    if result is not None:
        return result

    earliest = history_service.earliest_year(instrument_id)
    if earliest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Unknown instrument_id='{instrument_id}'. It does not appear in the "
                f"training data at all — check spelling (IDs must match exactly, e.g. "
                f"'INST-03B-CHAMBREDEFERMENTATION', not just 'INST-03B'). "
                f"See GET /model/instruments for the full list."
            ),
        )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            f"instrument_id='{instrument_id}' is known, but its earliest recorded year "
            f"is {earliest}, so there is no historical data strictly before year={year}. "
            f"Enter values manually below, or pick a year after {earliest}."
        ),
    )


@app.get("/model/instruments/{instrument_id}/history", tags=["model"])
def instrument_history(
    instrument_id: str,
    history_service: HistoryService = Depends(get_history_service),
) -> list[dict]:
    """Full historical series for an instrument, averaged per year across
    its measurement points (valeur_reference, EMT bounds, min/max,
    consigne) — used by the UI to plot a trend chart alongside a new
    prediction."""
    records = history_service.full_history(instrument_id)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No historical data found for instrument_id='{instrument_id}'.",
        )
    return records


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service_or_503),
    history_service: HistoryService = Depends(get_history_service),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    validate_year(request.year, settings)
    try:
        return run_prediction(request, model_service, history_service, db)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {e}")


@app.get("/history", response_model=list[HistoryEntry], tags=["history"])
def get_history(
    instrument_id: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[HistoryEntry]:
    rows = repository.list_history(db, limit=limit, instrument_id=instrument_id)
    return [
        HistoryEntry(
            prediction_id=r.prediction_id,
            instrument_id=r.instrument_id,
            timestamp=r.timestamp.isoformat(timespec="seconds"),
            year=r.year,
            point=r.point,
            prediction=r.prediction,
            model_version=r.model_version,
        )
        for r in rows
    ]
