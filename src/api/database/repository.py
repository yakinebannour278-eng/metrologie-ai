"""
database/repository.py
=======================
Thin persistence layer — keeps SQL/ORM concerns out of the API routes and
services, per the required separation of responsibilities.
"""
from sqlalchemy.orm import Session

from src.api.database.models import PredictionHistory


def save_prediction(
    db: Session,
    *,
    instrument_id: str,
    year: int,
    point: str | None,
    input_features: dict,
    prediction: float,
    model_version: str,
) -> PredictionHistory:
    entry = PredictionHistory(
        instrument_id=instrument_id,
        year=year,
        point=point,
        input_features=input_features,
        prediction=prediction,
        model_version=model_version,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_history(db: Session, limit: int = 50, instrument_id: str | None = None) -> list[PredictionHistory]:
    query = db.query(PredictionHistory)
    if instrument_id:
        query = query.filter(PredictionHistory.instrument_id == instrument_id)
    return query.order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
