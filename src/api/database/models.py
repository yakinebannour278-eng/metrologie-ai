"""
database/models.py
===================
ORM model for the local prediction-history table. Stores exactly what the
spec asked for: identifying info, the input features actually used, the
prediction, and the model version — nothing invented on top.
"""
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.api.database.database import Base


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    point: Mapped[str] = mapped_column(String, nullable=True)
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
