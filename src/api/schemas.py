"""
schemas.py
==========
Pydantic request/response models for the metrology drift-prediction API.

Field set matches EXACTLY the trained model's feature_order (see
models/metadata.json). Nothing here invents a feature the model does not
use, and nothing here invents a unit, threshold, or confidence score the
model does not produce.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class MeasurementFeatures(BaseModel):
    """The 12 numeric inputs the trained pipeline expects, in the exact
    field names used during training. All are individually optional at the
    schema level because the saved pipeline includes a median imputer and
    was itself trained on data with missing values (e.g. 'consigne' is
    missing in ~34% of training rows) — this mirrors training-time
    behaviour instead of inventing a stricter rule."""

    EMT_basse: Optional[float] = Field(None, description="Lower EMT tolerance bound.")
    EMT_haute: Optional[float] = Field(None, description="Upper EMT tolerance bound.")
    max: Optional[float] = Field(None, description="Upper bound of the measurement interval.")
    min: Optional[float] = Field(None, description="Lower bound of the measurement interval.")
    consigne: Optional[float] = Field(None, description="Nominal/setpoint value (often absent).")
    valeur_reference: Optional[float] = Field(None, description="Current reference value.")

    rolling_min_3: Optional[float] = Field(None, description="HISTORICAL: rolling min over last 3 campaigns.")
    rolling_max_3: Optional[float] = Field(None, description="HISTORICAL: rolling max over last 3 campaigns.")
    rolling_mean_3: Optional[float] = Field(None, description="HISTORICAL: rolling mean over last 3 campaigns.")
    rolling_median_3: Optional[float] = Field(None, description="HISTORICAL: rolling median over last 3 campaigns.")
    moving_average_5: Optional[float] = Field(None, description="HISTORICAL: moving average over last 5 campaigns.")
    ewm_mean: Optional[float] = Field(None, description="HISTORICAL: exponentially weighted mean.")

    @model_validator(mode="after")
    def check_finite_and_logical(self) -> "MeasurementFeatures":
        import math

        for name, value in self.__dict__.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Feature '{name}' must be a finite number (got {value}).")

        if self.EMT_basse is not None and self.EMT_haute is not None:
            if self.EMT_basse >= self.EMT_haute:
                raise ValueError("EMT_basse must be strictly lower than EMT_haute.")

        if self.min is not None and self.max is not None:
            if self.min > self.max:
                raise ValueError("min must be lower than or equal to max.")

        return self


class PredictionRequest(BaseModel):
    instrument_id: str = Field(..., min_length=1, description="Instrument identifier (e.g. 'INST-03B-...').")
    year: int = Field(..., description="Measurement campaign year the prediction is made FOR.")
    point: Optional[str] = Field(None, description="Measurement point identifier (display only, not a model feature).")
    features: MeasurementFeatures

    @model_validator(mode="after")
    def check_instrument_id(self) -> "PredictionRequest":
        if not self.instrument_id.strip():
            raise ValueError("instrument_id cannot be empty or whitespace.")
        return self


class PredictionResponse(BaseModel):
    success: bool
    model_name: str
    model_version: str
    prediction: float
    baseline_prediction: float
    recommended_prediction: float
    recommended_source: str
    prediction_unit: str
    instrument_id: str
    year: int
    point: Optional[str]
    input_features: MeasurementFeatures
    timestamp: str
    status: str
    history_status: str
    n_prior_observations: Optional[int] = None
    interpretation: str
    prediction_id: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    target_variable: str
    target_description: str
    prediction_unit: str
    feature_order: list[str]
    training_date: str
    n_training_rows: int
    n_instruments: int
    evaluation_metrics_groupkfold: dict
    known_limitations: list[str]


class HistoryEntry(BaseModel):
    prediction_id: int
    instrument_id: str
    timestamp: str
    year: int
    point: Optional[str]
    prediction: float
    model_version: str
