from fastapi.testclient import TestClient

from src.api.main import app
from tests.test_prediction import VALID_PAYLOAD


def test_emt_basse_must_be_lower_than_emt_haute():
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "features": {**VALID_PAYLOAD["features"], "EMT_basse": 26.0, "EMT_haute": 24.0}}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


def test_min_must_not_exceed_max():
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "features": {**VALID_PAYLOAD["features"], "min": 30.0, "max": 20.0}}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


def test_invalid_numeric_value_rejected():
    # Standard JSON has no literal for Infinity/NaN, so the realistic failure
    # mode over HTTP is a non-numeric value in a numeric field.
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "features": {**VALID_PAYLOAD["features"], "valeur_reference": "not-a-number"}}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422
