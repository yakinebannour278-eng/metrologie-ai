from fastapi.testclient import TestClient

from src.api.main import app

VALID_PAYLOAD = {
    "instrument_id": "INST-08A",  # real instrument with substantial history (18 observations) - needed
    "year": 2026,                  # so this test exercises the normal path, not INSUFFICIENT_HISTORY
    "point": "1",
    "features": {
        "EMT_basse": 24.0, "EMT_haute": 26.0,
        "min": 24.0, "max": 24.2,
        "consigne": 24.0, "valeur_reference": 24.1,
        "rolling_min_3": 24.0, "rolling_max_3": 24.2,
        "rolling_mean_3": 24.1, "rolling_median_3": 24.1,
        "moving_average_5": 24.1, "ewm_mean": 24.1,
    },
}


def test_valid_prediction_returns_200_and_persists_history():
    with TestClient(app) as client:
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["prediction"], float)
        assert body["status"] == "prediction_generated"
        assert body["prediction_id"] is not None
        # Honest baseline/recommendation fields must always be present
        assert "baseline_prediction" in body
        assert "recommended_prediction" in body
        assert body["recommended_source"] in ("baseline_persistence", "ml_model")
        assert body["history_status"] == "SUFFICIENT_HISTORY"
        assert body["n_prior_observations"] >= 2

        hist = client.get("/history", params={"instrument_id": "INST-08A"})
        assert hist.status_code == 200
        assert len(hist.json()) >= 1


def test_unknown_instrument_triggers_insufficient_history():
    """A brand-new instrument with zero recorded history must NOT receive a
    confident numeric recommendation - see experiments/audit_v3/FINAL_REPORT.md,
    'how much history is required'."""
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "instrument_id": "INST-NEVER-SEEN-BEFORE-XYZ"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["history_status"] == "INSUFFICIENT_HISTORY"
        assert body["status"] == "insufficient_history"
        assert body["n_prior_observations"] == 0
        assert "INSUFFICIENT_HISTORY" in body["interpretation"]


def test_missing_optional_feature_still_predicts_via_imputer():
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "features": {**VALID_PAYLOAD["features"]}}
        payload["features"]["consigne"] = None
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


def test_missing_instrument_id_rejected():
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "instrument_id": ""}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


def test_instrument_history_returns_one_row_per_year():
    with TestClient(app) as client:
        resp = client.get("/model/instruments/INST-08A/history")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) > 0
        years = [r["annee"] for r in rows]
        assert years == sorted(years)
        assert len(years) == len(set(years))  # one row per year, aggregated
        for r in rows:
            assert "valeur_reference" in r and "EMT_basse" in r and "EMT_haute" in r


def test_instrument_history_unknown_instrument_404():
    with TestClient(app) as client:
        resp = client.get("/model/instruments/DOES-NOT-EXIST/history")
        assert resp.status_code == 404
    """Regression test: prefill must never leak data from AFTER the year
    the metrologist is predicting for."""
    with TestClient(app) as client:
        no_filter = client.get("/model/instruments/INST-08A/last-known-features")
        assert no_filter.status_code == 200
        latest_year = no_filter.json()["source_year"]

        early = client.get(
            "/model/instruments/INST-08A/last-known-features",
            params={"year": latest_year},
        )
        assert early.status_code == 200
        assert early.json()["source_year"] < latest_year

        too_early = client.get(
            "/model/instruments/INST-08A/last-known-features",
            params={"year": 1900},
        )
        assert too_early.status_code == 404


def test_invalid_year_rejected():
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "year": 1899}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422
