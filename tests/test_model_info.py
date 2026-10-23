import json

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.core.config import get_settings


def test_model_info_reports_real_metadata():
    """Compares against the live models/metadata.json rather than a hardcoded
    model name, so this test does not go stale every time the benchmark
    (notebooks/05 or notebooks/06) legitimately promotes a different
    algorithm. What must stay true regardless of which algorithm is active:
    the API reports whatever is really in metadata.json (no hardcoding), and
    an unvalidated model must never be reported as beating baseline."""
    settings = get_settings()
    with open(settings.model_metadata_path, encoding="utf-8") as f:
        live_metadata = json.load(f)

    with TestClient(app) as client:
        resp = client.get("/model/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_name"] == live_metadata["model_name"]
        assert "delta" in body["target_variable"].lower() or "delta" in body["target_description"].lower()
        assert len(body["feature_order"]) == 5
        assert "known_limitations" in body
        assert len(body["known_limitations"]) > 0
        # The model must not overstate its performance: the honest baseline
        # comparison must be present and must show it does NOT beat baseline.
        assert body["evaluation_metrics_groupkfold"]["beats_baseline"] is False
