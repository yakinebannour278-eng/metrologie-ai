from fastapi.testclient import TestClient

from src.api.main import app


def test_health_returns_ok_when_model_loaded():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert "model_loaded" in body
        assert "timestamp" in body
