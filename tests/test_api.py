from fastapi.testclient import TestClient
from app.api import app


client = TestClient(app)


def test_metrics_include_legacy_true():
    resp = client.get("/coherence/metrics", params={"include_legacy": True})
    assert resp.status_code == 200
    data = resp.json()

    # New semantic fields always present
    assert "interactionStability" in data
    assert "signalVolatility" in data
    assert "trustContinuityRiskLevel" in data
    assert "coherenceTrend" in data

    # Legacy mirrors should be present when include_legacy=true
    assert "coherenceMean" in data
    assert "volatilityIndex" in data
    assert "predictedDriftRisk" in data


def test_metrics_include_legacy_false():
    resp = client.get("/coherence/metrics", params={"include_legacy": False})
    assert resp.status_code == 200
    data = resp.json()

    # Semantic fields still present
    assert "interactionStability" in data
    assert "signalVolatility" in data
    assert "trustContinuityRiskLevel" in data
    assert "coherenceTrend" in data

    # Legacy mirrors should be removed
    assert "coherenceMean" not in data
    assert "volatilityIndex" not in data
    assert "predictedDriftRisk" not in data


def test_status_includes_mode_and_thresholds(monkeypatch):
    # Set env to known values for this test only
    monkeypatch.setenv("COHERENCE_MODE", "production")
    monkeypatch.setenv("COHERENCE_WARN_THRESHOLD", "0.12")
    monkeypatch.setenv("COHERENCE_CRITICAL_THRESHOLD", "0.30")
    monkeypatch.setenv("TREND_SENSITIVITY", "0.02")

    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()

    assert data["mode"] == "production"
    # /status echoes raw env strings
    assert data["warn_threshold"] == "0.12"
    assert data["critical_threshold"] == "0.30"
    assert data["trend_sensitivity"] == "0.02"
    assert "timestamp" in data

