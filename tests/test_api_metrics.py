import os
import pytest
from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_and_status_endpoints():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    # Keys should exist even if values come from defaults
    for k in ("mode", "warn_threshold", "critical_threshold", "trend_sensitivity", "timestamp"):
        assert k in body


def test_metrics_default_includes_legacy_fields():
    r = client.get("/coherence/metrics")
    assert r.status_code == 200
    body = r.json()

    # New semantic fields
    for k in ("interactionStability", "signalVolatility", "trustContinuityRiskLevel", "coherenceTrend", "interpretation", "meta"):
        assert k in body

    # Legacy present by default
    for k in ("coherenceMean", "volatilityIndex", "predictedDriftRisk"):
        assert k in body


def test_metrics_exclude_legacy_fields_when_flag_false():
    r = client.get("/coherence/metrics?include_legacy=false")
    assert r.status_code == 200
    body = r.json()

    # New semantic fields present
    for k in ("interactionStability", "signalVolatility", "trustContinuityRiskLevel", "coherenceTrend"):
        assert k in body

    # Legacy stripped
    for k in ("coherenceMean", "volatilityIndex", "predictedDriftRisk"):
        assert k not in body


def test_metrics_meta_contains_expected_fields():
    r = client.get("/coherence/metrics")
    assert r.status_code == 200
    meta = r.json().get("meta", {})
    for k in ("method", "windowSec", "n", "timestamp"):
        assert k in meta
