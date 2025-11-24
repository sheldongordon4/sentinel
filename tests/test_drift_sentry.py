import json

import pytest

from automation import drift_sentry


def test_map_level_from_vol_uses_thresholds(monkeypatch):
    # Override module-level thresholds directly
    monkeypatch.setattr(drift_sentry, "WARN_TH", 0.20, raising=False)
    monkeypatch.setattr(drift_sentry, "CRIT_TH", 0.40, raising=False)

    assert drift_sentry.map_level_from_vol(0.10) == "low"
    assert drift_sentry.map_level_from_vol(0.30) == "medium"
    assert drift_sentry.map_level_from_vol(0.50) == "high"


def test_drift_sentry_dry_run_emits_trust_continuity_alert(monkeypatch, capsys):
    # Fake /coherence/metrics response
    def fake_http_get_json(url: str):
        return {
            "interactionStability": 0.84,
            "signalVolatility": 0.21,
            "trustContinuityRiskLevel": "medium",
        }

    monkeypatch.setattr(drift_sentry, "http_get_json", fake_http_get_json)

    # Ensure gating passes: min-level=low, use dry-run to avoid file writes
    argv = [
        "drift_sentry",
        "--window",
        "1h",
        "--api",
        "http://fake-api",
        "--min-level",
        "low",
        "--dry-run",
    ]
    # argparse in this module uses the shared sys module imported there
    monkeypatch.setattr(drift_sentry.sys, "argv", argv, raising=False)

    rc = drift_sentry.main()
    assert rc == 0

    out = capsys.readouterr().out.strip()
    # dry-run prints a single JSON object
    incident = json.loads(out)

    assert incident["event"] == "trust_continuity_alert"
    assert incident["window"] == "1h"
    assert incident["trustContinuityRisk"] == "medium"
    assert pytest.approx(incident["signalStability"], rel=1e-4) == 0.84
    assert pytest.approx(incident["signalLiquidity"], rel=1e-4) == 0.21

    trace = incident.get("trace", {})
    assert trace.get("api", "").startswith("http://fake-api")
    assert trace.get("source") == "coherence_engine_v0.2"
    assert "thresholds" in trace

