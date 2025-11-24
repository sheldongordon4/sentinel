import math
import os
import pytest

from app.compute.metrics import compute_metrics


def test_compute_metrics_basic_semantics():
    series = [0.8, 0.82, 0.81, 0.83, 0.84, 0.85, 0.86]
    out = compute_metrics(series, window_sec=86400)

    # New semantic fields
    assert "interactionStability" in out
    assert "signalVolatility" in out
    assert "trustContinuityRiskLevel" in out
    assert "coherenceTrend" in out
    assert "interpretation" in out and isinstance(out["interpretation"], dict)
    assert "meta" in out and isinstance(out["meta"], dict)

    # Values are finite and in expected ranges
    assert isinstance(out["interactionStability"], float)
    assert isinstance(out["signalVolatility"], float)
    assert math.isfinite(out["interactionStability"])
    assert math.isfinite(out["signalVolatility"])
    assert out["trustContinuityRiskLevel"] in {"low", "medium", "high"}
    assert out["coherenceTrend"] in {"Improving", "Steady", "Deteriorating"}

    # Legacy mirrors present (compute layer mirrors new -> old)
    assert out["coherenceMean"] == out["interactionStability"]
    assert out["volatilityIndex"] == out["signalVolatility"]
    assert out["predictedDriftRisk"] == out["trustContinuityRiskLevel"]


def test_compute_metrics_empty_series():
    out = compute_metrics([], window_sec=3600)
    # Defaults should not error
    assert out["interactionStability"] == 0.0
    assert out["signalVolatility"] == 0.0
    assert out["trustContinuityRiskLevel"] in {"low", "medium", "high"}  # thresholding may pick 'low'
    assert out["coherenceTrend"] in {"Improving", "Steady", "Deteriorating"}
