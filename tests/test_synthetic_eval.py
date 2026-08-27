import random

import pytest

from sentinel_synthetic_eval import (
    build_boundary_series,
    build_scenarios,
    cusum_detect,
    ewma_detect,
    linspace,
    mann_kendall,
    validate_paper_configuration,
)


def test_build_scenarios_is_deterministic_without_mutating_global_rng():
    random.seed(1234)
    before = random.getstate()

    first = build_scenarios()
    after = random.getstate()

    assert before == after
    assert first == build_scenarios()
    assert all(len(series) == 120 for _, series, _, _ in first)


def test_boundary_series_has_requested_coefficient_of_variation():
    series = build_boundary_series(0.10)
    series_mean = sum(series) / len(series)
    series_stdev = (
        sum((value - series_mean) ** 2 for value in series) / len(series)
    ) ** 0.5

    assert len(series) == 120
    assert pytest.approx(0.10) == series_stdev / series_mean


@pytest.mark.parametrize("detector", [ewma_detect, cusum_detect])
def test_detectors_reject_invalid_warmup(detector):
    with pytest.raises(ValueError, match="warmup"):
        detector([0.5, 0.5, 0.5], warmup=3)


def test_mann_kendall_handles_short_series():
    assert mann_kendall([0.5]) == ("Steady", 0, 0.0, 1.0, False)


def test_linspace_rejects_empty_output():
    with pytest.raises(ValueError, match="at least 1"):
        linspace(0, 1, 0)


def test_paper_configuration_uses_recorded_thresholds():
    validate_paper_configuration()
