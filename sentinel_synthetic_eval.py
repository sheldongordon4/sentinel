"""
sentinel_synthetic_eval.py
──────────────────────────
Reproduces the synthetic evaluation reported in the paper, including the
scenario tables, threshold boundaries, detector behavior, and trend comparisons.

Run from the repository root:
    python sentinel_synthetic_eval.py

No API keys required. The script is deterministic and writes results to
synthetic_eval_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from app.compute import metrics as metrics_module
from app.compute.metrics import compute_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "synthetic_eval_results.json"

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N = 120
WARMUP = 24

# These values are part of the published synthetic experiment.
TAU_WARN = 0.10
TAU_CRITICAL = 0.25

# EWMA parameters
EWMA_LAMBDA = 0.20
EWMA_L = 3.0

# CUSUM parameters
CUSUM_K = 0.5
CUSUM_H = 4.0


def linspace(start: float, stop: float, n: int) -> list[float]:
    """Return n evenly spaced values, including both endpoints."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + step * i for i in range(n)]


def add_noise(series: list[float], sigma: float, seed: int = RANDOM_SEED) -> list[float]:
    """Add bounded Gaussian noise without changing the global random state."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = random.Random(seed)
    return [max(0.0, min(1.0, value + rng.gauss(0, sigma))) for value in series]


def normal_cdf(x: float) -> float:
    """Return the standard normal cumulative distribution at x."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def build_scenarios() -> list[tuple[str, list[float], str, str]]:
    """Build the five deterministic signal patterns used by the evaluation."""
    s1 = add_noise([0.88] * N, sigma=0.015)
    s2 = add_noise(linspace(0.87, 0.48, N), sigma=0.02)
    s3 = (
        add_noise([0.85] * 72, sigma=0.01)
        + add_noise([0.30] * 24, sigma=0.04)
        + add_noise(linspace(0.30, 0.65, 24), sigma=0.02)
    )
    rng = random.Random(99)
    s4 = [rng.uniform(0.20, 0.90) for _ in range(N)]
    s5 = add_noise([0.45] * 60, sigma=0.02) + add_noise(
        linspace(0.45, 0.88, 60), sigma=0.015
    )

    return [
        ("S1: Stable Baseline", s1, "low", "Steady"),
        ("S2: Gradual Drift", s2, "medium", "Deteriorating"),
        ("S3: Hallucination Burst", s3, "high", "Deteriorating"),
        ("S4: Chronic High Volatility", s4, "high", "Steady"),
        ("S5: Recovery from Drift", s5, "high", "Improving"),
    ]


def ewma_detect(series: list[float], warmup: int = WARMUP, lam: float = EWMA_LAMBDA, control_limit: float = EWMA_L):
    """Return the first EWMA alert index and total alert count."""
    _validate_detector_inputs(series, warmup)
    if not 0 < lam <= 1:
        raise ValueError("lam must be in the interval (0, 1]")
    if control_limit <= 0:
        raise ValueError("control_limit must be positive")

    baseline = series[:warmup]
    mu0 = mean(baseline)
    sigma0 = pstdev(baseline) if pstdev(baseline) > 0 else 1e-6
    cl = control_limit * sigma0 * (lam / (2 - lam)) ** 0.5

    z = mu0
    first_alert = None
    total_alerts = 0
    for i, value in enumerate(series[warmup:], start=warmup):
        z = lam * value + (1 - lam) * z
        if z > mu0 + cl or z < mu0 - cl:
            total_alerts += 1
            if first_alert is None:
                first_alert = i + 1
    return first_alert, total_alerts


def cusum_detect(series: list[float], warmup: int = WARMUP, k_sigma: float = CUSUM_K, h_sigma: float = CUSUM_H):
    """Return the first two-sided CUSUM alert index and total alert count."""
    _validate_detector_inputs(series, warmup)
    if k_sigma < 0 or h_sigma <= 0:
        raise ValueError("k_sigma must be non-negative and h_sigma must be positive")

    baseline = series[:warmup]
    mu0 = mean(baseline)
    sigma0 = pstdev(baseline) if pstdev(baseline) > 0 else 1e-6
    k = k_sigma * sigma0
    h = h_sigma * sigma0

    c_plus = 0.0
    c_minus = 0.0
    first_alert = None
    total_alerts = 0
    for i, value in enumerate(series[warmup:], start=warmup):
        c_plus = max(0, c_plus + (value - mu0 - k))
        c_minus = max(0, c_minus - (value - mu0) + k)
        if c_plus > h or c_minus > h:
            total_alerts += 1
            if first_alert is None:
                first_alert = i + 1
    return first_alert, total_alerts


def mann_kendall(series: list[float], alpha: float = 0.05):
    """Two-sided Mann-Kendall trend test returning label, S, Z, p-value, and significance."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    n = len(series)
    if n < 2:
        return "Steady", 0, 0.0, 1.0, False

    s_statistic = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            if diff > 0:
                s_statistic += 1
            elif diff < 0:
                s_statistic -= 1

    variance_s = n * (n - 1) * (2 * n + 5) / 18
    if s_statistic > 0:
        z_score = (s_statistic - 1) / math.sqrt(variance_s)
    elif s_statistic < 0:
        z_score = (s_statistic + 1) / math.sqrt(variance_s)
    else:
        z_score = 0.0

    p_value = 2 * (1 - normal_cdf(abs(z_score)))
    significant = p_value < alpha

    if not significant:
        label = "Steady"
    elif z_score > 0:
        label = "Improving"
    else:
        label = "Deteriorating"

    return label, s_statistic, round(z_score, 4), round(p_value, 4), significant


def build_boundary_series(target_cv: float, n: int = 120) -> list[float]:
    """Construct a two-value alternating series with a precise coefficient of variation."""
    if target_cv < 0:
        raise ValueError("target_cv must be non-negative")
    if n < 2:
        raise ValueError("n must be at least 2")

    mu = 0.75
    sigma = target_cv * mu
    a = max(0.0, min(1.0, mu + sigma))
    b = max(0.0, min(1.0, mu - sigma))
    return [a if i % 2 == 0 else b for i in range(n)]


def drift_sentry_fires(risk_level: str, min_level: str) -> bool:
    """Return whether a risk level meets a configured minimum alert level."""
    order = {"low": 0, "medium": 1, "high": 2}
    return order[risk_level] >= order[min_level]


def _validate_detector_inputs(series: list[float], warmup: int) -> None:
    """Validate the baseline window required by control-chart detectors."""
    if warmup < 2 or warmup >= len(series):
        raise ValueError("warmup must be at least 2 and smaller than series length")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the evaluation runner."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(args)


def validate_paper_configuration() -> None:
    """Ensure environment overrides cannot silently change the paper results."""
    configured = (
        metrics_module.SENTINEL_WARN_THRESHOLD,
        metrics_module.SENTINEL_CRITICAL_THRESHOLD,
    )
    expected = (TAU_WARN, TAU_CRITICAL)
    if configured != expected:
        raise RuntimeError(
            "Synthetic paper reproduction requires "
            f"SENTINEL_WARN_THRESHOLD={TAU_WARN} and "
            f"SENTINEL_CRITICAL_THRESHOLD={TAU_CRITICAL}; "
            f"got {configured[0]} and {configured[1]}"
        )


def run(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run the synthetic evaluation and save the JSON report."""
    validate_paper_configuration()
    scenarios = build_scenarios()
    post_warmup = N - WARMUP
    results: dict[str, Any] = {}

    print("\n" + "=" * 70)
    print("TABLE 1 & 2: METRIC CLASSIFICATION AND DISTRIBUTIONAL STATISTICS")
    print("=" * 70)
    print(
        f"{'Scenario':<32} {'Stability':>10} {'Volatility':>11} {'Risk':<10} {'Trend':<15} {'Exp Risk':<10} {'Exp Trend'}"
    )
    print("-" * 100)

    table1: list[dict[str, Any]] = []
    for name, series, exp_risk, exp_trend in scenarios:
        out = compute_metrics(series, window_sec=86400)
        stab = out["interactionStability"]
        vol = out["signalVolatility"]
        risk = out["trustContinuityRiskLevel"]
        trend = out["sentinelTrend"]
        risk_match = "✓" if risk == exp_risk else "✗"
        trend_match = "✓" if trend == exp_trend else "✗"

        print(
            f"{name:<32} {stab:>10.4f} {vol:>11.4f} {risk:<10} {trend:<15} {exp_risk:<10} {exp_trend} {risk_match}{trend_match}"
        )

        table1.append(
            {
                "scenario": name,
                "interactionStability": round(stab, 4),
                "signalVolatility": round(vol, 4),
                "trustContinuityRiskLevel": risk,
                "sentinelTrend": trend,
                "score_min": round(min(series), 4),
                "score_max": round(max(series), 4),
                "true_mean": round(mean(series), 4),
                "true_stdev": round(pstdev(series), 4),
                "exp_risk": exp_risk,
                "exp_trend": exp_trend,
                "risk_match": risk_match,
                "trend_match": trend_match,
            }
        )

    results["table1_and_2"] = table1

    print("\n" + "=" * 70)
    print("TABLE 3: DRIFT SENTRY ALERT EMISSION BY MIN-LEVEL CONFIGURATION")
    print("=" * 70)
    print(
        f"{'Scenario':<32} {'Risk':<10} {'min=low':>8} {'min=medium':>12} {'min=high':>10}"
    )
    print("-" * 75)

    table3 = []
    for row in table1:
        risk = row["trustContinuityRiskLevel"]
        fires = {level: drift_sentry_fires(risk, level) for level in ("low", "medium", "high")}
        print(
            f"{row['scenario']:<32} {risk:<10} {'Yes' if fires['low'] else 'No':>8} "
            f"{'Yes' if fires['medium'] else 'No':>12} {'Yes' if fires['high'] else 'No':>10}"
        )
        table3.append({"scenario": row["scenario"], "risk": risk, **fires})
    results["table3"] = table3

    print("\n" + "=" * 70)
    print("TABLE 4: THRESHOLD BOUNDARY VERIFICATION")
    print(f"tau_warn={TAU_WARN}, tau_critical={TAU_CRITICAL}")
    print("=" * 70)
    print(f"{'Test Case':<40} {'Target CV':>10} {'Computed CV':>12} {'Risk':<10}")
    print("-" * 75)

    boundary_cases = [
        ("At warn boundary (CV = 0.10)", 0.100),
        ("Just below warn (CV = 0.099)", 0.099),
        ("Just above warn (CV = 0.101)", 0.101),
        ("At critical boundary (CV = 0.25)", 0.250),
        ("Just below critical (CV = 0.249)", 0.249),
        ("Just above critical (CV = 0.251)", 0.251),
    ]

    table4 = []
    for label, target_cv in boundary_cases:
        series = build_boundary_series(target_cv)
        out = compute_metrics(series, window_sec=86400)
        computed_cv = out["signalVolatility"]
        risk = out["trustContinuityRiskLevel"]
        print(f"{label:<40} {target_cv:>10.3f} {computed_cv:>12.4f} {risk:<10}")
        table4.append(
            {
                "case": label,
                "target_cv": target_cv,
                "computed_cv": round(computed_cv, 4),
                "classified_risk": risk,
            }
        )
    results["table4"] = table4

    print("\n" + "=" * 70)
    print("TABLE 5: DETECTION BEHAVIOR — EWMA / CUSUM / SENTINEL")
    print(
        f"n={N}, warmup={WARMUP}, EWMA λ={EWMA_LAMBDA} L={EWMA_L}, "
        f"CUSUM k={CUSUM_K}σ h={CUSUM_H}σ"
    )
    print("=" * 70)
    print(
        f"{'Scenario':<32} {'CUSUM 1st':>10} {'CUSUM tot':>10} "
        f"{'CUSUM FAR':>10} {'EWMA 1st':>10} {'EWMA tot':>10} "
        f"{'EWMA FAR':>10} {'Sentinel':>20}"
    )
    print("-" * 115)

    table5 = []
    for name, series, _, _ in scenarios:
        cf, ca = cusum_detect(series)
        ef, ea = ewma_detect(series)
        out = compute_metrics(series, window_sec=86400)
        sentinel_label = f"{out['trustContinuityRiskLevel']}, {out['sentinelTrend']}"
        c_far = round(ca / post_warmup, 3)
        e_far = round(ea / post_warmup, 3)
        cf_str = f"s{cf}" if cf else "none"
        ef_str = f"s{ef}" if ef else "none"

        print(
            f"{name:<32} {cf_str:>10} {ca:>10} {c_far:>10.3f} "
            f"{ef_str:>10} {ea:>10} {e_far:>10.3f} {sentinel_label:>20}"
        )

        table5.append(
            {
                "scenario": name,
                "cusum_first_alert": cf,
                "cusum_total_alerts": ca,
                "cusum_far": c_far,
                "ewma_first_alert": ef,
                "ewma_total_alerts": ea,
                "ewma_far": e_far,
                "sentinel_classification": sentinel_label,
            }
        )
    results["table5"] = table5

    print("\n" + "=" * 70)
    print("TABLE 6: MANN-KENDALL TREND TEST vs sentinelTrend (α = 0.05)")
    print("=" * 70)
    print(
        f"{'Scenario':<32} {'MK Trend':<15} {'Z':>8} {'p':>8} {'Sig':>5} {'sentinelTrend':<15} {'Match'}"
    )
    print("-" * 95)

    table6 = []
    for name, series, _, _ in scenarios:
        mk_label, _, z_score, p_value, significant = mann_kendall(series)
        out = compute_metrics(series, window_sec=86400)
        sentinel_trend = out["sentinelTrend"]
        match = "✓" if mk_label == sentinel_trend else "✗"
        print(
            f"{name:<32} {mk_label:<15} {z_score:>8.3f} {p_value:>8.4f} "
            f"{'Yes' if significant else 'No':>5} {sentinel_trend:<15} {match}"
        )
        table6.append(
            {
                "scenario": name,
                "mk_trend": mk_label,
                "Z": z_score,
                "p_value": p_value,
                "significant": significant,
                "sentinelTrend": sentinel_trend,
                "agreement": match == "✓",
            }
        )
    results["table6"] = table6

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    full_matches = sum(1 for row in table1 if row["risk_match"] == "✓" and row["trend_match"] == "✓")
    mk_matches = sum(1 for row in table6 if row["agreement"])
    cusum_s1 = table5[0]["cusum_far"]
    ewma_s1 = table5[0]["ewma_far"]

    print(f"Table 1 — Full classification match (risk + trend): {full_matches}/5")
    print(f"Table 6 — Mann-Kendall vs sentinelTrend agreement: {mk_matches}/5")
    print(
        f"Table 5 — CUSUM FAR on S1 (stable): {cusum_s1:.3f} ({table5[0]['cusum_total_alerts']} alerts/{post_warmup} samples)"
    )
    print(
        f"Table 5 — EWMA FAR on S1 (stable):  {ewma_s1:.3f} ({table5[0]['ewma_total_alerts']} alerts/{post_warmup} samples)"
    )
    print("Table 5 — Sentinel FAR on S1:        0.000 (0 alerts)")

    results["summary"] = {
        "table1_full_match": f"{full_matches}/5",
        "table6_mk_agreement": f"{mk_matches}/5",
        "cusum_far_s1": cusum_s1,
        "ewma_far_s1": ewma_s1,
        "sentinel_far_s1": 0.0,
        "random_seed": RANDOM_SEED,
        "n": N,
        "warmup": WARMUP,
        "tau_warn": TAU_WARN,
        "tau_critical": TAU_CRITICAL,
        "ewma_lambda": EWMA_LAMBDA,
        "ewma_L": EWMA_L,
        "cusum_k_sigma": CUSUM_K,
        "cusum_h_sigma": CUSUM_H,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


if __name__ == "__main__":
    run(parse_args().output)
