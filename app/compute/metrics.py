from typing import List, Dict, Any
from statistics import mean, pstdev
import os

# --- Thresholds from environment (Phase 2.1 externalization) ---
STABILITY_HIGH_MIN = float(os.getenv("STABILITY_HIGH_MIN", "0.80"))
STABILITY_MEDIUM_MIN = float(os.getenv("STABILITY_MEDIUM_MIN", "0.55"))

COHERENCE_WARN_THRESHOLD = float(os.getenv("COHERENCE_WARN_THRESHOLD", "0.10"))
COHERENCE_CRITICAL_THRESHOLD = float(os.getenv("COHERENCE_CRITICAL_THRESHOLD", "0.25"))


def _rolling_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _normalized_volatility(values: List[float]) -> float:
    """
    Normalized volatility = population stdev / mean.
    Returns 0.0 if empty or mean == 0 to avoid div-by-zero.
    """
    if not values:
        return 0.0
    m = mean(values)
    if m == 0:
        return 0.0
    return float(pstdev(values) / m)


def _trend_label(values: List[float]) -> str:
    """
    Dependency-free trend detection:
    compares the mean of the most recent half to the previous half.
    Returns one of: "Improving", "Steady", "Deteriorating".
    """
    n = len(values)
    if n < 6:
        return "Steady"
    mid = n // 2
    prev_m = mean(values[:mid])
    recent_m = mean(values[mid:])
    pct = 0.0 if prev_m == 0 else (recent_m - prev_m) / abs(prev_m)
    if pct >= 0.03:
        return "Improving"
    if pct <= -0.03:
        return "Deteriorating"
    return "Steady"


def _risk_from_liquidity(liq: float) -> str:
    """
    Coarse risk band derived from normalized volatility (aka signal volatility/liquidity).
    Returns one of: "low", "medium", "high".

    Uses env-driven thresholds:
      COHERENCE_WARN_THRESHOLD
      COHERENCE_CRITICAL_THRESHOLD
    """
    if liq < COHERENCE_WARN_THRESHOLD:
        return "low"
    if liq < COHERENCE_CRITICAL_THRESHOLD:
        return "medium"
    return "high"


def compute_metrics(series: List[float], window_sec: int) -> Dict[str, Any]:
    """
    Phase 2 API contract (exact field names):

    interactionStability       : float (rolling mean)
    signalVolatility           : float (normalized stdev/mean)
    trustContinuityRiskLevel   : "low" | "medium" | "high"
    coherenceTrend             : "Improving" | "Steady" | "Deteriorating"

    interpretation: {
        stability        : "High" | "Medium" | "Low"
        trustContinuity  : "Stable" | "At Risk" | "Critical"
        coherenceTrend   : (same as top-level)
    }

    meta: {
        method     : "rolling mean/stdev; half-window trend"
        windowSec  : int
        n          : int
        timestamp  : str (ISO8601)  # set by API layer
    }

    Legacy mirrors (only when include_legacy=true):
      coherenceMean, volatilityIndex, predictedDriftRisk
    """
    stability = _rolling_mean(series)
    liquidity = _normalized_volatility(series)
    risk = _risk_from_liquidity(liquidity)
    trend = _trend_label(series)

    # Lightweight interpretation layer (using env thresholds)
    stability_band = (
        "High"
        if stability >= STABILITY_HIGH_MIN
        else "Medium"
        if stability >= STABILITY_MEDIUM_MIN
        else "Low"
    )
    trust_band = "Stable" if risk == "low" else ("At Risk" if risk == "medium" else "Critical")

    payload: Dict[str, Any] = {
        "interactionStability": round(stability, 4),
        "signalVolatility": round(liquidity, 4),
        "trustContinuityRiskLevel": risk,
        "coherenceTrend": trend,
        "interpretation": {
            "stability": stability_band,
            "trustContinuity": trust_band,
            "coherenceTrend": trend,
        },
        "meta": {
            "method": "rolling mean/stdev; half-window trend",
            "windowSec": window_sec,
            "n": len(series),
            # "timestamp" is added in the API layer for audit friendliness
        },
    }

    # Legacy mirrors (filled/kept by API when include_legacy=true)
    payload["coherenceMean"] = payload["interactionStability"]
    payload["volatilityIndex"] = payload["signalVolatility"]
    payload["predictedDriftRisk"] = payload["trustContinuityRiskLevel"]

    return payload
