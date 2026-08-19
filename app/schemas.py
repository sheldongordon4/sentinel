from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SignalSummary(BaseModel):
    timestamp: datetime
    signal_id: str
    sentinelScore: float = Field(..., ge=0.0, le=1.0)
    agentStates: Dict[str, int] = Field(default_factory=dict)
    eventCount: int = Field(..., ge=0)


class SignalPage(BaseModel):
    data: List[SignalSummary]
    next_page: Optional[str] = None


@dataclass
class MetricsRecord:
    ts_utc: datetime
    window_sec: int
    n: int
    mean: float
    stdev: float
    drift_risk: str
    source: str
    request_id: Optional[str] = None
    sentinel_trend: str = ""

class SentinelMetricsResponse(BaseModel):
    """
    Phase 2 Sentinel metrics response model.
    """

    # --- Phase 2 fields (canonical) ---
    interactionStability: float = Field(..., description="Rolling mean of stability")
    signalVolatility: float = Field(..., description="Normalized volatility (stdev/mean)")
    trustContinuityRiskLevel: Literal["low", "medium", "high"] = Field(
        ..., description="Risk derived from signal volatility"
    )
    sentinelTrend: Literal["Improving", "Steady", "Deteriorating"] = Field(
        ..., description="Trend label across the window"
    )

    interpretation: Dict[str, str] = Field(
        ..., description="Human-readable summary for quick decision support"
    )
    meta: Dict[str, Any] = Field(
        ..., description="Computation metadata including windowSec, n, timestamp"
    )

    # --- Legacy mirrors (optional; only included when include_legacy=true) ---
    sentinelMean: Optional[float] = Field(
        None, description="Legacy mirror of interactionStability"
    )
    volatilityIndex: Optional[float] = Field(
        None, description="Legacy mirror of signalVolatility"
    )
    predictedDriftRisk: Optional[str] = Field(
        None, description="Legacy mirror of trustContinuityRiskLevel"
    )
