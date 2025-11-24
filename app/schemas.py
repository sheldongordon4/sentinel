from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal, Any

class CoherenceMetricsResponse(BaseModel):
    """
    Phase 2 Coherence metrics response model.
    """

    # --- Phase 2 fields (canonical) ---
    interactionStability: float = Field(..., description="Rolling mean of stability")
    signalVolatility: float = Field(..., description="Normalized volatility (stdev/mean)")
    trustContinuityRiskLevel: Literal["low", "medium", "high"] = Field(
        ..., description="Risk derived from signal volatility"
    )
    coherenceTrend: Literal["Improving", "Steady", "Deteriorating"] = Field(
        ..., description="Trend label across the window"
    )

    interpretation: Dict[str, str] = Field(
        ..., description="Human-readable summary for quick decision support"
    )
    meta: Dict[str, Any] = Field(
        ..., description="Computation metadata including windowSec, n, timestamp"
    )

    # --- Legacy mirrors (optional; only included when include_legacy=true) ---
    coherenceMean: Optional[float] = Field(
        None, description="Legacy mirror of interactionStability"
    )
    volatilityIndex: Optional[float] = Field(
        None, description="Legacy mirror of signalVolatility"
    )
    predictedDriftRisk: Optional[str] = Field(
        None, description="Legacy mirror of trustContinuityRiskLevel"
    )
