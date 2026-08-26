from datetime import datetime, timezone
from fastapi import FastAPI, APIRouter, Query
from dotenv import load_dotenv

load_dotenv()

from .schemas import SentinelMetricsResponse
from .compute.metrics import compute_metrics
from .persistence.csv_store import load_series

# --- FastAPI initialization ---
app = FastAPI(title="Sentinel Engine API", version="0.2.0")
router = APIRouter()

# --- Main metrics endpoint ---
@router.get(
    "/sentinel/metrics",
    response_model=SentinelMetricsResponse,
    response_model_exclude_none=True,
)
def get_metrics(
    window: int = Query(86400, description="Window in seconds"),
    include_legacy: bool = Query(
        True,
        description="Return deprecated fields for backward compatibility",
    ),
):

    series = load_series(window)
    payload = compute_metrics(series, window_sec=window)

    # Phase-2: metadata under meta
    payload["meta"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["meta"]["windowSec"] = window
    payload["meta"]["n"] = len(series)

    if not include_legacy:
        payload.pop("sentinelMean", None)
        payload.pop("volatilityIndex", None)
        payload.pop("predictedDriftRisk", None)

    return payload

# --- Health endpoint ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Status endpoint ---
@app.get("/status")
def status():
    from os import getenv
    return {
        "mode": getenv("SENTINEL_MODE", "demo"),
        "warn_threshold": getenv("SENTINEL_WARN_THRESHOLD", "0.10"),
        "critical_threshold": getenv("SENTINEL_CRITICAL_THRESHOLD", "0.25"),
        "trend_sensitivity": getenv("TREND_SENSITIVITY", "0.03"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# --- Router registration ---
app.include_router(router)
