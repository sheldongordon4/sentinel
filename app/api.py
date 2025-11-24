from datetime import datetime, timezone
from fastapi import FastAPI, APIRouter, Query

from .schemas import CoherenceMetricsResponse
from .compute.metrics import compute_metrics

# --- Series loader (try existing; fallback mock) ---
try:
    from .persistence.csv_store import load_series
except Exception:
    def load_series(window_sec: int):
        # fallback simple synthetic signal for local/dev
        return [0.8, 0.82, 0.81, 0.83, 0.84, 0.85, 0.86]

# --- FastAPI initialization ---
app = FastAPI(title="Coherence Engine API", version="0.2.0")
router = APIRouter()

# --- Main metrics endpoint ---
@router.get(
    "/coherence/metrics",
    response_model=CoherenceMetricsResponse,
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
        payload.pop("coherenceMean", None)
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
        "mode": getenv("COHERENCE_MODE", "demo"),
        "warn_threshold": getenv("COHERENCE_WARN_THRESHOLD", "0.10"),
        "critical_threshold": getenv("COHERENCE_CRITICAL_THRESHOLD", "0.25"),
        "trend_sensitivity": getenv("TREND_SENSITIVITY", "0.03"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# --- Router registration ---
app.include_router(router)
