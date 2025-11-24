from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import sys

import urllib.request

"""
Phase-2 Trust Continuity Alert emitter.
Polls /coherence/metrics (semantic fields), evaluates risk, and writes
ledger-ready incidents as JSON into artifacts/incidents/.

Usage:
  python -m automation.drift_sentry --window 24h --min-level medium
  python -m automation.drift_sentry --window 1h --api http://localhost:8000
Env (optional):
  API_BASE=http://localhost:8000
  COHERENCE_WARN_THRESHOLD=0.10
  COHERENCE_CRITICAL_THRESHOLD=0.25
  COHERENCE_MODE=demo|production
"""

# ---------- Config ----------
INCIDENTS_DIR = Path("artifacts/incidents")
DEFAULT_API = os.getenv("API_BASE", "http://localhost:8000")
MODE = os.getenv("COHERENCE_MODE", "demo")

WARN_TH = float(os.getenv("COHERENCE_WARN_THRESHOLD", "0.10"))
CRIT_TH = float(os.getenv("COHERENCE_CRITICAL_THRESHOLD", "0.25"))

LEVELS = ("low", "medium", "high")


def http_get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def map_level_from_vol(vol: float) -> str:
    if vol < WARN_TH:
        return "low"
    if vol < CRIT_TH:
        return "medium"
    return "high"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def filename_for(window: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"incident_{ts}_{window}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit trust_continuity_alert incidents (Phase-2).")
    parser.add_argument("--window", default="24h", help="Metrics window (e.g., 1h, 24h, 86400)")
    parser.add_argument("--api", default=DEFAULT_API, help="Base URL for the API service")
    parser.add_argument("--min-level", default="medium", choices=LEVELS,
                        help="Minimum risk level to emit an incident (low|medium|high)")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print, but don't write file")
    args = parser.parse_args()

    # Normalize window to seconds for the API if provided in h/m suffix
    win_arg = args.window
    if win_arg.endswith("h"):
        try:
            window_sec = int(float(win_arg[:-1]) * 3600)
        except ValueError:
            window_sec = 86400
    elif win_arg.endswith("m"):
        try:
            window_sec = int(float(win_arg[:-1]) * 60)
        except ValueError:
            window_sec = 3600
    else:
        # assume seconds or integer-like string
        window_sec = int(win_arg) if win_arg.isdigit() else 86400

    url = f"{args.api}/coherence/metrics?window={window_sec}&include_legacy=false"
    payload = http_get_json(url)

    # Phase-2 fields
    stability = float(payload.get("interactionStability", 0.0))
    volatility = float(payload.get("signalVolatility", 0.0))
    risk_level = payload.get("trustContinuityRiskLevel")
    if not isinstance(risk_level, str):
        # compute from volatility if server didn't label
        risk_level = map_level_from_vol(volatility)
    risk_level = risk_level.lower()

    # gating by min-level
    rank = {lvl: i for i, lvl in enumerate(LEVELS)}
    if rank[risk_level] < rank[args.min_level]:
        print(f"[drift_sentry] risk={risk_level} < min={args.min_level}; no incident emitted.")
        return 0

    # ledger-ready incident body
    incident: Dict[str, Any] = {
        "event": "trust_continuity_alert",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window": args.window,
        "signalStability": round(stability, 4),
        "signalLiquidity": round(volatility, 4),
        "trustContinuityRisk": risk_level, 
        "trace": {
            "source": "coherence_engine_v0.2",
            "upstream": "darshan_signals",
            "api": url,
            "mode": MODE,
            "thresholds": {
                "warn": WARN_TH,
                "critical": CRIT_TH,
            },
        },
    }

    if args.dry_run:
        print(json.dumps(incident, indent=2))
        return 0

    ensure_dir(INCIDENTS_DIR)
    out = INCIDENTS_DIR / filename_for(args.window)
    out.write_text(json.dumps(incident, ensure_ascii=False, indent=2))
    print(f"[drift_sentry] wrote {out} (risk={risk_level})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
