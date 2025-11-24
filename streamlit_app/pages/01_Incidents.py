from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---- Config ----
INCIDENTS_DIR = Path("artifacts/incidents")
MODE = os.getenv("COHERENCE_MODE", "demo")
REFRESH_MS = int(os.getenv("UI_REFRESH_MS", "3000"))

st.set_page_config(page_title="Trust Continuity Alerts", layout="wide")

try:
    if MODE == "demo":
        from streamlit_autorefresh import st_autorefresh  # type: ignore

        st_autorefresh(interval=REFRESH_MS, key="alerts_auto")
except Exception:
    pass


# ---- Helpers ----
@st.cache_data(show_spinner=False)
def list_incident_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))

@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_all(directory: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in list_incident_files(directory):
        obj = load_json(p)
        if obj:
            out.append(obj)
    return out

def parse_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def arrow_sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].map(lambda x: isinstance(x, (bytes, bytearray))).any():
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else x
                )
            if df[col].map(type).nunique() > 1:
                df[col] = df[col].astype("string")
        if col.lower() in {"value", "signalstability", "signalliquidity"}:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

def summarize(incidents: List[Dict[str, Any]]) -> Tuple[int, int, int, Optional[datetime]]:
    total = 0
    hi = 0
    med = 0
    last: Optional[datetime] = None
    for inc in incidents:
        risk = inc.get("trustContinuityRisk") or inc.get("trustContinuityRiskLevel")
        ts = inc.get("timestamp") or inc.get("created_at")
        dt = parse_ts(ts) if ts else None

        total += 1
        if isinstance(risk, str):
            r = risk.lower()
            if r == "high":
                hi += 1
            elif r == "medium":
                med += 1
        if dt:
            last = max(last, dt) if last else dt
    return total, hi, med, last

def sorted_newest(incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(i: Dict[str, Any]) -> float:
        ts = i.get("timestamp") or i.get("created_at")
        dt = parse_ts(ts) if ts else None
        return dt.timestamp() if dt else 0.0
    return sorted(incidents, key=key, reverse=True)

def emit_demo_alert(window: str = "1h", min_level: str = "low") -> str:
    """
    Tries to call the Phase-2 automation locally:
    python -m automation.drift_sentry --window 1h --min-level low
    """
    try:
        proc = subprocess.run(
            ["python", "-m", "automation.drift_sentry", "--window", window, "--min-level", min_level],
            capture_output=True,
            text=True,
            check=False,
        )
        return (proc.stdout or proc.stderr).strip()
    except Exception as e:
        return f"Failed to emit demo alert: {e}"


# ---- UI ----
st.title("Trust Continuity Alerts")
st.caption("Ledger-ready incidents emitted by the Coherence Engine. Fields: signalStability, signalLiquidity, trustContinuityRisk, window, trace.")

# Demo button (optional)
with st.sidebar:
    st.subheader("Demo")
    if st.button("Generate Sample Alert (low, 1h)"):
        msg = emit_demo_alert("1h", "low")
        st.toast("Triggered. Check incidents list.")
        st.code(msg or "No output.")

incidents = load_all(INCIDENTS_DIR)
total, high, medium, last_dt = summarize(incidents)

# KPIs
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.3])
with c1:
    st.metric("Total Alerts", total)
with c2:
    st.metric("High", high)
with c3:
    st.metric("Medium", medium)
with c4:
    st.metric("Last Alert (UTC)", last_dt.isoformat().replace("+00:00", "Z") if last_dt else "—")

st.divider()
st.markdown("**Labels per Phase-2:** Signal **Stability**, Signal **Liquidity**, **Trust Continuity Risk**.")

if not incidents:
    st.info("No incidents found yet. When incidents are emitted, they will appear here.")
else:
    # Summary table
    rows = []
    for inc in incidents:
        ts = inc.get("timestamp") or inc.get("created_at") or "—"
        rows.append({
            "Timestamp": ts,
            "Window": inc.get("window", "—"),
            "Signal Stability": inc.get("signalStability", inc.get("interactionStability", "—")),
            "Signal Liquidity": inc.get("signalLiquidity", inc.get("signalVolatility", "—")),
            "Risk": inc.get("trustContinuityRisk", inc.get("trustContinuityRiskLevel", "—")),
            "Event": inc.get("event", "trust_continuity_alert"),
        })
    df = arrow_sanitize(pd.DataFrame(rows))
    st.subheader("Alerts (Table)")
    st.dataframe(df, use_container_width=True)

    st.subheader("Alerts (Cards)")
    for inc in sorted_newest(incidents):
        with st.container(border=True):
            event = inc.get("event", "trust_continuity_alert")
            ts = inc.get("timestamp") or inc.get("created_at") or "—"
            window = inc.get("window", "—")
            stability = inc.get("signalStability", inc.get("interactionStability", "—"))
            liquidity = inc.get("signalLiquidity", inc.get("signalVolatility", "—"))
            risk = inc.get("trustContinuityRisk", inc.get("trustContinuityRiskLevel", "—"))
            trace = inc.get("trace", {})

            left, right = st.columns([2, 1])
            with left:
                st.subheader(event.replace("_", " ").title())
                st.write(f"**Timestamp (UTC):** {ts}")
                st.write(f"**Window:** {window}")
                st.write(f"**Signal Stability:** {stability}")
                st.write(f"**Signal Liquidity:** {liquidity}")
                st.write(f"**Trust Continuity Risk:** {risk}")
            with right:
                st.caption("Trace")
                st.json(trace, expanded=False)

    st.caption(f"Folder: `{INCIDENTS_DIR}` · Mode: `{MODE}`")
