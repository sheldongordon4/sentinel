from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import urllib.request
import urllib.error

# ---- Config ----
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
MODE = os.getenv("COHERENCE_MODE", "demo")
REFRESH_MS = int(os.getenv("UI_REFRESH_MS", "3000"))

st.set_page_config(page_title="Coherence Operations Console", layout="wide")

try:
    if MODE == "demo":
        from streamlit_autorefresh import st_autorefresh  # type: ignore

        st_autorefresh(interval=REFRESH_MS, key="metrics_auto")
except Exception:
    pass


# ---- Helpers ----
@st.cache_data(show_spinner=False)
def fetch_json(url: str, timeout: int = 8) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def arrow_sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].map(lambda x: isinstance(x, (bytes, bytearray))).any():
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else x
                )
            # if mixed object types remain, cast to string
            if df[col].map(type).nunique() > 1:
                df[col] = df[col].astype("string")
        if col.lower() in {"value", "interactionstability", "signalvolatility"}:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def metrics_endpoint(base: str, window_sec: int, include_legacy: bool) -> str:
    return f"{base}/coherence/metrics?window={window_sec}&include_legacy={'true' if include_legacy else 'false'}"


# ---- Sidebar Controls ----
st.sidebar.title("Controls")
api_base = st.sidebar.text_input("API Base", value=API_BASE, help="Where the FastAPI service is running.")
window_sec = st.sidebar.number_input("Window (sec)", min_value=60, step=60, value=86400)
include_legacy = st.sidebar.toggle("Show legacy fields", value=False)
st.sidebar.caption(f"Mode: **{MODE}** · Auto-refresh: **{REFRESH_MS} ms** (demo)")

# ---- Fetch and Render Metrics ----
st.title("Coherence Operations Console")
st.caption("Phase-2 semantics: **Signal Stability**, **Signal Liquidity**, **Trust Continuity Risk**, **Trend**")

url = metrics_endpoint(api_base, int(window_sec), include_legacy)
error_box = st.empty()

try:
    payload = fetch_json(url)
    error_box.empty()
except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
    error_box.error(f"Could not reach API: {e}")
    st.stop()

# Pull Phase-2 fields with graceful fallbacks
stability = payload.get("interactionStability") or payload.get("coherenceMean")
volatility = payload.get("signalVolatility") or payload.get("volatilityIndex")
risk = payload.get("trustContinuityRiskLevel") or payload.get("predictedDriftRisk")
trend = payload.get("coherenceTrend", "—")
interp = payload.get("interpretation", {})
meta = payload.get("meta", {})

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Signal Stability", f"{stability:.4f}" if isinstance(stability, (int, float)) else str(stability))
with c2:
    st.metric("Signal Liquidity", f"{volatility:.4f}" if isinstance(volatility, (int, float)) else str(volatility))
with c3:
    st.metric("Trust Continuity Risk", str(risk).capitalize() if risk else "—")
with c4:
    st.metric("Trend", trend)

st.divider()

# Interpretation table
interp_rows = [
    {"Metric": "Stability Band", "Label": interp.get("stability", "—")},
    {"Metric": "Trust Continuity", "Label": interp.get("trustContinuity", "—")},
    {"Metric": "Trend", "Label": interp.get("coherenceTrend", trend)},
]
df_interp = arrow_sanitize(pd.DataFrame(interp_rows))
st.subheader("Interpretation")
st.dataframe(df_interp, use_container_width=True)

# Meta + Raw JSON
left, right = st.columns([1, 1])
with left:
    st.subheader("Meta")
    meta_rows = [{"Key": k, "Value": v} for k, v in meta.items()]
    df_meta = arrow_sanitize(pd.DataFrame(meta_rows))
    st.dataframe(df_meta, use_container_width=True)
with right:
    with st.expander("Raw JSON"):
        st.json(payload, expanded=False)

st.caption(f"Source: `{url}` · Timestamp: {datetime.now(timezone.utc).isoformat()}`")
