# Coherence Engine — Phase 2 Update (Semantic Metrics & Interpretation Layer)

The **Coherence Engine** is the runtime layer that maintains **stable, trustworthy behavior** across human and machine agents.  

## Overview

The **Coherence Engine** measures not just statistical drift but **nervous-system stability**, **trust continuity**, and **signal coherence** over time.  
It ingests streaming signal summaries (linguistic, biometric, behavioral, operational), computes **semantic coherence metrics**, detects **trust continuity risks**, and emits **ledger-ready incident reports** for governance and recovery where every number has meaning, traceability, and actionability.

### Core Features

- **Signal Ingestion:** Retrieves summaries from Darshan’s `/signals/summary` endpoint or local mock JSON.
- **Semantic Metrics:**
  - `interactionStability` — how steady the system’s internal state remains  
  - `signalVolatility` — how fast the state oscillates (behavioral liquidity)  
  - `trustContinuityRiskLevel` — likelihood of coherence breakdown (`low | medium | high`)  
  - `coherenceTrend` — trajectory across the window (`Improving | Steady | Deteriorating`)
- **Interpretation Block:** Maps numeric bands to human-readable labels for decision-making.
- **API Endpoints:**
  - `GET /coherence/metrics` → semantic coherence summary  
  - `GET /health`, `GET /status` → diagnostics  
- **Persistence Layer:** CSV or SQLite rolling data store.
- **Streamlit Dashboard:** Live **Coherence Operations Console**.
- **Automation:** *Drift Sentry* emits `trust_continuity_alert` events (ledger-ready format).
- **Docker Support:** Lightweight container image for deployment or demos.


## Folder Structure

```
coherence_engine/
│
├── .env
├── Makefile
├── requirements.txt
├── Dockerfile
├── .dockerignore
│
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── schemas.py
│   │
│   ├── compute/
│   │   └── metrics.py
│   │
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── csv_store.py
│   │   └── sqlite_store.py
│   │
│   └── ingest/
│       └── darshan_client.py
│
├── automation/
│   ├── __init__.py
│   └── drift_sentry.py
│
├── artifacts/
│   └── incidents/
│       └── (auto-generated JSON trust_continuity_alerts)
│
├── data/
│   └── mock_signals.json
│
├── streamlit_app/
│   ├── app.py
│   └── pages/
│       └── 01_Incidents.py
│
└── tests/
```


## Quick Start

```bash
git clone https://github.com/<your-username>/coherence_engine.git
cd coherence_engine
make install
make env
make api
```


## Environment Configuration (Phase 2)

Create `.env` in the project root and include:

```env
COHERENCE_MODE=demo                # demo | production
COHERENCE_WARN_THRESHOLD=0.10
COHERENCE_CRITICAL_THRESHOLD=0.25
TREND_SENSITIVITY=0.03
STABILITY_HIGH_MIN=0.80
STABILITY_MEDIUM_MIN=0.55
UI_REFRESH_MS=3000
```

Legacy PSI fields (`DRIFT_PSI_WARN`, `DRIFT_PSI_CRIT`) can remain temporarily for backward compatibility.


## Run the FastAPI Service

```bash
make api
# or
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Test:

```bash
curl "http://localhost:8000/coherence/metrics?window=86400"
```

### Example Response

```json
{
  "interactionStability": 0.8621,
  "signalVolatility": 0.1422,
  "trustContinuityRiskLevel": "low",
  "coherenceTrend": "Steady",
  "interpretation": {
    "stability": "High",
    "trustContinuity": "Stable",
    "coherenceTrend": "Steady"
  },
  "meta": {
    "method": "rolling mean/stdev; half-window trend",
    "windowSec": 86400,
    "n": 120,
    "timestamp": "2025-11-06T20:12:41.391Z"
  },
  "coherenceMean": 0.8621,
  "volatilityIndex": 0.1422,
  "predictedDriftRisk": "low"
}
```


## Dashboard

Launch the Streamlit verification UI:

```bash
make ui
# or
streamlit run streamlit_app/app.py
```

### Dashboard Labels (Phase 2)

- **Signal Stability**  
- **Signal Liquidity**  
- **Trust Continuity Risk**  
- **Trust Continuity Alerts**

When `COHERENCE_MODE=demo`, the dashboard auto-refreshes every 3 s.


## Automation — Trust Continuity Alerts

`automation/drift_sentry.py` emits ledger-ready events like:

```json
{
  "event": "trust_continuity_alert",
  "timestamp": "2025-11-10T20:12:41Z",
  "window": "24h",
  "signalStability": 0.84,
  "signalLiquidity": 0.21,
  "trustContinuityRisk": "medium",
  "trace": {
    "source": "coherence_engine_v0.1",
    "upstream": "darshan_signals"
  }
}
```

You can trigger alerts manually:

```bash
make automation-drift        # normal cadence
make automation-demo         # fast 1h demo mode
```


## Docker Deployment

### Build Image

```bash
docker build -t coherence-engine:latest .
```

### Run API Container

```bash
docker run --rm -p 8000:8000   --env-file .env   -v "$(pwd)/artifacts:/app/artifacts"   coherence-engine:latest
```

### Run Streamlit Dashboard

```bash
docker run --rm -p 8501:8501   --env-file .env   -v "$(pwd)/artifacts:/app/artifacts"   coherence-engine:latest   bash -lc "streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0"
```

### Docker Ignore

Your `.dockerignore` excludes local caches, test data, `.venv`, and `artifacts/` for smaller, cleaner builds.  
You can preserve folder structure using a `.gitkeep` file inside `artifacts/incidents`.


## Backward Compatibility

- **Legacy fields remain** (default) for any downstream code.  
- Clients can set `include_legacy=false` to migrate to new naming.  
- In **v0.2**, legacy names are preserved.  
- In **v0.3**, they will be fully removed after deprecation warnings.


## Makefile Highlights

| Command | Purpose |
|----------|----------|
| `make install` | Install dependencies |
| `make env` | Prepare `.env` with Phase-2 fields |
| `make api` | Run FastAPI service |
| `make ui` | Run Streamlit dashboard |
| `make metrics` | GET /coherence/metrics (default) |
| `make metrics_new` | include_legacy = false |
| `make metrics_legacy` | include_legacy = true |
| `make automation-drift` | Emit trust continuity alert |
| `make automation-demo` | Emit sample demo alert |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run container |
| `make test` | Run pytest |
| `make clean` | Clean env & caches |


## Testing

```bash
make test
# or
pytest -v
```

Unit tests cover metric semantics, endpoint logic, backward compatibility, and automation emission.


## Interpretation Bands (Defaults)

| Metric | Rule | Label |
|--------|------|-------|
| `interactionStability ≥ 0.80` | High |
| `0.55 ≤ interactionStability < 0.80` | Medium |
| `< 0.55` | Low |
| `signalVolatility < 0.10` | Risk = low |
| `0.10–0.25` | Risk = medium |
| `≥ 0.25` | Risk = high |
| Trend Δ ≥ +3 % | Improving |
| Trend Δ ≤ −3 % | Deteriorating |
| Otherwise | Steady |


## Phase 2 Roadmap

1. Externalize thresholds via `.env` (complete).  
2. Expose trend interpretation layer (`rising`, `stable`, `declining`).  
3. Emit incidents based on trend + risk logic.  
4. Integrate coherence metrics with multi-agent governance dashboard.  
5. Add combined API + UI Docker service for single-container deployment.  


## License

MIT License © 2025  
Coherence Engine Project — Developed by Sheldon H. Gordon  

**Version:** 0.2.1  
**Last Updated:** November 11, 2025  
