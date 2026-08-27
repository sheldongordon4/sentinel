# Sentinel

**Runtime monitoring for LLM output reliability using statistical process control.**

Sentinel is a lightweight, upstream-agnostic framework that continuously monitors the behavioral consistency of deployed large language models. Rather than evaluating individual outputs against ground-truth labels, Sentinel treats LLM output quality as a time-series signal and applies statistical process control (SPC) to detect drift, instability, and hallucination-onset patterns before they compound into production incidents. It measures not just statistical drift but nervous-system stability, trust continuity, and signal integrity over time — and every number it emits has meaning, traceability, and actionability. The runtime monitoring path requires no embeddings, judge models, or ground-truth labels.

## How It Works

Any upstream LLM deployment exposes a paginated HTTP endpoint returning time-stamped records, each carrying a normalized quality score (`sentinelScore` ∈ [0, 1]). Sentinel ingests that stream, computes four interpretable metrics over a rolling observation window, and emits structured, ledger-ready alerts when risk thresholds are breached.
| Metric | What It Measures |
|---|---|
| `interactionStability` | Rolling mean of the quality signal — the process level |
| `signalVolatility` | Coefficient of variation (σ/μ) — how fast the state oscillates (behavioral liquidity) |
| `trustContinuityRiskLevel` | Three-band risk classification derived from volatility: `low`, `medium`, `high` |
| `sentinelTrend` | Half-window directional classifier: `Improving`, `Steady`, `Deteriorating` |

Risk level and trend are deliberately richer together than either alone. A system at `risk=high, trend=Improving` is recovering but not yet clear. A system at `risk=medium, trend=Deteriorating` warrants intervention before it crosses the critical threshold. These two dimensions give operators actionable diagnostic signal, not just an alarm.

---

## Architecture

```
 Upstream LLM Deployment
 ┌─────────────────────┐
 │  Any signal source  │  sentinelScore ∈ [0,1] per record
 └─────────┬───────────┘
           │ async HTTP · paginated · retried
           ▼
 ┌─────────────────────┐
 │   Signal Ingestion  │  app/ingest/signal_client.py
 │   (SignalClient)    │  httpx · HTTP/2 · tenacity backoff
 └─────────┬───────────┘
           │ List[float]
           ▼
 ┌─────────────────────┐
 │  Metric Computation │  app/compute/metrics.py
 │  (Pure Function)    │  stdlib only · deterministic · env-configured
 └─────────┬───────────┘
           │ SentinelMetricsResponse
           ▼
 ┌─────────────────────┐     ┌──────────────────────┐
 │   FastAPI Service   │────▶│  Streamlit Dashboard │
 │  /sentinel/metrics  │     │  (Operations Console)│
 └─────────┬───────────┘     └──────────────────────┘
           │ poll
           ▼
 ┌─────────────────────┐
 │    Drift Sentry     │  automation/drift_sentry.py
 │  (CLI Automation)   │  threshold gate · ledger-ready JSON
 └─────────┬───────────┘
           ▼
   artifacts/incidents/
   incident_<ts>_<window>.json
```

**Three pillars, independently deployable:**

`Signal Ingestion` — the `SignalClient` connects to any upstream system that exposes a paginated quality signal endpoint. The score generation method is entirely up to the integrating organization: LLM self-evaluation, semantic similarity, task accuracy, classifier confidence, or any other normalized [0, 1] quality indicator. Sentinel is upstream-agnostic by design.

`Metric Computation` — a pure function implemented entirely in Python's standard library (`statistics.mean`, `statistics.pstdev`). No external numerical dependencies. Deterministic: same input always produces the same output. Thresholds are loaded from environment variables at import time.

`Governance and Alerting` — the FastAPI service exposes the metric payload as a REST endpoint. The Drift Sentry CLI polls that endpoint and writes self-describing incident JSON when risk exceeds the configured minimum level. Each incident embeds the active threshold values at time of emission, enabling audit and post-mortem without reference to the running system.

---

## Repository Structure

```
sentinel/
├── Makefile
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── pytest.ini
├── .env
│
├── app/
│   ├── __init__.py
│   ├── api.py                    # FastAPI service
│   ├── schemas.py                # Pydantic response models
│   ├── compute/
│   │   └── metrics.py            # Core metric computation
│   ├── ingest/
│   │   └── signal_client.py      # Upstream HTTP client
│   └── persistence/
│       ├── __init__.py
│       ├── base.py               # MetricsStore protocol
│       ├── csv_store.py          # CSV backend with schema migration
│       └── sqlite_store.py       # SQLite backend
│
├── automation/
│   ├── __init__.py
│   └── drift_sentry.py           # Trust continuity alert emitter
├── sentinel_realworld_eval.py     # Legacy TruthfulQA evaluation
├── sentinel_synthetic_eval.py     # Deterministic synthetic evaluation
├── sentinel_realworld_eval_v2.py  # Optional external real-world evaluation
│
├── streamlit_app/
│   ├── app.py                    # Sentinel Operations Console
│   └── pages/
│       └── 01_Incidents.py       # Alert history and summary
│
├── data/
│   └── mock_signals.json         # Local development fixture
│
├── artifacts/
│   └── incidents/                # Auto-generated alert JSON
│
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_api_metrics.py
    ├── test_drift_sentry.py
    ├── test_incident_emitter.py
    ├── test_ingest_persistence.py
    └── test_metrics.py
```

## Quick Start

```bash
git clone https://github.com/sheldongordon4/sentinel.git
cd sentinel
make venv
make env
make api
```

Test the endpoint:

```bash
curl "http://localhost:8000/sentinel/metrics?window=86400"
```

## Real-World Evaluation

The optional evaluation in `sentinel_realworld_eval_v2.py` exercises Sentinel with live model and dataset calls:

- Stable condition: TriviaQA (`rc.nocontext`, validation split)
- Degraded condition: TruthfulQA (`generation`, validation split) with an overconfident system prompt
- Subject model: `gpt-4o-mini`
- Independent judge: `gemini-2.5-flash`

Set both API keys before running it. The evaluation is networked, takes approximately 15–20 minutes, and has an estimated cost of $2–4:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
make eval
```

API keys may be stored in the local `.env` file instead of exported in the shell. `.env` is ignored by Git; never commit it or place real credentials in example configuration files. Rotate the keys if they have been exposed or shared.

The run writes `eval_results_v3.json` and `eval_cache_v3.json`; both are local artifacts excluded from Git. Cached judge scores can become stale after changing models, prompts, datasets, or judge configuration. Failed judge calls are assigned a neutral score of `0.5`, so inspect the evaluation output before drawing conclusions. This workflow is manual and is not part of `make test`.

The earlier TruthfulQA-only experiment remains available as `sentinel_realworld_eval.py`.
It uses GPT-4o-mini for both generation and judging and writes `eval_results_v2.json`
and `eval_cache_v2.json`:

```bash
make eval-v1
# or
python sentinel_realworld_eval.py
```

The v2 workflow above is the current real-world evaluation entry point; the legacy
workflow is retained for reproducibility of earlier paper results.

## Synthetic Evaluation

The deterministic, offline evaluation in `sentinel_synthetic_eval.py` exercises five
known signal patterns and reports metric classification, threshold boundaries,
EWMA/CUSUM comparisons, and Mann-Kendall agreement. It requires no API keys or
network access. It reproduces the paper with the recorded `0.10` warning and
`0.25` critical thresholds; it fails fast if environment overrides change them:

```bash
make synthetic-eval
# or
python sentinel_synthetic_eval.py
```

Results are written to `synthetic_eval_results.json` in the repository directory.
Use `--output PATH` to write them elsewhere. This workflow is manual and is not
part of `make test`.

### Evaluation Results

The recorded v3 run produced the following metrics:

| Window | Mean score | Volatility | Risk | Trend | Expected risk | Result |
|---|---:|---:|---|---|---|---|
| W1: Stable Baseline | 0.8667 | 0.3385 | high | Steady | low | Risk hypothesis failed |
| W2: Hallucination Burst | 0.8375 | 0.3885 | high | Deteriorating | high | Matched |
| W3: Recovery from Drift | 0.8367 | 0.3764 | high | Improving | high | Matched |

W2 and W3 matched the expected risk and trend classifications. W1 matched the expected steady trend but did not validate the low-risk hypothesis: the observed score volatility exceeded the critical threshold. This result indicates that a high mean score alone does not imply low volatility and should be treated as an evaluation finding, not as a production guarantee.

---

## Environment Configuration

`make env` creates a `.env` file with defaults. Override any value before running:

```env
SIGNAL_BASE_URL=https://api.example.com
SIGNAL_API_KEY=changeme
SIGNAL_TIMEOUT_S=10
SIGNAL_PAGE_SIZE=500
SENTINEL_MODE=demo                   # demo | production
SENTINEL_WARN_THRESHOLD=0.10         # volatility threshold for medium risk
SENTINEL_CRITICAL_THRESHOLD=0.25     # volatility threshold for high risk
TREND_SENSITIVITY=0.03               # minimum % change to classify as Improving/Deteriorating
STABILITY_HIGH_MIN=0.80              # mean threshold for High stability band
STABILITY_MEDIUM_MIN=0.55            # mean threshold for Medium stability band
API_BASE=http://0.0.0.0:8000         # used by Drift Sentry
```

---

## API

### `GET /sentinel/metrics`

| Parameter | Default | Description |
|---|---|---|
| `window` | `86400` | Observation window in seconds |
| `include_legacy` | `true` | Include deprecated field aliases |

Example response (`include_legacy=false`):

```json
{
  "interactionStability": 0.8621,
  "signalVolatility": 0.1422,
  "trustContinuityRiskLevel": "medium",
  "sentinelTrend": "Deteriorating",
  "interpretation": {
    "stability": "High",
    "trustContinuity": "At Risk",
    "sentinelTrend": "Deteriorating"
  },
    "method": "rolling mean/stdev; half-window trend",
    "windowSec": 86400,
    "n": 120,
    "timestamp": "2025-11-06T20:12:41.391Z"
  }
}
```

When `include_legacy=true` (default), three additional fields are appended: `sentinelMean`, `volatilityIndex`, and `predictedDriftRisk`. These mirror the canonical fields and exist for backward compatibility only. They will be removed in v0.3.
### `GET /health`

Returns `{"status": "ok"}`. Used by container health checks.

### `GET /status`

Returns the currently active threshold configuration and operational mode.

## Dashboard

```bash
make ui
# or
streamlit run streamlit_app/app.py
```

The Streamlit dashboard provides a live **Sentinel Operations Console** with four KPI metrics, an interpretation summary, and an incidents page. Auto-refreshes every 3 seconds in demo mode.

**Dashboard KPIs:**

- Signal Stability
- Signal Liquidity
- Trust Continuity Risk
- Trust Continuity Alerts


## Drift Sentry

The Drift Sentry polls `/sentinel/metrics` and writes a ledger-ready incident when risk level meets or exceeds the configured minimum:

```bash
make automation-drift     # emit incident if risk is medium or above
make automation-demo      # dry run — prints incident JSON without writing to disk
```

Direct invocation:

```bash
python -m automation.drift_sentry \
  --window 24h \
  --min-level medium \
  --api http://localhost:8000
```

| Flag | Default | Description |
|---|---|---|
| `--window` | `24h` | Observation window (`1h`, `24h`, `86400`, etc.) |
| `--min-level` | `medium` | Minimum risk level to emit: `low`, `medium`, `high` |
| `--api` | `$API_BASE` | Base URL of the Sentinel API service |
| `--dry-run` | off | Print incident JSON without writing to disk |

**Incident schema:**

```json
{
  "event": "trust_continuity_alert",
  "timestamp": "2025-11-10T20:12:41Z",
  "window": "24h",
  "signalStability": 0.84,
  "signalLiquidity": 0.21,
  "trace": {
    "source": "sentinel_v0.2.1",
    "upstream": "signal_source",
    "api": "http://localhost:8000/sentinel/metrics?window=86400&include_legacy=false",
    "mode": "production",
    "thresholds": {
      "warn": 0.10,
      "critical": 0.25
    }
  }
}
```

Incidents are written to `artifacts/incidents/incident_<timestamp>_<window>.json`. Each record is self-describing: active threshold values are embedded at write time, so the conditions that triggered the alert are fully auditable without reference to the running system's current configuration.

---

## Docker

### Build

```bash
docker build -t sentinel-engine:latest .
```

### Run the API

```bash
docker run --rm \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/artifacts:/app/artifacts" \
  sentinel-engine:latest
```

### Run the Dashboard

```bash
docker run --rm \
  -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/artifacts:/app/artifacts" \
  sentinel-engine:latest \
  bash -lc "streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0"
```

The container runs as a non-root user. Incidents are persisted via the mounted `/app/artifacts` volume. The built-in healthcheck polls `GET /health` every 30 seconds with a 5-second timeout. The tracked `.gitkeep` preserves the incident directory in Git; generated artifacts remain excluded from the Docker build context.

---

## Testing

```bash
make test
# or
pytest -v
```

The test suite covers metric semantics, API endpoint correctness, legacy field inclusion and exclusion, threshold patching, Drift Sentry dry-run emission, incident file writing, and the no-write gate when risk falls below the configured minimum level.

---

## Interpretation Reference

**Interaction Stability** (rolling mean)

| Band | Condition | Label |
|---|---|---|
| High | ≥ 0.80 | Nominal operation |
| Medium | 0.55 – 0.79 | Degraded but operational |
| Low | < 0.55 | Critical degradation |

**Signal Volatility** (coefficient of variation)

| Risk | Condition | Trust Continuity |
|---|---|---|
| low | CV < 0.10 | Stable |
| medium | 0.10 ≤ CV < 0.25 | At Risk |
| high | CV ≥ 0.25 | Critical |

**Sentinel Trend** (half-window delta)

| Label | Condition |
|---|---|
| Improving | Δ ≥ +3% |
| Steady | −3% < Δ < +3% |
| Deteriorating | Δ ≤ −3% |

All thresholds are configurable via `.env`.

---

## Backward Compatibility

Legacy field names from v0.1 (`sentinelMean`, `volatilityIndex`, `predictedDriftRisk`) are preserved by default (`include_legacy=true`) and mirror the canonical Phase 2 fields. Set `include_legacy=false` to receive the canonical schema only. Legacy fields will be removed in v0.3 following a deprecation period.

---

## Makefile Reference

| Command | Purpose |
|---|---|
| `make venv` | Create virtualenv and install dependencies |
| `make env` | Create `.env` with default configuration |
| `make api` | Start FastAPI service on port 8000 |
| `make ui` | Start Streamlit dashboard |
| `make metrics` | `GET /sentinel/metrics` (legacy fields included) |
| `make metrics_new` | `GET /sentinel/metrics?include_legacy=false` |
| `make metrics_legacy` | `GET /sentinel/metrics?include_legacy=true` |
| `make health` | `GET /health` |
| `make status` | `GET /status` — active threshold config |
| `make automation-drift` | Run Drift Sentry (24h window, min-level=low) |
| `make automation-demo` | Dry-run Drift Sentry (1h window) |
| `make test` | Run pytest |
| `make fmt` | Format with black |
| `make lint` | Lint with pylint |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run API container |
| `make clean` | Remove virtualenv, caches, and incident files |

---

## Roadmap

1. Externalize thresholds via `.env` — complete
2. Expose trend interpretation layer (`rising`, `stable`, `declining`)
3. Emit incidents based on combined trend + risk logic
4. Integrate sentinel metrics with multi-agent governance dashboard
5. Add combined API + UI Docker service for single-container deployment

---

## License

MIT License © 2025 Sheldon H. Gordon

**Version:** 0.2.1 · **Last updated:** August 26, 2026
