# ---- Base: slim, fast, reproducible ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (minimal; wheels will cover most libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bash ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# ---- Dependencies layer (better caching) ----
FROM base AS deps
# Copy only requirement files first to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# ---- Final runtime image ----
FROM base AS runtime

# Non-root user for safety
RUN useradd -m -u 10001 appuser
USER appuser

# Copy installed site-packages from deps
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin /usr/local/bin

# App files
COPY . /app

# Default Phase-2 environment (can be overridden by --env-file .env)
ENV COHERENCE_MODE=demo \
    COHERENCE_WARN_THRESHOLD=0.10 \
    COHERENCE_CRITICAL_THRESHOLD=0.25 \
    TREND_SENSITIVITY=0.03 \
    STABILITY_HIGH_MIN=0.80 \
    STABILITY_MEDIUM_MIN=0.55 \
    UI_REFRESH_MS=3000 \
    API_BASE=http://localhost:8000

# Persist incidents & artifacts
VOLUME ["/app/artifacts"]

# Expose FastAPI port
EXPOSE 8000

# Healthcheck hits FastAPI /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default: run the API (uvicorn). Can be overridden at runtime to launch Streamlit.
ENV APP_MODULE=app.api:app
CMD ["bash", "-lc", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port 8000 --workers 1"]
