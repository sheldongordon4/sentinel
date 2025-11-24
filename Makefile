# Makefile — Coherence Engine (Phase 2)

SHELL := /bin/bash
PY := $(shell command -v python3 || command -v python)
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON := $(VENV_BIN)/python

APP_HOST := 0.0.0.0
APP_PORT := 8000
APP_MODULE := app.api:app

ENV_FILE := .env
ENV_EXAMPLE := .env.example
INCIDENTS_DIR := /app/artifacts
DOCKER_IMAGE := coherence-engine:latest

.DEFAULT_GOAL := help

# -------------------------
# Helpers
# -------------------------

$(ENV_FILE):
	@echo "[make] Creating $(ENV_FILE) with default coherence settings"
	@echo "COHERENCE_MODE=demo" > $(ENV_FILE)
	@echo "COHERENCE_WARN_THRESHOLD=0.10" >> $(ENV_FILE)
	@echo "COHERENCE_CRITICAL_THRESHOLD=0.25" >> $(ENV_FILE)
	@echo "TREND_SENSITIVITY=0.02" >> $(ENV_FILE)
	@echo "STABILITY_HIGH_MIN=0.80" >> $(ENV_FILE)
	@echo "STABILITY_MEDIUM_MIN=0.55" >> $(ENV_FILE)
	@echo "UI_REFRESH_MS=3000" >> $(ENV_FILE)
	@echo "API_BASE=http://$(APP_HOST):$(APP_PORT)" >> $(ENV_FILE)
	@echo "[make] Wrote defaults to $(ENV_FILE)"

# Create venv + install deps if needed
$(VENV): requirements.txt
	@echo "[make] Creating virtualenv at $(VENV)"
	@$(PY) -m venv $(VENV)
	@echo "[make] Installing dependencies"
	@. $(VENV_BIN)/activate; pip install -r requirements.txt

# -------------------------
# Core dev targets
# -------------------------

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*$$' Makefile | sed 's/:.*//' | sort -u

venv: $(VENV)

env: $(ENV_FILE)

test: $(ENV_FILE) $(VENV)
	@echo "[make] Running tests"
	@. $(VENV_BIN)/activate; pytest -q

fmt: $(VENV)
	@echo "[make] Formatting with black"
	@. $(VENV_BIN)/activate; black app automation streamlit_app tests

lint: $(VENV)
	@echo "[make] Linting with pylint"
	@. $(VENV_BIN)/activate; pylint app

# -------------------------
# API & UI
# -------------------------

api: $(ENV_FILE) $(VENV)
	@echo "[make] Starting FastAPI on $(APP_HOST):$(APP_PORT)"
	@. $(VENV_BIN)/activate; uvicorn $(APP_MODULE) --host $(APP_HOST) --port $(APP_PORT) --reload

ui: $(ENV_FILE) $(VENV)
	@echo "[make] Starting Streamlit dashboard"
	@. $(VENV_BIN)/activate; streamlit run streamlit_app/app.py

# -------------------------
# Metrics & status helpers
# -------------------------

metrics:
	@echo "[make] GET /coherence/metrics (default include_legacy=true)"
	@curl -s "http://$(APP_HOST):$(APP_PORT)/coherence/metrics" | python -m json.tool

metrics_new:
	@echo "[make] GET /coherence/metrics?include_legacy=false"
	@curl -s "http://$(APP_HOST):$(APP_PORT)/coherence/metrics?include_legacy=false" | python -m json.tool

metrics_legacy:
	@echo "[make] GET /coherence/metrics?include_legacy=true"
	@curl -s "http://$(APP_HOST):$(APP_PORT)/coherence/metrics?include_legacy=true" | python -m json.tool

health:
	@echo "[make] GET /health"
	@curl -s "http://$(APP_HOST):$(APP_PORT)/health" || true
	@echo

status:
	@echo "[make] GET /status"
	@curl -s "http://$(APP_HOST):$(APP_PORT)/status" | python -m json.tool || true

# -------------------------
# Automation / incidents
# -------------------------

automation-drift: $(ENV_FILE) $(VENV)
	@echo "[make] Running drift_sentry once (24h window, low min-level)"
	@. $(VENV_BIN)/activate; $(PYTHON) -m automation.drift_sentry --window 24h --min-level low

automation-demo: $(ENV_FILE) $(VENV)
	@echo "[make] Running drift_sentry in demo mode (1h window, low min-level, dry-run)"
	@. $(VENV_BIN)/activate; $(PYTHON) -m automation.drift_sentry --window 1h --min-level low --dry-run

# -------------------------
# Docker
# -------------------------

docker-build:
	@echo "[make] Building docker image $(DOCKER_IMAGE)"
	@docker build -t $(DOCKER_IMAGE) .

docker-run:
	@echo "[make] Running docker image $(DOCKER_IMAGE)"
	@docker run --rm \
		-p $(APP_PORT):8000 \
		-e COHERENCE_MODE=demo \
		-v "$$(pwd)/artifacts:$(INCIDENTS_DIR)" \
		$(DOCKER_IMAGE)

# -------------------------
# Cleanup
# -------------------------

clean:
	@echo "[make] Cleaning venv, caches, and incident JSON"
	@rm -rf $(VENV) .pytest_cache __pycache__
	@find app automation streamlit_app tests -name '__pycache__' -type d -exec rm -rf {} +
	@rm -f artifacts/incidents/*.json || true

.PHONY: help venv env test fmt lint api ui metrics metrics_new metrics_legacy health status \
        automation-drift automation-demo docker-build docker-run clean
