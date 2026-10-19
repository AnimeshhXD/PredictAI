# =============================================================================
# PredictAI – Makefile
# =============================================================================
# Targets
#   setup      – create venv and install all dependencies
#   test       – run the full pytest suite with coverage
#   episodes   – generate synthetic what-if scenario episodes (data pipeline)
#   memory     – build and index FAISS episodic memory
#   baselines  – run B1/B2/B3 baseline forecasts
#   evaluate   – run forecasting evaluation, tables, and journal figures
#   summary    – generate results/summary.md with all paper citation numbers
#   app        – launch the Streamlit UI
#   reproduce  – full end-to-end: test → episodes → memory → baselines → evaluate → summary
#
# Usage:
#   make setup
#   make test
#   make reproduce
#   make app
# =============================================================================

PYTHON      := python
VENV        := .venv
PIP         := $(VENV)/Scripts/pip          # Windows path inside venv
PY          := $(VENV)/Scripts/python       # Windows path inside venv
PYTEST      := $(PY) -m pytest
STREAMLIT   := $(VENV)/Scripts/streamlit

# Fall back to system python when venv doesn't exist yet
ifeq ($(wildcard $(VENV)/Scripts/python),)
  PY        := $(PYTHON)
  PYTEST    := $(PYTHON) -m pytest
  STREAMLIT := streamlit
endif

SRC         := src
TESTS       := tests
RESULTS     := results
EPISODES    := scripts/generate_episodes.py
MEMORY      := scripts/build_memory.py
BASELINES   := scripts/run_baselines.py
EVALUATE    := scripts/evaluate.py
SUMMARY     := scripts/generate_summary_md.py
APP         := app.py

SETUP_DONE  := $(VENV)/.setup_done

.PHONY: all setup test episodes memory baselines evaluate summary app reproduce clean help

# ── Default ───────────────────────────────────────────────────────────────────
all: help

# ── help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  PredictAI – available targets"
	@echo "  ─────────────────────────────────────────────────────────────────────"
	@echo "  setup      Create .venv and install requirements.txt"
	@echo "  test       Run pytest test suite (fails on data leakage)"
	@echo "  episodes   Generate 90 what-if scenario episodes -> data/processed/"
	@echo "  memory     Build and persist FAISS episodic memory -> data/processed/memory/"
	@echo "  baselines  Run B1/B2/B3 baseline forecasts -> results/baselines.csv"
	@echo "  evaluate   Run master evaluation, tables, and figures -> results/"
	@echo "  summary    Generate paper citation report -> results/summary.md"
	@echo "  app        Launch Streamlit UI (http://localhost:8501)"
	@echo "  reproduce  Full end-to-end pipeline: test -> episodes -> memory -> baselines -> evaluate -> summary"
	@echo "  clean      Remove __pycache__, .pytest_cache, .llm_cache"
	@echo ""

# ── setup ─────────────────────────────────────────────────────────────────────
setup: $(SETUP_DONE)

$(SETUP_DONE): requirements.txt
	@echo ">>> Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo ">>> Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(SETUP_DONE)
	@echo ">>> Setup complete."

# ── test ──────────────────────────────────────────────────────────────────────
test:
	@echo ">>> Running test suite..."
	$(PYTEST) $(TESTS)/ -v --tb=short

# ── episodes ──────────────────────────────────────────────────────────────────
episodes:
	@echo ">>> Generating what-if scenario episodes..."
	$(PY) $(EPISODES)

# ── memory ────────────────────────────────────────────────────────────────────
memory:
	@echo ">>> Building and persisting FAISS episodic memory..."
	$(PY) $(MEMORY)

# ── baselines ─────────────────────────────────────────────────────────────────
baselines:
	@echo ">>> Running baseline models (B1, B2, B3)..."
	$(PY) $(BASELINES)

# ── evaluate ──────────────────────────────────────────────────────────────────
evaluate:
	@echo ">>> Running evaluation pipeline, generating tables and figures..."
	$(PY) $(EVALUATE)

# ── summary ───────────────────────────────────────────────────────────────────
summary:
	@echo ">>> Generating results/summary.md for paper citation..."
	$(PY) $(SUMMARY)

# ── app ───────────────────────────────────────────────────────────────────────
app:
	@echo ">>> Launching Streamlit app at http://localhost:8501 ..."
	$(STREAMLIT) run $(APP)

# ── reproduce ─────────────────────────────────────────────────────────────────
reproduce: test episodes memory baselines evaluate summary
	@echo ""
	@echo "====================================================================="
	@echo ">>> Full reproduction pipeline completed successfully."
	@echo "    - Paper Summary: $(RESULTS)/summary.md"
	@echo "    - Main Table:    $(RESULTS)/table1_main_results.csv"
	@echo "    - Ablations:     $(RESULTS)/table2_ablation_study.csv"
	@echo "    - Scenarios:     $(RESULTS)/table3_by_scenario_type.csv"
	@echo "    - Figures (PDF): $(RESULTS)/figures/"
	@echo "====================================================================="
	@echo ""

# ── clean ─────────────────────────────────────────────────────────────────────
clean:
	@echo ">>> Cleaning caches..."
	-@if exist .llm_cache     rmdir /s /q .llm_cache
	-@if exist .pytest_cache  rmdir /s /q .pytest_cache
	-@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo ">>> Done."
