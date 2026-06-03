# Covariance Denoiser

## Research Question

This repository answers one interview-defensible question:

**Do denoised covariance features improve next-window realized variance forecasts for a small ETF universe in an offline reproducible workflow?**

## Project Goal

The project is a compact quantitative research pipeline:

1. Load local parquet prices from `data/raw/`.
2. Build log returns.
3. Build forward realized variance targets.
4. Build denoised covariance features.
5. Train walk-forward models.
6. Evaluate against a naive baseline.
7. Export compact static artifacts.

The default runtime path is fully offline. ClickHouse is optional and only used for one-time raw-cache refresh.

## Repository Layout

- `data/raw/`: tracked parquet prices and metadata contract.
- `src/covariance_denoiser/`: importable package (CLI, pipelines, estimators).
- `scripts/`: thin shell wrappers around package commands.
- `tests/unit/` and `tests/integration/`: contract and end-to-end tests.
- `notebooks/01_offline_research_pipeline.ipynb`: teaching notebook for the full pipeline.
- `outputs/demo/`: generated run artifacts (gitignored).
- `logs/`: runtime logs (gitignored).
- `docs/user/` and `docs/reference/`: run guide and architecture notes.

## Quick Start

### 1) Install dependencies

```bash
uv sync --all-groups
```

### 2) Run tests

```bash
uv run pytest -v
```

### 3) Run the offline demo pipeline

```bash
uv run covariance-denoiser run-offline-demo --data-dir data/raw --output-dir outputs/demo
```

Or use the thin script:

```bash
chmod +x scripts/run_offline_demo.sh
./scripts/run_offline_demo.sh
```

### 4) Run the teaching notebook

```bash
uv run jupyter lab notebooks/01_offline_research_pipeline.ipynb
```

Execute top to bottom (offline, no ClickHouse):

```bash
uv run python -m nbconvert --execute notebooks/01_offline_research_pipeline.ipynb --output /tmp/executed.ipynb
```

## Output Artifacts

A successful demo run writes:

- `metrics.csv`
- `fold_predictions.csv`
- `model_coefficients.csv`
- `summary.md`
- static PNG plots

## Scope

- Offline parquet runtime by default.
- One small ETF universe.
- One naive baseline model.
- One trained linear model.
- One denoiser-driven feature set.

## Out Of Scope

- Dashboards, HTML apps, and templating systems.
- Portfolio optimization as a first-class deliverable.
- Large model families and hyperparameter search.
- Mandatory database access for normal runs.
