# Run Offline Demo

## Purpose

This guide runs the full offline realized variance research pipeline from local
parquet files under `data/raw/`.

## Prerequisites

```bash
uv sync --all-groups
```

Required files:

- `data/raw/raw_prices.parquet`
- `data/raw/raw_prices.metadata.json`

## Run Offline Pipeline

```bash
uv run covariance-denoiser run-offline-demo --data-dir data/raw --output-dir outputs/demo
```

Or:

```bash
./scripts/run_offline_demo.sh
```

## Output Files

After a successful run, `outputs/demo/` contains:

- `metrics.csv`
- `fold_predictions.csv`
- `model_coefficients.csv`
- `summary.md`
- PNG plots

## Run Notebook

```bash
uv run jupyter lab notebooks/01_offline_research_pipeline.ipynb
```

The notebook teaches the same pipeline stages while calling package functions.
It does not require ClickHouse for runtime.

## Optional One-Time ClickHouse Refresh

Use this only when you need to repopulate `data/raw/`.

```bash
uv run covariance-denoiser refresh-raw-cache \
  --data-dir data/raw \
  --host <host> \
  --port 8123 \
  --username <user> \
  --password <password> \
  --database <db> \
  --table <table> \
  --start-date 2008-01-01 \
  --end-date 2024-12-31 \
  --assets SPY TLT GLD QQQ EEM IWM HYG VNQ
```

Refresh is optional and never needed for default tests, CLI runs, or notebook execution.
