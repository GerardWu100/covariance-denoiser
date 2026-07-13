# System Design

## Layer 1: Raw Cache Contract

Runtime inputs are local parquet files under `data/raw/`.

Required files:

- `data/raw/raw_prices.parquet`
- `data/raw/raw_prices.metadata.json`

The validator checks file existence, metadata keys, schema, row count, asset set,
and date span before any analytics run.

## Layer 2: Data And Target Construction

The data module loads long-format prices from validated parquet.

The target module constructs exact daily-rebalanced equal-weight portfolio log
returns, then sums their future squares over a fixed horizon. Each target carries
an explicit availability lag equal to that horizon.

## Layer 3: Feature Engineering

The feature module builds a small denoiser-focused set from rolling windows:

- sample covariance diagnostics
- Ledoit-Wolf diagnostics
- RMT diagnostics
- compact ratio and trailing volatility features

Feature rows are indexed by in-sample window end timestamps.

## Layer 4: Walk-Forward Modeling

The modeling layer runs expanding-window walk-forward splits. It purges labels
whose future-return windows are incomplete at the first test timestamp, then fits
the feature scaler and ridge model on the remaining training rows only.

Models:

- naive last-value baseline
- ridge regression

Ridge forecasts are floored at zero because realized variance cannot be negative.
The naive forecast rolls daily: at timestamp `t`, it uses the forward target
stamped `t-h`, which has just become fully observable.

## Layer 5: Evaluation And Artifacts

The evaluation layer computes deterministic MAE and RMSE.

The artifact layer exports:

- `metrics.csv`
- `fold_predictions.csv`
- `model_coefficients.csv`
- `summary.md`
- static PNG plots

## Runtime Paths

### Default Runtime Path

`run-offline-demo` uses only local files under `data/raw/`.

### Optional Refresh Path

`refresh-raw-cache` is the only command that touches ClickHouse.
It is optional and never runs implicitly.
