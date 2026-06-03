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

The target module constructs forward realized variance from future equal-weight
log returns over a fixed horizon. This keeps the forecasting target explicit and
avoids lookahead bias.

## Layer 3: Feature Engineering

The feature module builds a small denoiser-focused set from rolling windows:

- sample covariance diagnostics
- Ledoit-Wolf diagnostics
- RMT diagnostics
- compact ratio and trailing volatility features

Feature rows are indexed by in-sample window end timestamps.

## Layer 4: Walk-Forward Modeling

The modeling layer runs expanding-window walk-forward splits with strict temporal
ordering (`train < test` for every fold).

Models:

- naive last-value baseline
- ridge regression

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
