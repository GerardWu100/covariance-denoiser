# Covariance Denoiser

Offline research pipeline that tests whether denoised covariance features improve
next-window realized variance forecasts for a small ETF portfolio.

## What it does

The pipeline runs on a fixed ETF universe (SPY, TLT, GLD, QQQ, EEM, IWM, HYG, VNQ)
stored as local parquet prices:

1. Loads `data/raw/raw_prices.parquet` and builds log returns.
2. Builds a forward realized variance target for an equal-weight, daily-rebalanced
   portfolio: the target at date $t$ sums squared portfolio returns over
   $t+1, \dots, t+h$ and is not admitted to training until the full window is
   known ($h$ = forecast horizon in trading days, default 21).
3. Builds rolling covariance features using two estimators:
   - Random-matrix-theory (Marchenko-Pastur) denoising: eigenvalues of the
     sample correlation matrix below the Marchenko-Pastur noise edge
     $\lambda_+ = (1 + q^{-1/2})^2$ (with $q = T/N$, the observation-to-asset
     ratio) are replaced by their average.
   - Ledoit-Wolf linear shrinkage of the sample covariance matrix (via
     scikit-learn).
4. Trains walk-forward ridge regression models with purged folds (unavailable
   target rows are dropped, features are standardized within each fold) and
   compares them to a persistence (naive) baseline.
5. Exports metrics, predictions, coefficients, and plots to `outputs/demo/`.

Full derivation of the target and features is in `docs/reference/math.md`. The
timing and evaluation rules are in `docs/reference/system_design.md`.

## Requirements

- Python >= 3.13
- No external service is required for the default offline pipeline, tests, or
  notebook.
- ClickHouse is only needed for the optional one-time `refresh-raw-cache`
  command, which repopulates `data/raw/`. Connection details (host, port,
  username, password, database, table) are passed as CLI flags, not read from
  the environment.

## Setup

```bash
uv sync --all-groups
```

## Usage

```bash
uv run covariance-denoiser run-offline-demo --data-dir data/raw --output-dir outputs/demo
```

Runs the full offline pipeline (steps 1-5 above) against the tracked parquet
cache.

```bash
uv run covariance-denoiser refresh-raw-cache --data-dir data/raw \
  --host <host> --port 8123 --username <user> --password <password> \
  --database <db> --table <table> \
  --start-date 2008-01-01 --end-date 2024-12-31 \
  --assets SPY TLT GLD QQQ EEM IWM HYG VNQ
```

Optional: repopulates `data/raw/` from ClickHouse. Never required for normal
runs.

```bash
uv run pytest -v
```

Runs the unit and integration test suite.

```bash
uv run jupyter lab notebooks/01_offline_research_pipeline.ipynb
```

Opens the teaching notebook, which walks the same pipeline stages offline.

Thin wrappers for the two most common commands live in `scripts/`:
`run_offline_demo.sh` and `run_tests.sh`.

## Configuration

The demo pipeline is configured through CLI flags on `run-offline-demo` (see
`src/covariance_denoiser/cli.py`), not a config file. The defaults are:

- `--lookback-days 63`: covariance feature window.
- `--horizon-days 21`: forecast horizon $h$.
- `--min-train-size 252`, `--test-size 21`, `--step-size 21`: walk-forward fold sizes.
- `--annualization-days 252`, `--ridge-alpha 1.0`.

## Layout

```text
data/raw/                        tracked parquet prices and metadata contract
src/covariance_denoiser/         package: CLI, pipelines, estimators, features, models
scripts/                         thin wrappers for the demo and test suite
tests/unit/ tests/integration/   contract and end-to-end tests
notebooks/                       teaching notebook for the full pipeline
docs/reference/, docs/user/      math/system reference and run guide
outputs/demo/                    generated run artifacts (gitignored)
```

## Output

A successful `run-offline-demo` run writes to `outputs/demo/`:

- `metrics.csv`, `fold_predictions.csv`, `model_coefficients.csv`
- `summary.md`
- static PNG plots

Adjacent 21-day targets overlap, so aggregate MAE and RMSE are descriptive
rather than independent-observation inference; see
`docs/reference/system_design.md` for the full evaluation guardrails.

## License

All rights reserved. See [LICENSE](LICENSE).
