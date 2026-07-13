# GUIDE_tests

## Part 1: Conceptual explanation

`tests/` protects the offline data contract, covariance features, target timing,
portfolio-return math, walk-forward label availability, nonnegative forecasts,
daily persistence updates, the teaching notebook, and the end-to-end artifact bundle.

The timing tests use row positions deliberately. If a target contains $h$ future
returns, the last admitted training label must be stamped $h$ rows before the
first test feature. The portfolio test separately checks that asset simple
returns are averaged before conversion back to a portfolio log return.

## Part 2: Code reference

- `unit/data/`: cache path, metadata, schema, and loader behavior.
- `unit/features/`: rolling timestamp alignment and covariance symmetry.
- `unit/targets/`: log-return calculation, future-only target windows, and exact equal-weight aggregation.
- `unit/models/`: purged fold boundaries, forecast outputs, zero floor, and deterministic metrics.
- `unit/notebooks/`: notebook structure and offline-runtime teaching contract.
- `integration/pipelines/`: command-line contract and complete output generation.

Run all tests with `uv run pytest -v` or `./scripts/run_tests.sh`.

## Part 3: Short journal

- 2026-07-13: Regression tests now fail if future target windows leak into training or ridge emits a negative variance forecast.
