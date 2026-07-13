# GUIDE_src

## Part 1: Conceptual explanation

`src/` contains the importable package. The package keeps data validation,
estimation, feature construction, target construction, modeling, evaluation, and
artifact export in separate modules. The pipeline layer assembles those modules;
scripts and notebooks call the package rather than duplicating its logic.

The target and model layers share an important invariant: a forward target enters
training only after every return in its horizon has occurred. The model layer
also owns fold-local scaling and the nonnegative variance constraint.

## Part 2: Code reference

- `covariance_denoiser/cli.py`: `covariance-denoiser` command-line interface.
- `covariance_denoiser/data/`: tracked cache validation, price loading, and optional ClickHouse refresh.
- `covariance_denoiser/estimators/`: sample, Ledoit-Wolf, and RMT covariance estimators.
- `covariance_denoiser/features/`: rolling covariance diagnostics and trailing portfolio volatility.
- `covariance_denoiser/targets/`: exact equal-weight portfolio return and forward variance construction.
- `covariance_denoiser/models/`: persistence baseline and purged, scaled ridge evaluation.
- `covariance_denoiser/evaluation/`: deterministic MAE and RMSE calculations.
- `covariance_denoiser/artifacts/`: CSV, Markdown, and PNG writers.
- `covariance_denoiser/pipelines/`: offline workflow orchestration.

Start with `pipelines/offline_demo.py`, then follow its imports into the domain modules.

## Part 3: Short journal

- 2026-07-13: Model training now respects forward-label availability and applies ridge penalties in standardized feature space.
