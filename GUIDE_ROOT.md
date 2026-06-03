# GUIDE_ROOT

## Purpose

This repository is an offline-first quantitative finance resume project.

The core workflow forecasts forward realized variance from denoised covariance
features built from local parquet prices in `data/raw/`.

## Where Things Live

- `data/raw/`: tracked raw parquet inputs and metadata contract.
- `src/covariance_denoiser/`: package implementation.
- `scripts/`: thin shell wrappers; logic stays in the package.
- `tests/unit/` and `tests/integration/`: fast module tests and pipeline integration tests.
- `logs/`: dated runtime logs (gitignored).
- `src/covariance_denoiser/data/`: raw-cache validation, local loading, optional ClickHouse refresh.
- `src/covariance_denoiser/targets/`: log-return and realized-variance target construction.
- `src/covariance_denoiser/features/`: denoiser-driven feature engineering.
- `src/covariance_denoiser/models/`: naive baseline and linear walk-forward modeling.
- `src/covariance_denoiser/evaluation/`: deterministic regression metrics.
- `src/covariance_denoiser/artifacts/`: CSV, markdown, and PNG export helpers.
- `src/covariance_denoiser/pipelines/`: end-to-end offline demo pipeline.
- `notebooks/01_offline_research_pipeline.ipynb`: teaching notebook.
- `docs/reference/`: architecture and math references.
- `docs/user/`: practical run guide.
- `tests/`: focused contract tests for data, targets, features, models, pipelines, and notebook.

## Runtime Contract

- Default commands run from local parquet files only.
- Raw inputs must live under `data/raw/`.
- Notebook execution must not require database access.
- ClickHouse refresh is optional and isolated to explicit refresh code paths.
- If `data/raw/` is valid, the project runs in any clone with no database.
