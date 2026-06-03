# GUIDE_covariance_denoiser

## Purpose

`covariance_denoiser` is the runtime package for the offline resume project.

It turns local parquet prices under `data/raw/` into denoiser-driven features,
forward realized variance forecasts, and compact static artifacts.

## Subpackages

- `data/`: raw-cache paths, metadata validation, local loading, optional ClickHouse refresh.
- `estimators/`: sample covariance, Ledoit-Wolf shrinkage, and RMT denoising.
- `targets/`: forward realized variance target construction.
- `features/`: compact covariance-based feature engineering.
- `models/`: naive baseline and ridge walk-forward training helpers.
- `evaluation/`: MAE and RMSE metrics.
- `artifacts/`: CSV, markdown, and plot export.
- `pipelines/`: end-to-end offline demo pipeline assembly.

## Entry Point

- `cli.py`: command-line interface exposing:
  - `run-offline-demo`
  - `refresh-raw-cache` (optional, ClickHouse-only refresh path)

## Contract

- Default runtime is offline and never requires ClickHouse.
- Raw files are expected only in `data/raw/`.
- Feature windows are in-sample only.
- Forward targets use future returns only.
