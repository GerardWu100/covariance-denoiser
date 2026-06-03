# GUIDE_src

## Purpose

`src/` contains the implementation package for the offline research pipeline.

## Structure

- `covariance_denoiser/__init__.py`: package version marker.
- `covariance_denoiser/cli.py`: command-line entry points (`covariance-denoiser` console script).
- `covariance_denoiser/__main__.py`: `python -m covariance_denoiser` entry point.
- `covariance_denoiser/data/`: local raw-cache contract and optional refresh path.
- `covariance_denoiser/estimators/`: sample, Ledoit-Wolf, and RMT covariance estimators.
- `covariance_denoiser/targets/`: target construction for forward realized variance.
- `covariance_denoiser/features/`: denoiser-driven feature engineering.
- `covariance_denoiser/models/`: walk-forward model training utilities.
- `covariance_denoiser/evaluation/`: error metrics.
- `covariance_denoiser/artifacts/`: export helpers for static outputs.
- `covariance_denoiser/pipelines/`: end-to-end offline pipeline orchestration.

## Design Rules

- Keep runtime defaults fully offline.
- Isolate optional ClickHouse logic from default execution paths.
- Use typed function signatures.
- Keep modules narrow and interview-explainable.
