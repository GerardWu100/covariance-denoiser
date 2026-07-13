# GUIDE_covariance_denoiser

## Part 1: Conceptual explanation

The package turns adjusted daily prices into audited realized-variance forecasts.
Three covariance estimators consume the same rolling asset-return window. Feature
engineering compresses those matrices into scalar diagnostics. Target
construction independently builds exact daily-rebalanced equal-weight portfolio
returns and a future sum of squared returns.

If the target horizon is $h$, the target attached to timestamp $t$ is known at
$t+h$. The expanding-window model therefore keeps $h-1$ rows between its final
training-label timestamp and first test timestamp. Each ridge fit standardizes
the seven features using only that fold's training rows. A fixed zero floor keeps
the resulting variance forecasts inside their economic domain. The persistence
benchmark rolls daily with the target lagged by $h$ rows.

## Part 2: Code reference

- `cli.py` and `__main__.py`: package entrypoints for the offline demo and optional cache refresh.
- `data/prices.py`: validated local-price loading and asset log-return matrix.
- `estimators/sample.py`: sample covariance and covariance-to-correlation scaling.
- `estimators/shrinkage.py`: scikit-learn Ledoit-Wolf estimate.
- `estimators/rmt.py`: Marchenko-Pastur threshold, eigenvalue averaging, correlation renormalization, and covariance rescaling.
- `targets/realized_variance.py`: `compute_equal_weight_portfolio_log_returns` and `compute_forward_realized_variance_target`.
- `features/covariance_features.py`: seven rolling predictors.
- `models/baselines.py`: daily rolling last-observable-target forecast.
- `models/walk_forward.py`: `WalkForwardConfig`, purged fold generation, scaling, ridge fit, and forecast floor.
- `evaluation/metrics.py`: MAE and RMSE tables.
- `artifacts/export.py`: static output bundle.
- `pipelines/offline_demo.py`: end-to-end assembly and horizon propagation.

For forecast timing work, read `targets/realized_variance.py` and
`models/walk_forward.py` together.

## Part 3: Short journal

- 2026-07-13: The audit replaced the approximate mean of asset log returns with the exact log return of a daily-rebalanced equal-weight basket.
