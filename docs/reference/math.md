# Mathematical Reference

## Symbols

- `P_t`: close price at timestamp `t`.
- `r_t`: log return at timestamp `t`.
- `N`: number of assets.
- `T`: number of observations in one feature window.
- `q = T / N`: observation-to-dimension ratio.
- `lambda_plus = (1 + q^(-1/2))^2`: Marchenko-Pastur upper noise edge.
- `h`: forecast horizon in trading days.

## Log Returns

The pipeline uses log returns in decimal form:

`r_t = log(P_t / P_{t-1})`

This representation is additive through time and is standard in volatility modeling.

## Forward Realized Variance Target

Asset log returns are converted to simple returns before portfolio aggregation.
The exact daily-rebalanced equal-weight portfolio log return is:

`r_p,t = log(1 + (1 / N) * sum_{i=1}^N (exp(r_i,t) - 1))`

Forward realized variance for timestamp `t` and horizon `h` is:

`RV_{t,h} = sum_{j=1}^h r_p,t+j^2`

Annualized target scaling is:

`RV_annualized_{t,h} = RV_{t,h} * (252 / h)`

The target uses returns `t+1` through `t+h` and becomes observable at `t+h`.
Walk-forward training therefore purges the `h-1` target timestamps immediately
before each test origin; chronological row ordering by itself is insufficient.

## Covariance Feature Set

For each in-sample window ending at `t`, the feature builder computes:

- sample average pairwise correlation
- sample covariance condition number
- Ledoit-Wolf covariance condition number
- RMT covariance condition number
- Ledoit-Wolf to sample condition ratio
- RMT to sample condition ratio
- trailing realized volatility baseline

All features use in-sample data only.

## Denoising Conventions

### Ledoit-Wolf

Linear shrinkage is estimated with scikit-learn `LedoitWolf` and returns a labeled covariance matrix.

### Random Matrix Theory

RMT denoising follows this sequence:

1. Convert sample covariance to sample correlation.
2. Eigendecompose sample correlation.
3. Compute `lambda_plus = (1 + q^(-1/2))^2`.
4. Replace eigenvalues `lambda_i <= lambda_plus` by their average.
5. Reconstruct cleaned correlation and rescale by sample volatilities.

## Walk-Forward Modeling

Two models are evaluated:

- naive baseline: target lagged by `h`, updated at every forecast timestamp
- ridge regression: linear model with L2 regularization

Train folds contain only targets observable by the first test timestamp. Features
are standardized using each fold's training statistics before ridge fitting.
The fixed ridge penalty is applied in standardized feature space, and final
variance forecasts are floored at zero.

Adjacent target timestamps share `h-1` future returns. Aggregate error measures
are descriptive unless inference explicitly accounts for this overlap.

## Evaluation Metrics

For realized targets `y_t` and predictions `y_hat_t`:

- `MAE = mean(abs(y_t - y_hat_t))`
- `RMSE = sqrt(mean((y_t - y_hat_t)^2))`

Lower values indicate better forecast quality.
