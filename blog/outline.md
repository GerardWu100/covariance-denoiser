# Adaptive Outline: Cleaning Covariance Is Not the Same as Forecasting Variance

## Archetype

Mixed **risk-model** and **forecast evaluation**. The numerical core is covariance
estimation under noise; the empirical question is whether diagnostics from those
estimators improve an out-of-sample realized-variance forecast.

## Section Blueprint

1. Start with the negative result: the persistence baseline wins.
2. Explain why an eight-asset, 63-day covariance matrix is noisy.
3. Define the sample covariance, the Marchenko-Pastur upper edge, eigenvalue
   cleaning, and the forward realized-variance target.
4. Walk through the leakage-safe expanding-window experiment.
5. Present estimator conditioning, aggregate errors, and the forecast path.
6. Interpret the result: numerical stability and predictive information are
   separate claims.
7. State the limits and the next experiment that would be worth running.

## Planned Equations

- Log return: $r_{i,t}=\log(P_{i,t}/P_{i,t-1})$.
- Sample covariance: $S=(T-1)^{-1}(R-\bar R)^\top(R-\bar R)$.
- Observation-to-dimension ratio: $q=T/N$.
- Marchenko-Pastur upper noise edge: $\lambda_+=(1+q^{-1/2})^2$.
- Equal-weight portfolio return and annualized forward realized variance.
- Mean absolute error and root mean squared error, with every symbol defined.

## Planned Code

- The compact eigenvalue-cleaning block from `estimate_rmt_covariance`.
- The train-before-test slice generator from the walk-forward model.

## Planned Graphs

1. **Condition numbers on the latest window:** compare sample, Ledoit-Wolf, and
   random-matrix-theory covariance estimates; takeaway: both cleaning methods
   make the matrix less ill-conditioned.
2. **Out-of-sample error:** compare mean absolute error and root mean squared
   error; takeaway: the naive last-value forecast wins both metrics.

## Known Gaps and Assumptions

- The tracked cache contains adjusted daily closes for eight exchange-traded
  funds from 2008-01-02 through 2024-12-31.
- This is one fixed feature set and one ridge penalty, not a hyperparameter study.
- Forecasts overlap within each 21-day block, so the article treats aggregate
  errors as descriptive rather than attaching inferential confidence intervals.
- No trading strategy or portfolio utility is evaluated.

## Workspace and Deployment

The canonical workspace is `covariance-denoiser/blog/`. The normal publish target
would be `~/projects/website/content/post/covariance-denoising-realized-variance/`,
but the user explicitly deferred publication. No website files, Hugo build, or
website commit will be made in this task.

## Review Decision

Approved for drafting after checking the code, mathematical reference, raw-cache
metadata, and a fresh offline demo run. The evidence supports a candid negative
result, not a claim that denoising improved forecasts.
