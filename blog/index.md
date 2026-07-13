---
title: "Cleaning Covariance Is Not the Same as Forecasting Variance"
description: "A walk-forward test of random matrix theory and Ledoit-Wolf covariance features on eight ETFs, with a negative result worth understanding."
date: 2026-07-12
image: "images/cover-covariance-denoising.png"
categories: ["Quantitative Finance", "Risk Modeling"]
---

A cleaner covariance matrix ought to make a risk model more stable. It does not
follow that features extracted from that matrix will forecast variance better.
I tested that distinction on eight exchange-traded funds (ETFs) from 2008 through
2024. The denoised features fed a ridge regression, and the forecasts were judged
strictly out of sample. A naive last-value forecast won.

That result is more useful than a narrow model victory would have been. Random
matrix theory (RMT) did what it was designed to do: it reduced an ill-conditioned
covariance estimate. The failure happened one step later, when numerical
stability was asked to become predictive information.

![A noisy covariance matrix passing through a spectral filter and emerging in a cleaner form.](images/cover-covariance-denoising.png)

The image captures the experiment's two separate stages: filtering the noisy
matrix on the left, then asking whether the ordered output on the right contains
information about the future.

## Why covariance needs cleaning

Let $P_{i,t}$ be the adjusted closing price of asset $i$ on trading day $t$. Its
log return is

$$
r_{i,t}=\log\left(\frac{P_{i,t}}{P_{i,t-1}}\right).
$$

Put $T$ observations for $N$ assets into a return matrix $R$. If $\bar R$ is the
matrix whose rows contain the column means of $R$, the sample covariance matrix is

$$
S=\frac{1}{T-1}(R-\bar R)^\top(R-\bar R).
$$

Here, $T=63$ trading days and $N=8$ ETFs: EEM, GLD, HYG, IWM, QQQ, SPY, TLT,
and VNQ. Sixty-three observations are enough to invert an eight-dimensional
matrix, but that is a low bar. Sampling error can still produce unstable small
eigenvalues. The condition number, defined as the largest singular value divided
by the smallest, measures this sensitivity. A large condition number means a
small change in the data can cause a large change in calculations that depend on
the matrix.

I compared the sample estimate with two cleaners. Ledoit-Wolf shrinkage pulls the
sample covariance toward a structured target. RMT cleaning works in correlation
space. Define the observation-to-dimension ratio as $q=T/N$. Under the
Marchenko-Pastur model, the upper edge of the noise eigenvalue bulk is

$$
\lambda_+=\left(1+q^{-1/2}\right)^2.
$$

The implementation replaces every correlation eigenvalue $\lambda_i$ satisfying
$\lambda_i\leq\lambda_+$ with the average of those noise eigenvalues. It then
reconstructs a unit-diagonal correlation matrix and scales it by the sample
volatilities.

```python
eigenvalues, eigenvectors = np.linalg.eigh(sample_correlation.to_numpy())
q = len(returns_window) / returns_window.shape[1]
lambda_plus = (1.0 + q**-0.5) ** 2

noise_mask = eigenvalues <= lambda_plus
eigenvalues[noise_mask] = eigenvalues[noise_mask].mean()
cleaned = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

On the final 63-day window, the sample covariance condition number was 271.2.
Ledoit-Wolf reduced it to 9.1; RMT reduced it to 91.1. The cleaners are not
equivalent, but both made the estimate less sensitive.

![Condition numbers for sample, Ledoit-Wolf, and RMT covariance estimates on the final 63-day window.](images/01-condition-numbers.png)

Ledoit-Wolf produces the largest reduction on this window. RMT is less aggressive,
but its condition number is still roughly one-third of the sample estimate.

## Turning matrices into a forecast

The model does not forecast a covariance matrix directly. For each rolling
63-day window, it records seven features:

- average pairwise sample correlation;
- condition numbers for sample, Ledoit-Wolf, and RMT covariance;
- each cleaned condition number divided by the sample condition number;
- trailing 21-day annualized realized volatility of an equal-weight portfolio.

Let $r_{\mathrm{eq},t}=N^{-1}\sum_{i=1}^{N}r_{i,t}$ be the equal-weight portfolio
return. For a forecast horizon of $h=21$ trading days, the annualized forward
realized variance target is

$$
RV_{t,h}=\frac{252}{h}\sum_{j=1}^{h}r_{\mathrm{eq},t+j}^{2}.
$$

The date alignment matters. Every feature at time $t$ uses returns up to and
including $t$. The target uses returns from $t+1$ through $t+h$. No future return
enters a feature.

The evaluation uses an expanding training window. The first fold has 252 rows of
training data and 21 rows of test data. Each subsequent fold advances by 21 rows,
while keeping all earlier training observations. There are 187 folds and 3,927
out-of-sample predictions from April 2009 through November 2024.

```python
test_start = config.min_train_size
while test_start + config.test_size <= sample_count:
    train_slice = slice(0, test_start)
    test_slice = slice(test_start, test_start + config.test_size)
    slices.append((train_slice, test_slice))
    test_start += config.step_size
```

The benchmark predicts every observation in a test fold with the last target
observed in training. The competing model is ridge regression, a linear
regression with an L2 penalty that discourages large coefficients. Its penalty
parameter is fixed at 1.0.

## The baseline wins

Let $y_k$ be realized variance and $\hat y_k$ its forecast for out-of-sample
observation $k$, with $K$ total predictions. Mean absolute error (MAE) is

$$
\operatorname{MAE}=\frac{1}{K}\sum_{k=1}^{K}|y_k-\hat y_k|,
$$

and root mean squared error (RMSE) is

$$
\operatorname{RMSE}=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(y_k-\hat y_k)^2}.
$$

Both are measured in annualized variance units, and lower is better.

| Model | MAE | RMSE |
|---|---:|---:|
| Last observed value | 0.00797 | 0.03134 |
| Ridge regression | 0.01908 | 0.03613 |

Ridge's MAE was 139.5% higher than the baseline's, and its RMSE was 15.3% higher.
The gap between those two comparisons is revealing. Squaring the errors makes
RMSE especially sensitive to large misses. Both models suffer during variance
bursts, so ridge looks less bad under RMSE than it does under MAE. It still loses
on both.

![Mean absolute error and root mean squared error for the two out-of-sample forecasts.](images/02-forecast-errors.png)

The last-value forecast is crude, yet volatility persistence gives it a
hard-to-beat advantage. Ridge produces a smoother conditional estimate, while
the 21-day target can change abruptly as shocks enter and leave its forward
window.

## What failed, and what did not

The RMT estimator did not fail its own test. It changed the eigenvalue spectrum
and lowered the covariance condition number. That is evidence of better
conditioning, not evidence of better forecasts.

The forecasting design asks seven contemporaneous summaries to predict an
equal-weight portfolio's next 21-day variance. Several reasons may explain the
weak result:

- condition numbers describe numerical geometry, not necessarily the direction
  of future volatility;
- a 21-day target is dominated by shocks that trailing covariance cannot foresee;
- ridge is linear and uses one fixed penalty across every fold;
- the trailing realized-volatility feature may carry most of the useful
  persistence already available to the model;
- eight broad ETFs leave little room for high-dimensional noise cleaning to show
  its strongest advantage.

This test also does not answer whether denoising improves minimum-variance
portfolios, hedge ratios, or risk attribution. Those applications consume the
covariance matrix itself. Here the matrix is compressed into scalar diagnostics,
then passed to a forecasting model. That extra transformation changes the
question.

## The next experiment I would run

I would separate the claims. First, evaluate covariance estimators on their own
terms: out-of-sample portfolio variance, weight turnover, and sensitivity to the
lookback window. Second, test variance forecasting with a baseline ladder: last
value, a heterogeneous autoregressive volatility model, then regularized models
with nested time-series tuning. A larger asset universe would also make the
RMT setting more meaningful because the ratio $N/T$ would be less forgiving.

The present result is still a sound stopping point. A risk estimate can be
numerically cleaner and economically useful without becoming a superior
forecasting signal. Treating those as separate hypotheses prevents an attractive
matrix from receiving credit it has not earned.

## Reproducibility

The run uses the repository's tracked adjusted-close cache, dated 2008-01-02 to
2024-12-31. The article's frozen metrics and predictions live under `blog/data/`,
and `blog/generate_charts.py` regenerates both evidence charts. The cover was
created specifically for this article with an image-generation model. The study
is a fixed demonstration, not a search over universes, horizons, penalties, or
feature sets.
