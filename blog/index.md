---
title: "A Cleaner Covariance Matrix Is Not Yet a Better Forecast"
description: "A leakage-safe walk-forward test of random matrix and Ledoit-Wolf covariance features on eight ETFs, including the math, code, and audited negative result."
date: 2026-07-12
image: "images/cover-covariance-denoising.png"
categories: ["Quantitative Finance", "Risk Modeling"]
---

Covariance cleaning solves a real problem. A noisy covariance matrix can make a
minimum-variance portfolio or a hedge ratio unstable. I wanted to test a harder
claim: do diagnostics from a cleaned matrix help predict the next month of
realized variance?

I ran the experiment on eight exchange-traded funds (ETFs) from 2008 through
2024. A rolling last-observed target benchmark wins. Relative to that benchmark,
scaled ridge regression raises mean absolute error (MAE) by 73.0% and root mean
squared error (RMSE) by 4.2%. Cleaning clearly improves matrix conditioning. These
seven cleaned-matrix features do not improve this forecast.

![A noisy covariance matrix passing through a spectral filter and emerging in a cleaner form.](images/cover-covariance-denoising.png)

The distinction in that last sentence matters. Conditioning asks whether a
matrix reacts wildly to small input changes. Forecasting asks whether today's
matrix contains information about returns that have not happened. One can
improve without the other.

## The object being estimated

Let $P_{i,t}$ denote the adjusted closing price of asset $i$ on trading day $t$.
Its log return, in decimal units per day, is

$$
r_{i,t}=\log\left(\frac{P_{i,t}}{P_{i,t-1}}\right).
$$

Place $T$ daily observations for $N$ assets in the return matrix $R$. Let
$\bar R$ be a matrix whose rows repeat the column means of $R$. The sample
covariance is

$$
S=\frac{1}{T-1}(R-\bar R)^\top(R-\bar R).
$$

Each element $S_{ij}$ has units of daily return squared. The experiment uses
$T=63$ observations and $N=8$ ETFs: EEM, GLD, HYG, IWM, QQQ, SPY, TLT, and VNQ.
The matrix is invertible, but invertibility is a weak standard. Sampling error
can still push the smallest eigenvalues around enough to destabilize an inverse.

For a symmetric positive-definite covariance matrix, the condition number is

$$
\kappa(S)=\frac{\lambda_{\max}(S)}{\lambda_{\min}(S)},
$$

where $\lambda_{\max}(S)$ and $\lambda_{\min}(S)$ are its largest and smallest
eigenvalues. The ratio is dimensionless. A high value warns that an inverse-based
calculation may be sensitive to a small perturbation in the returns.

## Two ways to clean the estimate

Ledoit-Wolf shrinkage blends the sample covariance with a structured target. The
shrinkage intensity is estimated from the data rather than selected by hand. The
method trades a little bias for lower estimation variance.

The random matrix theory (RMT) estimator works on correlation instead. Write
$D=\operatorname{diag}(\sigma_1,\ldots,\sigma_N)$, where $\sigma_i$ is the sample
daily volatility of asset $i$. The sample correlation is

$$
C=D^{-1}SD^{-1}.
$$

Unlike covariance, correlation is unitless. Eigendecompose it as
$C=V\Lambda V^\top$, where $V$ contains eigenvectors and
$\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_N)$ contains eigenvalues.

The Marchenko-Pastur result describes the asymptotic eigenvalue distribution of a
large sample covariance matrix built from independent, identically distributed
noise. Define $q=T/N$. For unit-variance noise, its upper spectral edge is

$$
\lambda_+=\left(1+q^{-1/2}\right)^2.
$$

Here, $q=63/8=7.875$ and $\lambda_+=1.8397$. The implementation classifies every
$\lambda_i\leq\lambda_+$ as noise, replaces those eigenvalues by their mean,
reconstructs a unit-diagonal correlation matrix, and maps it back to covariance:

$$
\widetilde S=D\widetilde C D.
$$

```python
eigenvalues, eigenvectors = np.linalg.eigh(sample_correlation.to_numpy())
aspect_ratio = len(returns_window) / returns_window.shape[1]
lambda_plus = (1.0 + aspect_ratio**-0.5) ** 2

noise_mask = eigenvalues <= lambda_plus
eigenvalues[noise_mask] = eigenvalues[noise_mask].mean()
cleaned = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

This threshold is a filtering rule, not a literal model of ETF returns. Returns
are heteroskedastic, cross-correlated, and serially dependent, while $N=8$ is far
from an asymptotic high-dimensional setting. Renormalizing the reconstructed
matrix also changes the cleaned spectrum slightly. Those caveats make RMT a
useful heuristic here, not ground truth.

On the final 63-day window, the sample covariance condition number is 271.2.
Ledoit-Wolf reduces it to 9.1 and RMT to 91.1.

![Condition numbers for sample, Ledoit-Wolf, and RMT covariance estimates on the final 63-day window.](images/01-condition-numbers.png)

The chart supports a narrow claim: both cleaners improve conditioning on this
window, and Ledoit-Wolf is more aggressive. It does not establish a better
covariance forecast or better portfolio performance.

## From a matrix to seven predictors

For every rolling 63-day window, the pipeline records seven predictors:

- average pairwise sample correlation;
- condition numbers for the sample, Ledoit-Wolf, and RMT covariances;
- each cleaned condition number divided by the sample condition number;
- trailing 21-day annualized volatility of a daily rebalanced equal-weight portfolio.

The portfolio-return construction deserves care. Averaging asset log returns is
only an approximation to an equal-weight portfolio return. The code first
converts each asset log return to a simple return, averages those simple returns,
then converts the portfolio result back to log form. If $N$ is the asset count,

$$
r_{p,t}=\log\left(1+\frac{1}{N}\sum_{i=1}^{N}\left(e^{r_{i,t}}-1\right)\right).
$$

The response variable is annualized forward realized variance over $h=21$
trading days:

$$
RV_{t,h}=\frac{252}{h}\sum_{j=1}^{h}r_{p,t+j}^{2}.
$$

$RV_{t,h}$ is variance, not volatility, so its units are annualized return
squared. The trailing predictor is volatility, in annualized return units. That
difference is deliberate and does not require equal units in a regression, but
it does require feature scaling before applying a common ridge penalty.

## The timing rule that prevents label leakage

A feature stamped $t$ uses returns through $t$. Its target uses returns from
$t+1$ through $t+21$ and becomes observable only at $t+21$. A chronological
train-before-test split is therefore insufficient. At the first test timestamp,
the final 20 target rows immediately before it are still incomplete.

The corrected walk-forward code purges those unavailable labels. Each first fold
contains 252 observed training labels, 20 purged label timestamps, and 21 test
rows. The model advances by 21 rows and retains all earlier labels that have
become observable.

```python
test_start = config.min_train_size + config.target_horizon_days - 1
while test_start + config.test_size <= sample_count:
    train_stop = test_start - config.target_horizon_days + 1
    train_slice = slice(0, train_stop)
    test_slice = slice(test_start, test_start + config.test_size)
    slices.append((train_slice, test_slice))
    test_start += config.step_size
```

This produces 186 folds and 3,906 out-of-sample forecasts from May 2009 through
November 2024. Targets on adjacent test dates overlap for 20 of 21 returns, so
the aggregate errors are descriptive. Ordinary independent-observation standard
errors would be misleading.

## Benchmark and ridge model

The benchmark updates on every test date. At date $t$, it predicts with the
target stamped $t-21$, whose final return has just become observable. It is both
information-matched and strong because volatility persists. Freezing that value
for an entire 21-day block would give the model fresh daily features while
denying the benchmark fresh daily observations.

The competing model is ridge regression. Within each fold, predictor $j$ is
standardized using only the training mean $\mu_j$ and training standard deviation
$s_j$:

$$
z_{k,j}=\frac{x_{k,j}-\mu_j}{s_j},
$$

where $x_{k,j}$ is predictor $j$ for training observation $k$. Ridge estimates
the intercept $\beta_0$ and coefficient vector $\beta$ by minimizing

$$
\sum_{k=1}^{M}\left(y_k-\beta_0-z_k^\top\beta\right)^2
+\alpha\sum_{j=1}^{7}\beta_j^2,
$$

where $M$ is the number of training observations, $y_k$ is realized variance,
$z_k$ is the standardized feature vector, and the fixed penalty is $\alpha=1$.
Scaling matters because condition numbers can be in the hundreds while trailing
volatility is a decimal. Without scaling, the penalty treats their coefficients
unequally for no economic reason.

An unconstrained linear model can predict negative variance. The pipeline applies
a prespecified economic constraint,

$$
\widehat{RV}_{t,h}=\max\left(0,\widehat{RV}^{\mathrm{raw}}_{t,h}\right).
$$

The scaler, regression, and zero floor are identical in every fold. No parameter
was selected using the test results.

## Rolling persistence wins

For $K$ out-of-sample forecasts, let $y_k$ be realized variance and $\hat y_k$
its prediction. The two reported loss functions are

$$
\operatorname{MAE}=\frac{1}{K}\sum_{k=1}^{K}|y_k-\hat y_k|
$$

and

$$
\operatorname{RMSE}=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(y_k-\hat y_k)^2}.
$$

Both are measured in annualized variance units. RMSE penalizes large misses more
heavily because it squares each error.

| Model | MAE | RMSE |
|---|---:|---:|
| Rolling last observable target | 0.01061 | 0.03401 |
| Scaled ridge with zero floor | 0.01835 | 0.03543 |

![Mean absolute error and root mean squared error for the two out-of-sample forecasts.](images/02-forecast-errors.png)

Ridge loses on both measures. The 73.0% MAE gap is large, while the 4.2% RMSE gap
is narrow. Squaring errors compresses the difference because both models miss
abrupt variance shocks. The result still depends on the stated loss function for
economic interpretation, but not for the ranking in this sample.

![Realized annualized variance and both out-of-sample forecasts through time.](images/03-forecast-paths.png)

The time series explains the split verdict. Both forecasts lag abrupt shocks.
Ridge also produces false positives and reaches its zero floor on 784 of 3,906
forecasts, or 20.1% of the sample. Those ordinary-date misses hurt MAE. Neither
model anticipates crisis variance reliably, which keeps the RMSE gap much smaller.

## What this experiment does and does not show

The conditioning result and the baseline victory both survive the audit, but the
original implementation did not. Unavailable labels had to be purged, portfolio
returns reconstructed exactly, predictors scaled, negative variance forecasts
floored, and the benchmark updated at the same daily frequency as the model. The
frequency with which the floor binds also shows that linear ridge is a poor match
for the shape of conditional variance.

The test still cannot isolate the incremental contribution of denoising. Ridge
receives sample diagnostics, cleaned diagnostics, ratios, and trailing volatility
together. An ablation study would compare nested feature sets. The experiment
also holds the universe, 63-day lookback, 21-day horizon, and ridge penalty fixed.
It evaluates no minimum-variance portfolio, turnover cost, hedge ratio, or risk
attribution task, all of which use covariance more directly than seven scalar
summaries do.

My next version would separate three questions:

1. Does cleaning reduce out-of-sample covariance estimation error?
2. Does it improve realized risk and turnover for a constrained portfolio?
3. Do cleaned-matrix features add forecast value beyond trailing volatility?

Those tests need nested temporal tuning and an ablation table. A larger asset
universe would also put the RMT approximation in a setting closer to the problem
it was designed for.

## Reproducibility and references

The tracked cache contains adjusted closes from 2008-01-02 through 2024-12-31.
`blog/data/` freezes the audited metrics, predictions, and coefficients.
`blog/generate_charts.py` regenerates all three evidence charts from those files
and the tracked price cache. The repository test suite includes explicit checks
for target timing, exact equal-weight portfolio returns, and nonnegative variance
forecasts.

Primary references:

- V. A. Marchenko and L. A. Pastur, [“Distribution of Eigenvalues for Some Sets of Random Matrices” (1967)](https://doi.org/10.1070/SM1967v001n04ABEH001994).
- Olivier Ledoit and Michael Wolf, [“A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices” (2004)](https://doi.org/10.1016/S0047-259X(03)00096-4).
- Arthur E. Hoerl and Robert W. Kennard, [“Ridge Regression: Biased Estimation for Nonorthogonal Problems” (1970)](https://doi.org/10.1080/00401706.1970.10488634).

The cover was created for this article with an image-generation model. No market
data or forecast result in the post comes from the generated image.
