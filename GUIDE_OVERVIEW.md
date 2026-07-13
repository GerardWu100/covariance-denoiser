# Project overview

```text
data/raw
   |
   v
prices and returns ---> covariance estimators ---> seven rolling features
   |                                                |
   +--> equal-weight portfolio ---> forward target -+
                                                    v
                                        purged walk-forward evaluation
                                                    |
                                                    v
                                      CSV, Markdown, and PNG artifacts
```

## Purpose

The project separates two claims that are easy to conflate. Covariance cleaning
may improve numerical conditioning, while features derived from the cleaned
matrix may or may not improve a future variance forecast. The research pipeline
tests the second claim without losing sight of the first.

## Estimation and forecast flow

Each 63-day return window produces sample, Ledoit-Wolf, and random-matrix-theory
covariance estimates. Their condition numbers, ratios, average sample
correlation, and trailing portfolio volatility form seven predictors.

The response is the annualized sum of squared equal-weight portfolio log returns
over the next 21 trading days. Portfolio returns are exact for daily rebalancing:
the pipeline averages asset simple returns before converting the portfolio result
back to log form.

At any forecast origin, recent forward targets are still unobserved. The
walk-forward design purges them, trains on an expanding history of fully observed
labels, standardizes features within that history, and evaluates ridge regression
against a last-observed target benchmark that updates daily. Adjacent target
windows overlap, so the exported MAE and RMSE are descriptive comparisons.

## Main assumptions and limits

- Eight assets and 63 observations are a small setting for asymptotic random matrix theory.
- The Marchenko-Pastur threshold is a filtering heuristic because ETF returns are not independent, identically distributed noise.
- The experiment uses one universe, horizon, lookback, feature set, and ridge penalty.
- Scalar covariance diagnostics do not test portfolio optimization, turnover, or hedge performance directly.
- ClickHouse can refresh the raw cache but is outside the default reproducible path.
