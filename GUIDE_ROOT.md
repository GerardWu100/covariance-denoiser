# GUIDE_ROOT

## Part 1: Conceptual explanation

This repository tests whether covariance-denoising diagnostics help forecast the
next 21 trading days of realized variance for a daily-rebalanced equal-weight ETF
portfolio. The default workflow is offline: it reads the tracked parquet cache,
builds log returns, creates rolling covariance features, and evaluates a
persistence benchmark against ridge regression.

The timing contract is stricter than ordinary train-before-test ordering. A
target stamped at date $t$ contains returns from $t+1$ through $t+h$, where $h$
is the forecast horizon, and becomes observable only at $t+h$. Each walk-forward
fold removes the $h-1$ unavailable target rows before its test origin. The
persistence benchmark updates daily with the newly observable lagged target.
Ridge features are standardized from training data only, and variance forecasts
are floored at zero.

The data flow is:

```text
tracked prices -> asset log returns -> rolling covariance features
               -> exact equal-weight portfolio returns -> forward variance target
               -> purged expanding folds -> benchmark and scaled ridge -> artifacts
```

ClickHouse is an explicit cache-refresh option. It is never required by the
normal pipeline, notebook, or tests.

## Part 2: Code reference

- `README.md`: project scope, commands, outputs, and evaluation guardrails.
- `pyproject.toml`: package metadata, dependencies, command-line entrypoint, and tool settings.
- `data/raw/`: tracked adjusted-price cache and metadata contract.
- `src/covariance_denoiser/`: package implementation; start with its package guide.
- `scripts/`: thin wrappers for the demo and test suite.
- `tests/`: unit and integration contracts, including timing and portfolio-return checks.
- `notebooks/01_offline_research_pipeline.ipynb`: top-to-bottom teaching walkthrough.
- `docs/reference/`: mathematical and system contracts.
- `docs/user/`: practical run instructions.
- `blog/`: bilingual article, frozen audited evidence, chart generator, and images.
- `outputs/demo/`: gitignored artifacts from a local run.

## Part 3: Short journal

- 2026-07-13: The forecast audit added a target-availability purge, exact portfolio aggregation, fold-local feature scaling, and a zero floor for variance predictions.
