"""Tests for walk-forward model training and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariance_denoiser.evaluation.metrics import build_metrics_table
from covariance_denoiser.models.baselines import predict_naive_last_value
from covariance_denoiser.models.walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_slices,
    run_walk_forward_models,
)


def test_walk_forward_slices_respect_train_before_test() -> None:
    """Every fold should purge labels unavailable at the forecast origin."""
    config = WalkForwardConfig(
        min_train_size=30,
        test_size=10,
        step_size=10,
        target_horizon_days=5,
    )
    folds = generate_walk_forward_slices(sample_count=80, config=config)

    for train_slice, test_slice in folds:
        assert train_slice.stop is not None
        assert test_slice.start is not None
        assert train_slice.stop == test_slice.start - config.target_horizon_days + 1

    first_train_slice, first_test_slice = folds[0]
    assert first_train_slice.stop == config.min_train_size
    assert first_test_slice.start == config.min_train_size + config.target_horizon_days - 1


def test_walk_forward_config_rejects_nonpositive_horizon() -> None:
    """A target horizon must include at least one future observation."""
    with pytest.raises(ValueError, match="target_horizon_days"):
        WalkForwardConfig(
            min_train_size=30,
            test_size=10,
            step_size=10,
            target_horizon_days=0,
        )


def test_walk_forward_predictions_include_naive_and_ridge_models() -> None:
    """Pipeline should return naive and ridge predictions for each fold row."""
    index = pd.date_range("2020-01-01", periods=90, freq="B")

    x1 = np.linspace(0.0, 1.0, num=90)
    x2 = np.cos(np.linspace(0.0, 6.0, num=90))
    features = pd.DataFrame({"x1": x1, "x2": x2}, index=index)
    target = pd.Series(0.5 * x1 - 0.2 * x2 + 0.01, index=index)

    config = WalkForwardConfig(
        min_train_size=40,
        test_size=10,
        step_size=10,
        target_horizon_days=5,
    )
    predictions, coefficients = run_walk_forward_models(
        features=features,
        target=target,
        config=config,
        ridge_alpha=1.0,
    )

    assert "naive_prediction" in predictions.columns
    assert "ridge_raw_prediction" in predictions.columns
    assert "ridge_prediction" in predictions.columns
    assert len(predictions) > 0
    assert len(coefficients) > 0


def test_naive_forecast_updates_with_each_newly_observable_target() -> None:
    """Persistence benchmark should roll daily instead of freezing for a test block."""
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    target = pd.Series(np.arange(8, dtype=float), index=index)

    prediction = predict_naive_last_value(
        target=target,
        test_index=index[3:6],
        target_horizon_days=3,
    )

    assert prediction.to_list() == [0.0, 1.0, 2.0]


def test_ridge_variance_forecasts_are_nonnegative() -> None:
    """Ridge post-processing should respect the nonnegative variance domain."""
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    features = pd.DataFrame(
        {
            "small_scale": np.linspace(0.0, 0.01, num=80),
            "large_scale": np.linspace(0.0, 10_000.0, num=80),
        },
        index=index,
    )
    target = pd.Series(np.linspace(0.02, 0.001, num=80), index=index)
    config = WalkForwardConfig(
        min_train_size=30,
        test_size=10,
        step_size=10,
        target_horizon_days=5,
    )

    predictions, _ = run_walk_forward_models(
        features=features,
        target=target,
        config=config,
        ridge_alpha=1.0,
    )

    assert (predictions["ridge_prediction"] >= 0.0).all()


def test_metric_calculations_are_deterministic_on_toy_data() -> None:
    """Metric helper should return stable values for fixed prediction input."""
    predictions = pd.DataFrame(
        {
            "y_true": [0.10, 0.20, 0.15, 0.25],
            "naive_prediction": [0.12, 0.19, 0.13, 0.22],
            "ridge_prediction": [0.11, 0.21, 0.14, 0.24],
        }
    )

    metrics_first = build_metrics_table(predictions=predictions)
    metrics_second = build_metrics_table(predictions=predictions)

    assert metrics_first.equals(metrics_second)
