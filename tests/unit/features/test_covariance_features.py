"""Tests for denoised covariance feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from covariance_denoiser.estimators.rmt import estimate_rmt_covariance
from covariance_denoiser.estimators.sample import estimate_sample_covariance
from covariance_denoiser.estimators.shrinkage import estimate_ledoit_wolf_covariance
from covariance_denoiser.features.covariance_features import build_covariance_feature_table
from covariance_denoiser.targets.realized_variance import compute_forward_realized_variance_target


EXPECTED_FEATURE_COLUMNS: set[str] = {
    "sample_avg_pairwise_correlation",
    "sample_condition_number",
    "ledoit_wolf_condition_number",
    "rmt_condition_number",
    "ledoit_wolf_to_sample_condition_ratio",
    "rmt_to_sample_condition_ratio",
    "trailing_realized_volatility",
}


def test_covariance_features_use_in_sample_windows_only() -> None:
    """First feature row should align with lookback window end timestamp."""
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(80, 4)),
        index=pd.date_range("2020-01-01", periods=80, freq="B"),
        columns=["SPY", "TLT", "GLD", "QQQ"],
    )

    features = build_covariance_feature_table(returns=returns, lookback_days=20)

    assert features.index.min() == returns.index[19]
    assert set(features.columns) == EXPECTED_FEATURE_COLUMNS
    assert len(features.columns) <= 8


def test_denoiser_outputs_are_symmetric() -> None:
    """Sample, Ledoit-Wolf, and RMT covariance outputs should be symmetric."""
    rng = np.random.default_rng(13)
    window = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(63, 5)),
        columns=["A", "B", "C", "D", "E"],
    )

    sample_covariance = estimate_sample_covariance(window)
    lw_covariance = estimate_ledoit_wolf_covariance(window)
    rmt_covariance = estimate_rmt_covariance(window)

    assert np.allclose(sample_covariance.to_numpy(), sample_covariance.to_numpy().T)
    assert np.allclose(lw_covariance.to_numpy(), lw_covariance.to_numpy().T)
    assert np.allclose(rmt_covariance.to_numpy(), rmt_covariance.to_numpy().T)


def test_feature_rows_align_with_target_timestamps() -> None:
    """Feature and target indexes should overlap on valid modeling timestamps."""
    rng = np.random.default_rng(23)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(90, 3)),
        index=pd.date_range("2021-01-01", periods=90, freq="B"),
        columns=["SPY", "TLT", "GLD"],
    )

    features = build_covariance_feature_table(returns=returns, lookback_days=21)
    target = compute_forward_realized_variance_target(
        returns=returns, horizon_days=5, annualize=False
    )

    overlap = features.index.intersection(target.dropna().index)
    assert len(overlap) > 0
