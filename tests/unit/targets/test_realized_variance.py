"""Tests for log-return and realized-variance target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from covariance_denoiser.data.prices import build_log_return_matrix
from covariance_denoiser.targets.realized_variance import compute_forward_realized_variance_target
from covariance_denoiser.targets.realized_variance import (
    compute_equal_weight_portfolio_log_returns,
)


def test_build_log_return_matrix_matches_log_price_ratio() -> None:
    """Log-return helper should match r_t = log(P_t / P_{t-1})."""
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset": ["SPY", "SPY", "SPY"],
            "close": [100.0, 102.0, 101.0],
        }
    )

    returns = build_log_return_matrix(prices=prices)
    expected = np.log(np.array([102.0 / 100.0, 101.0 / 102.0]))

    assert np.allclose(returns["SPY"].to_numpy(), expected)


def test_forward_realized_variance_uses_future_returns_only() -> None:
    """Target at timestamp t should use returns from t+1 onward, not current return."""
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame({"SPY": [0.01, 0.02, 0.03, 0.04]}, index=index)

    target = compute_forward_realized_variance_target(
        returns=returns,
        horizon_days=2,
        annualize=False,
    )

    expected_t0 = 0.02**2 + 0.03**2
    expected_t1 = 0.03**2 + 0.04**2

    assert np.isclose(target.iloc[0], expected_t0)
    assert np.isclose(target.iloc[1], expected_t1)
    assert np.isnan(target.iloc[2])
    assert np.isnan(target.iloc[3])


def test_equal_weight_portfolio_return_averages_simple_returns_first() -> None:
    """Portfolio log return should represent an exactly rebalanced equal-weight basket."""
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    asset_simple_returns = pd.DataFrame(
        {"A": [0.10, -0.20], "B": [-0.05, 0.30]},
        index=index,
    )
    asset_log_returns = np.log1p(asset_simple_returns)

    portfolio_log_returns = compute_equal_weight_portfolio_log_returns(asset_log_returns)
    expected = np.log1p(asset_simple_returns.mean(axis=1))

    assert np.allclose(portfolio_log_returns.to_numpy(), expected.to_numpy())
