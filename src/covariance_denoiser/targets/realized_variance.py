"""Forward realized variance target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ANNUALIZATION_DAYS: int = 252


def compute_equal_weight_portfolio_log_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute exact daily-rebalanced equal-weight portfolio log returns.

    Parameters
    ----------
    returns
        Asset log returns in decimal form, with dates on rows and assets on columns.

    Returns
    -------
    pd.Series
        Portfolio log return at each timestamp. Asset simple returns are averaged
        before conversion back to log returns.
    """
    asset_simple_returns = np.expm1(returns)
    portfolio_simple_returns = asset_simple_returns.mean(axis=1)
    return np.log1p(portfolio_simple_returns).rename("equal_weight_portfolio_log_return")


def compute_forward_realized_variance_target(
    returns: pd.DataFrame,
    horizon_days: int,
    annualize: bool,
    annualization_days: int = DEFAULT_ANNUALIZATION_DAYS,
) -> pd.Series:
    """Build forward realized variance target from future equal-weight returns.

    Parameters
    ----------
    returns
        Wide log-return matrix indexed by date and columned by assets.
    horizon_days
        Forecast horizon in trading days.
    annualize
        If True, scale realized variance by `annualization_days / horizon_days`.
    annualization_days
        Trading days per year used for annualized target scaling.

    Returns
    -------
    pd.Series
        Forward realized variance target indexed by current timestamp.
    """
    equal_weight_returns = compute_equal_weight_portfolio_log_returns(returns=returns)
    squared_equal_weight_returns = equal_weight_returns.pow(2)

    # Rolling sum over the next `horizon_days` squared returns; shift aligns label to today.
    forward_realized_variance = (
        squared_equal_weight_returns.rolling(window=horizon_days).sum().shift(-horizon_days)
    )

    if annualize:
        scaling_factor = float(annualization_days) / float(horizon_days)
        forward_realized_variance = forward_realized_variance * scaling_factor

    target_name = (
        "forward_realized_variance_annualized" if annualize else "forward_realized_variance"
    )
    return forward_realized_variance.rename(target_name)
