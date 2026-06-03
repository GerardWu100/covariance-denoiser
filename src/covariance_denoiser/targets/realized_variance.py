"""Forward realized variance target construction."""

from __future__ import annotations

import pandas as pd

DEFAULT_ANNUALIZATION_DAYS: int = 252


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
    equal_weight_returns = returns.mean(axis=1)
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
