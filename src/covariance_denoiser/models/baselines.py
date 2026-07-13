"""Baseline forecasting helpers for walk-forward evaluation."""

from __future__ import annotations

import pandas as pd


def predict_naive_last_value(
    target: pd.Series,
    test_index: pd.Index,
    target_horizon_days: int,
) -> pd.Series:
    """Predict each test row with its latest observable realized-variance target.

    Parameters
    ----------
    target
        Complete target series indexed by forecast timestamp.
    test_index
        Forecast timestamps for the current test fold.
    target_horizon_days
        Number of rows after which a forward target becomes observable.

    Returns
    -------
    pd.Series
        Rolling persistence forecast. At row ``t``, it uses the target stamped
        ``t - target_horizon_days``.
    """
    lagged_observable_target = target.shift(target_horizon_days)
    prediction = lagged_observable_target.loc[test_index]
    if prediction.isna().any():
        raise ValueError("Naive forecast requires an observable lagged target for every test row.")
    return prediction.rename("naive_prediction")
