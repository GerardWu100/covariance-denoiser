"""Baseline forecasting helpers for walk-forward evaluation."""

from __future__ import annotations

import pandas as pd


def predict_naive_last_value(train_target: pd.Series, test_index: pd.Index) -> pd.Series:
    """Predict each test row with the last observed training target value."""
    last_observed_value = float(train_target.iloc[-1])
    return pd.Series(last_observed_value, index=test_index, name="naive_prediction")
