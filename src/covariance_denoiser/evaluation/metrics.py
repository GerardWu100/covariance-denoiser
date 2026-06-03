"""Deterministic regression metrics for model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_PREDICTION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("naive_last_value", "naive_prediction"),
    ("ridge_regression", "ridge_prediction"),
)


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute mean absolute error."""
    absolute_errors = np.abs(y_true.to_numpy() - y_pred.to_numpy())
    return float(absolute_errors.mean())


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute root mean squared error."""
    squared_errors = np.square(y_true.to_numpy() - y_pred.to_numpy())
    return float(np.sqrt(squared_errors.mean()))


def build_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build model-level metrics table from fold prediction rows."""
    y_true = predictions["y_true"]

    metric_rows: list[dict[str, float | str]] = []
    for model_name, prediction_column in MODEL_PREDICTION_COLUMNS:
        y_pred = predictions[prediction_column]
        metric_rows.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(y_true=y_true, y_pred=y_pred),
                "rmse": root_mean_squared_error(y_true=y_true, y_pred=y_pred),
            }
        )

    return pd.DataFrame(metric_rows)
