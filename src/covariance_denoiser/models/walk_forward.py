"""Walk-forward model training and prediction orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import Ridge

from covariance_denoiser.models.baselines import predict_naive_last_value


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward split configuration.

    Parameters
    ----------
    min_train_size
        Minimum number of rows in the expanding training window.
    test_size
        Number of rows in each test fold.
    step_size
        Number of rows to advance after each fold.
    """

    min_train_size: int
    test_size: int
    step_size: int


def generate_walk_forward_slices(
    sample_count: int,
    config: WalkForwardConfig,
) -> list[tuple[slice, slice]]:
    """Generate ordered train/test slices for expanding-window evaluation."""
    slices: list[tuple[slice, slice]] = []

    # First test fold starts only after the minimum training history is available.
    test_start = config.min_train_size
    while test_start + config.test_size <= sample_count:
        train_slice = slice(0, test_start)
        test_slice = slice(test_start, test_start + config.test_size)
        slices.append((train_slice, test_slice))
        test_start += config.step_size

    return slices


def run_walk_forward_models(
    features: pd.DataFrame,
    target: pd.Series,
    config: WalkForwardConfig,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward naive and ridge forecasting.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Prediction table and model coefficient table.
    """
    # Align features and target on shared timestamps before slicing folds.
    aligned_frame = features.join(target.rename("target"), how="inner").dropna()
    feature_columns = features.columns

    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, float | int | str]] = []

    folds = generate_walk_forward_slices(sample_count=len(aligned_frame), config=config)

    for fold_id, (train_slice, test_slice) in enumerate(folds):
        train_frame = aligned_frame.iloc[train_slice]
        test_frame = aligned_frame.iloc[test_slice]

        train_features = train_frame[feature_columns]
        train_target = train_frame["target"]
        test_features = test_frame[feature_columns]
        test_target = test_frame["target"]

        naive_prediction = predict_naive_last_value(
            train_target=train_target,
            test_index=test_frame.index,
        )

        ridge_model = Ridge(alpha=ridge_alpha)
        ridge_model.fit(train_features, train_target)
        ridge_prediction = pd.Series(
            ridge_model.predict(test_features),
            index=test_features.index,
            name="ridge_prediction",
        )

        fold_predictions = pd.DataFrame(
            {
                "timestamp": test_frame.index,
                "y_true": test_target.to_numpy(),
                "naive_prediction": naive_prediction.to_numpy(),
                "ridge_prediction": ridge_prediction.to_numpy(),
                "fold_id": fold_id,
                "train_end_timestamp": train_frame.index[-1],
            }
        )
        prediction_rows.append(fold_predictions)

        for feature_name, coefficient in zip(feature_columns, ridge_model.coef_, strict=True):
            coefficient_rows.append(
                {
                    "fold_id": fold_id,
                    "feature": str(feature_name),
                    "coefficient": float(coefficient),
                }
            )

        coefficient_rows.append(
            {
                "fold_id": fold_id,
                "feature": "__intercept__",
                "coefficient": float(ridge_model.intercept_),
            }
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    return predictions, coefficients
