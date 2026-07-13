"""Walk-forward model training and prediction orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from covariance_denoiser.models.baselines import predict_naive_last_value

MIN_VARIANCE_FORECAST: float = 0.0


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
    target_horizon_days
        Number of future return observations used by each target. A target
        stamped at row ``t`` becomes observable at row ``t + target_horizon_days``.
    """

    min_train_size: int
    test_size: int
    step_size: int
    target_horizon_days: int

    def __post_init__(self) -> None:
        """Validate that every split parameter is strictly positive.

        Raises
        ------
        ValueError
            If any configured row count or horizon is less than one.
        """
        values = {
            "min_train_size": self.min_train_size,
            "test_size": self.test_size,
            "step_size": self.step_size,
            "target_horizon_days": self.target_horizon_days,
        }
        for name, value in values.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1; received {value}.")


def generate_walk_forward_slices(
    sample_count: int,
    config: WalkForwardConfig,
) -> list[tuple[slice, slice]]:
    """Generate label-availability-safe expanding-window slices.

    Parameters
    ----------
    sample_count
        Number of aligned feature and target rows.
    config
        Split sizes and forward-target horizon.

    Returns
    -------
    list[tuple[slice, slice]]
        Training and test slices. Training excludes labels whose future-return
        windows are incomplete at the first test timestamp.
    """
    slices: list[tuple[slice, slice]] = []

    # A target at position t uses returns t+1 through t+h and is known at t+h.
    # Starting at min_train_size + h - 1 preserves the requested count of fully
    # observed training labels while purging the h - 1 unavailable labels.
    test_start = config.min_train_size + config.target_horizon_days - 1
    while test_start + config.test_size <= sample_count:
        train_stop = test_start - config.target_horizon_days + 1
        train_slice = slice(0, train_stop)
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
    """Run label-availability-safe naive and ridge forecasts.

    Parameters
    ----------
    features
        Feature rows indexed by forecast timestamp.
    target
        Forward target indexed by forecast timestamp.
    config
        Expanding-window sizes and target horizon.
    ridge_alpha
        L2 penalty applied after standardizing each feature on the training fold.

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
            target=aligned_frame["target"],
            test_index=test_frame.index,
            target_horizon_days=config.target_horizon_days,
        )

        # Ridge is not scale invariant. Fit the scaler on each training fold so
        # condition numbers and decimal volatility receive comparable penalties.
        ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        ridge_model.fit(train_features, train_target)
        # A linear model is unconstrained, but variance cannot be negative. The
        # zero floor is fixed before evaluation and applies to every fold.
        ridge_raw_prediction = pd.Series(
            ridge_model.predict(test_features),
            index=test_features.index,
            name="ridge_raw_prediction",
        )
        ridge_prediction = pd.Series(
            ridge_raw_prediction.clip(lower=MIN_VARIANCE_FORECAST),
            index=test_features.index,
            name="ridge_prediction",
        )

        fold_predictions = pd.DataFrame(
            {
                "timestamp": test_frame.index,
                "y_true": test_target.to_numpy(),
                "naive_prediction": naive_prediction.to_numpy(),
                "ridge_raw_prediction": ridge_raw_prediction.to_numpy(),
                "ridge_prediction": ridge_prediction.to_numpy(),
                "fold_id": fold_id,
                "train_end_timestamp": train_frame.index[-1],
            }
        )
        prediction_rows.append(fold_predictions)

        fitted_ridge = ridge_model.named_steps["ridge"]
        for feature_name, coefficient in zip(feature_columns, fitted_ridge.coef_, strict=True):
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
                "coefficient": float(fitted_ridge.intercept_),
            }
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    return predictions, coefficients
