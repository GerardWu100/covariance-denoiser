"""Denoised covariance feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from covariance_denoiser.estimators.rmt import estimate_rmt_covariance
from covariance_denoiser.estimators.sample import (
    covariance_to_correlation,
    estimate_sample_covariance,
)
from covariance_denoiser.estimators.shrinkage import estimate_ledoit_wolf_covariance
from covariance_denoiser.targets.realized_variance import (
    compute_equal_weight_portfolio_log_returns,
)

DEFAULT_TRAILING_VOL_DAYS: int = 21
DEFAULT_ANNUALIZATION_DAYS: int = 252
MIN_CONDITION_NUMBER_DENOMINATOR: float = 1e-12


def _covariance_condition_number(covariance: pd.DataFrame) -> float:
    """Compute covariance condition number as a scalar feature."""
    return float(np.linalg.cond(covariance.to_numpy()))


def _average_pairwise_correlation(covariance: pd.DataFrame) -> float:
    """Compute average off-diagonal correlation implied by covariance."""
    correlation = covariance_to_correlation(covariance)
    correlation_array = correlation.to_numpy()
    upper_indices = np.triu_indices_from(correlation_array, k=1)
    return float(correlation_array[upper_indices].mean())


def build_covariance_feature_table(
    returns: pd.DataFrame,
    lookback_days: int,
    annualization_days: int = DEFAULT_ANNUALIZATION_DAYS,
) -> pd.DataFrame:
    """Build denoiser-driven features from rolling in-sample windows.

    Parameters
    ----------
    returns
        Wide log-return matrix indexed by date and columned by assets.
    lookback_days
        Number of in-sample observations per feature row.
    annualization_days
        Trading days used for annualized trailing volatility baseline feature.

    Returns
    -------
    pd.DataFrame
        Feature table indexed by window-end timestamps.
    """
    feature_rows: list[dict[str, float | pd.Timestamp]] = []

    for end_index in range(lookback_days - 1, len(returns)):
        window_start = end_index - lookback_days + 1
        in_sample_window = returns.iloc[window_start : end_index + 1]
        window_timestamp = pd.Timestamp(returns.index[end_index])

        # Three estimators share the same in-sample window; only post-processing differs.
        covariances = {
            "sample": estimate_sample_covariance(in_sample_window),
            "ledoit_wolf": estimate_ledoit_wolf_covariance(in_sample_window),
            "rmt": estimate_rmt_covariance(in_sample_window),
        }

        condition_numbers = {
            name: _covariance_condition_number(matrix) for name, matrix in covariances.items()
        }

        sample_condition_number = condition_numbers["sample"]
        # Ratios divide by sample condition number; guard against near-singular matrices.
        safe_sample_condition_number = max(
            sample_condition_number, MIN_CONDITION_NUMBER_DENOMINATOR
        )

        equal_weight_window_returns = compute_equal_weight_portfolio_log_returns(
            returns=in_sample_window
        )
        trailing_window = equal_weight_window_returns.tail(DEFAULT_TRAILING_VOL_DAYS)
        trailing_realized_volatility = float(
            trailing_window.std(ddof=1) * np.sqrt(float(annualization_days))
        )

        feature_rows.append(
            {
                "timestamp": window_timestamp,
                "sample_avg_pairwise_correlation": _average_pairwise_correlation(
                    covariances["sample"]
                ),
                "sample_condition_number": sample_condition_number,
                "ledoit_wolf_condition_number": condition_numbers["ledoit_wolf"],
                "rmt_condition_number": condition_numbers["rmt"],
                "ledoit_wolf_to_sample_condition_ratio": (
                    condition_numbers["ledoit_wolf"] / safe_sample_condition_number
                ),
                "rmt_to_sample_condition_ratio": (
                    condition_numbers["rmt"] / safe_sample_condition_number
                ),
                "trailing_realized_volatility": trailing_realized_volatility,
            }
        )

    feature_table = pd.DataFrame(feature_rows)
    feature_table = feature_table.set_index("timestamp").sort_index()
    return feature_table
