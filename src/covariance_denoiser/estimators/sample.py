"""Sample covariance estimator and related helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_COVARIANCE_VARIANCE: float = 1e-12


def estimate_sample_covariance(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Estimate sample covariance from aligned return observations."""
    return returns_window.cov()


def covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance matrix to correlation matrix safely."""
    covariance_array = covariance.to_numpy()

    # Scale by volatilities; clamp tiny variances so division stays finite.
    diagonal = np.sqrt(np.maximum(np.diag(covariance_array), MIN_COVARIANCE_VARIANCE))
    scaling = np.outer(diagonal, diagonal)

    correlation_array = covariance_array / scaling
    correlation_array = np.nan_to_num(correlation_array, nan=0.0, posinf=0.0, neginf=0.0)
    correlation_array = 0.5 * (correlation_array + correlation_array.T)
    np.fill_diagonal(correlation_array, 1.0)

    return pd.DataFrame(correlation_array, index=covariance.index, columns=covariance.columns)
