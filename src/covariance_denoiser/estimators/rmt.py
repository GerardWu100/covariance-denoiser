"""Random-matrix-theory covariance denoising estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from covariance_denoiser.estimators.sample import covariance_to_correlation

MIN_CORRELATION_VARIANCE: float = 1e-12


def estimate_rmt_covariance(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Estimate covariance with RMT eigenvalue cleaning.

    Parameters
    ----------
    returns_window
        In-sample log return matrix with rows as dates and columns as assets.

    Returns
    -------
    pd.DataFrame
        Denoised covariance matrix preserving input asset labels.
    """
    sample_covariance = returns_window.cov()
    sample_correlation = covariance_to_correlation(sample_covariance)

    eigenvalues, eigenvectors = np.linalg.eigh(sample_correlation.to_numpy())

    # Marcenko-Pastur upper edge for the noise bulk (T observations, N assets).
    observation_count = float(len(returns_window))
    asset_count = float(returns_window.shape[1])
    aspect_ratio = observation_count / asset_count
    lambda_plus = (1.0 + aspect_ratio**-0.5) ** 2

    # Replace noise eigenvalues with their average rather than zeroing them out.
    noise_mask = eigenvalues <= lambda_plus
    if noise_mask.any():
        eigenvalues[noise_mask] = float(eigenvalues[noise_mask].mean())

    cleaned_correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    cleaned_correlation = 0.5 * (cleaned_correlation + cleaned_correlation.T)

    # Re-normalize to a valid correlation matrix with unit diagonal.
    diagonal = np.sqrt(np.maximum(np.diag(cleaned_correlation), MIN_CORRELATION_VARIANCE))
    scaling = np.outer(diagonal, diagonal)
    cleaned_correlation = cleaned_correlation / scaling
    cleaned_correlation = 0.5 * (cleaned_correlation + cleaned_correlation.T)
    np.fill_diagonal(cleaned_correlation, 1.0)

    # Map cleaned correlation back to covariance using sample volatilities.
    sample_volatility = returns_window.std(ddof=1).to_numpy()
    volatility_scaling = np.diag(sample_volatility)
    cleaned_covariance = volatility_scaling @ cleaned_correlation @ volatility_scaling

    return pd.DataFrame(
        cleaned_covariance,
        index=returns_window.columns,
        columns=returns_window.columns,
    )
