"""Ledoit-Wolf shrinkage covariance estimator."""

from __future__ import annotations

import pandas as pd
from sklearn.covariance import LedoitWolf


def estimate_ledoit_wolf_covariance(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Estimate covariance matrix using Ledoit-Wolf linear shrinkage."""
    estimator = LedoitWolf()
    estimator.fit(returns_window.to_numpy())

    covariance = pd.DataFrame(
        estimator.covariance_,
        index=returns_window.columns,
        columns=returns_window.columns,
    )
    return covariance
