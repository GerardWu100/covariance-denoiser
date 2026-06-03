"""Price loading and return-construction helpers.

The default loader reads only local parquet files under `data/raw/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from covariance_denoiser.data.raw_cache import validate_raw_cache


def load_raw_prices(data_dir: Path | None = None) -> pd.DataFrame:
    """Load long-format price panel from validated local raw cache.

    Parameters
    ----------
    data_dir
        Raw cache directory. If omitted, uses project default `data/raw/`.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns `date`, `asset`, `close`.

    Raises
    ------
    FileNotFoundError
        If required files are missing or metadata contract is invalid.
    """
    validation = validate_raw_cache(data_dir=data_dir)
    if not validation.is_valid:
        raise FileNotFoundError(validation.message)

    prices = pd.read_parquet(validation.parquet_path)
    return prices.sort_values(["date", "asset"]).reset_index(drop=True)


def build_log_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Build wide log-return matrix from long-format prices.

    Parameters
    ----------
    prices
        Long-format table with columns `date`, `asset`, `close`.

    Returns
    -------
    pd.DataFrame
        Wide matrix of log returns indexed by date and columned by asset.
    """
    wide_prices = prices.pivot(index="date", columns="asset", values="close").sort_index()

    log_prices = np.log(wide_prices)
    log_returns = log_prices.diff().dropna(how="any")
    log_returns.index = pd.to_datetime(log_returns.index)
    return log_returns
