"""Generate the covariance-denoising blog's evidence charts.

Inputs
------
blog/data/metrics.csv
    Model-level mean absolute error and root mean squared error from a fresh
    offline demo run.
blog/data/fold_predictions.csv
    Walk-forward realized values and forecasts, one row per test timestamp.
data/raw/raw_prices.parquet
    Tracked adjusted daily close prices used only to reconstruct the latest
    63-observation covariance-estimation window.

Outputs
-------
blog/images/01-condition-numbers.png
blog/images/02-forecast-errors.png
    Static, high-resolution figures referenced by both language versions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOG_DIR = Path(__file__).resolve().parent
DATA_DIR = BLOG_DIR / "data"
IMAGE_DIR = BLOG_DIR / "images"
RAW_PRICE_PATH = PROJECT_ROOT / "data" / "raw" / "raw_prices.parquet"
LOOKBACK_DAYS = 63
FIGURE_DPI = 180
ANNUALIZATION_DAYS = 252


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into a correlation matrix.

    Parameters
    ----------
    covariance
        Square covariance matrix whose two axes follow the same asset order.

    Returns
    -------
    np.ndarray
        Correlation matrix with unit diagonal.
    """
    volatility = np.sqrt(np.diag(covariance))
    return covariance / np.outer(volatility, volatility)


def rmt_covariance(returns: np.ndarray) -> np.ndarray:
    """Clean a sample covariance matrix using the project's RMT convention.

    Parameters
    ----------
    returns
        Matrix of log returns with observations on rows and assets on columns.

    Returns
    -------
    np.ndarray
        Random-matrix-theory-denoised covariance matrix in the input asset order.
    """
    sample = np.cov(returns, rowvar=False, ddof=1)
    correlation = covariance_to_correlation(sample)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    q = returns.shape[0] / returns.shape[1]
    lambda_plus = (1.0 + q**-0.5) ** 2
    noise = eigenvalues <= lambda_plus
    eigenvalues[noise] = eigenvalues[noise].mean()
    cleaned = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    cleaned = 0.5 * (cleaned + cleaned.T)
    diagonal = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(diagonal, diagonal)
    volatility = np.std(returns, axis=0, ddof=1)
    return np.diag(volatility) @ cleaned @ np.diag(volatility)


def load_latest_return_window() -> np.ndarray:
    """Load the latest complete 63-day log-return window.

    Returns
    -------
    np.ndarray
        Dense return matrix with 63 rows and one column per tracked asset.
    """
    prices = pd.read_parquet(RAW_PRICE_PATH)
    wide = prices.pivot(index="date", columns="asset", values="close").sort_index()
    returns = np.log(wide / wide.shift(1)).dropna()
    return returns.tail(LOOKBACK_DAYS).to_numpy()


def plot_condition_numbers() -> None:
    """Plot latest-window condition numbers for all covariance estimators."""
    returns = load_latest_return_window()
    matrices = {
        "Sample": np.cov(returns, rowvar=False, ddof=1),
        "Ledoit-Wolf": LedoitWolf().fit(returns).covariance_,
        "RMT": rmt_covariance(returns),
    }
    values = [np.linalg.cond(matrix) for matrix in matrices.values()]
    colors = ["#64748b", "#0f766e", "#0891b2"]

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    bars = ax.bar(matrices.keys(), values, color=colors, width=0.62)
    ax.set_title("Cleaning improves covariance conditioning")
    ax.set_ylabel("Condition number (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f", padding=4)
    fig.savefig(IMAGE_DIR / "01-condition-numbers.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_forecast_errors() -> None:
    """Plot out-of-sample model errors from the frozen demo run."""
    metrics = pd.read_csv(DATA_DIR / "metrics.csv").set_index("model")
    labels = ["Last value", "Ridge"]
    positions = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    mae = metrics.loc[["naive_last_value", "ridge_regression"], "mae"].to_numpy()
    rmse = metrics.loc[["naive_last_value", "ridge_regression"], "rmse"].to_numpy()
    bars_mae = ax.bar(positions - width / 2, mae, width, label="MAE", color="#0f766e")
    bars_rmse = ax.bar(positions + width / 2, rmse, width, label="RMSE", color="#d97706")
    ax.set_title("The persistence baseline wins out of sample")
    ax.set_ylabel("Annualized variance error")
    ax.set_xticks(positions, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars_mae, fmt="%.4f", padding=3)
    ax.bar_label(bars_rmse, fmt="%.4f", padding=3)
    fig.savefig(IMAGE_DIR / "02-forecast-errors.png", dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    """Generate every chart referenced by the bilingual article."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    plot_condition_numbers()
    plot_forecast_errors()


if __name__ == "__main__":
    main()
