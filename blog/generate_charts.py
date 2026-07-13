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
blog/images/03-forecast-paths.png
    Static, high-resolution figures referenced by both language versions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from covariance_denoiser.data.prices import build_log_return_matrix, load_raw_prices
from covariance_denoiser.estimators.rmt import estimate_rmt_covariance
from covariance_denoiser.estimators.sample import estimate_sample_covariance
from covariance_denoiser.estimators.shrinkage import estimate_ledoit_wolf_covariance


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOG_DIR = Path(__file__).resolve().parent
DATA_DIR = BLOG_DIR / "data"
IMAGE_DIR = BLOG_DIR / "images"
RAW_PRICE_PATH = PROJECT_ROOT / "data" / "raw" / "raw_prices.parquet"
LOOKBACK_DAYS = 63
FIGURE_DPI = 180
ANNUALIZATION_DAYS = 252


def load_latest_return_window() -> pd.DataFrame:
    """Load the latest complete 63-day log-return window.

    Returns
    -------
    pd.DataFrame
        Return matrix with 63 rows and one column per tracked asset.
    """
    prices = load_raw_prices(data_dir=RAW_PRICE_PATH.parent)
    returns = build_log_return_matrix(prices=prices)
    return returns.tail(LOOKBACK_DAYS)


def plot_condition_numbers() -> None:
    """Plot latest-window condition numbers for all covariance estimators."""
    returns = load_latest_return_window()
    matrices = {
        "Sample": estimate_sample_covariance(returns),
        "Ledoit-Wolf": estimate_ledoit_wolf_covariance(returns),
        "RMT": estimate_rmt_covariance(returns),
    }
    values = [np.linalg.cond(matrix.to_numpy()) for matrix in matrices.values()]
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
    ax.set_title("Rolling persistence wins both forecast metrics")
    ax.set_ylabel("Annualized variance error")
    ax.set_xticks(positions, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars_mae, fmt="%.4f", padding=3)
    ax.bar_label(bars_rmse, fmt="%.4f", padding=3)
    fig.savefig(IMAGE_DIR / "02-forecast-errors.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_forecast_paths() -> None:
    """Plot realized and predicted annualized variance through time."""
    predictions = pd.read_csv(DATA_DIR / "fold_predictions.csv", parse_dates=["timestamp"])
    predictions = predictions.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    ax.plot(
        predictions["timestamp"],
        predictions["y_true"],
        color="#111827",
        linewidth=1.2,
        label="Realized",
    )
    ax.plot(
        predictions["timestamp"],
        predictions["naive_prediction"],
        color="#d97706",
        linewidth=1.0,
        alpha=0.75,
        label="Last observed target",
    )
    ax.plot(
        predictions["timestamp"],
        predictions["ridge_prediction"],
        color="#0f766e",
        linewidth=1.0,
        alpha=0.8,
        label="Scaled ridge with zero floor",
    )
    ax.set_title("Both models lag abrupt variance shocks")
    ax.set_xlabel("Forecast timestamp")
    ax.set_ylabel("Annualized variance")
    ax.set_ylim(bottom=0.0)
    ax.legend(ncols=3)
    ax.grid(alpha=0.2)
    fig.savefig(IMAGE_DIR / "03-forecast-paths.png", dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    """Generate every chart referenced by the bilingual article."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    plot_condition_numbers()
    plot_forecast_errors()
    plot_forecast_paths()


if __name__ == "__main__":
    main()
