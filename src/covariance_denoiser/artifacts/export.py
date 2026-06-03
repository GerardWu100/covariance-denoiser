"""Export helpers for offline demo artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _write_summary_markdown(
    output_path: Path,
    metrics: pd.DataFrame,
    horizon_days: int,
    lookback_days: int,
) -> None:
    """Write compact markdown summary for interview review."""
    best_rmse_row = metrics.sort_values("rmse", ascending=True).iloc[0]
    best_model = str(best_rmse_row["model"])

    lines = [
        "# Offline Realized Variance Forecast Summary",
        "",
        (
            f"The best root mean squared error model is `{best_model}` "
            f"for horizon `{horizon_days}` days and lookback `{lookback_days}` days."
        ),
        "",
        "## Notes",
        "- Runtime path uses local parquet files under `data/raw/`.",
        "- Denoised covariance features are computed from in-sample windows only.",
        "- Walk-forward splits enforce train-before-test ordering.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _save_prediction_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save predicted versus realized time-series plot."""
    figure, axis = plt.subplots(figsize=(12, 6), dpi=180, constrained_layout=True)

    ordered_predictions = predictions.sort_values("timestamp")
    axis.plot(
        ordered_predictions["timestamp"],
        ordered_predictions["y_true"],
        label="realized",
        lw=2,
    )
    axis.plot(
        ordered_predictions["timestamp"],
        ordered_predictions["naive_prediction"],
        label="naive",
        alpha=0.8,
    )
    axis.plot(
        ordered_predictions["timestamp"],
        ordered_predictions["ridge_prediction"],
        label="ridge",
        alpha=0.8,
    )

    axis.set_title("Forward Realized Variance: Realized vs Predictions")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Variance")
    axis.legend()

    figure.savefig(output_path)
    plt.close(figure)


def _save_metric_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    """Save bar chart for MAE and RMSE across models."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180, constrained_layout=True)

    axes[0].bar(metrics["model"], metrics["mae"], color="#4e79a7")
    axes[0].set_title("Mean Absolute Error")
    axes[0].set_ylabel("MAE")

    axes[1].bar(metrics["model"], metrics["rmse"], color="#f28e2b")
    axes[1].set_title("Root Mean Squared Error")
    axes[1].set_ylabel("RMSE")

    figure.savefig(output_path)
    plt.close(figure)


def export_offline_demo_artifacts(
    output_dir: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    horizon_days: int,
    lookback_days: int,
) -> None:
    """Export canonical offline demo artifact set.

    This function writes all required output files:
    metrics.csv, fold_predictions.csv, model_coefficients.csv, summary.md,
    and static PNG plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "fold_predictions.csv", index=False)
    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False)

    _write_summary_markdown(
        output_path=output_dir / "summary.md",
        metrics=metrics,
        horizon_days=horizon_days,
        lookback_days=lookback_days,
    )

    _save_prediction_plot(
        predictions=predictions,
        output_path=output_dir / "prediction_vs_realized.png",
    )
    _save_metric_plot(
        metrics=metrics,
        output_path=output_dir / "model_error_metrics.png",
    )
