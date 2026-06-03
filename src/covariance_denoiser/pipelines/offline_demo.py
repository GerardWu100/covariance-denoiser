"""Offline realized variance research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from covariance_denoiser.artifacts.export import export_offline_demo_artifacts
from covariance_denoiser.data.prices import build_log_return_matrix, load_raw_prices
from covariance_denoiser.evaluation.metrics import build_metrics_table
from covariance_denoiser.features.covariance_features import build_covariance_feature_table
from covariance_denoiser.models.walk_forward import WalkForwardConfig, run_walk_forward_models
from covariance_denoiser.targets.realized_variance import compute_forward_realized_variance_target


@dataclass(frozen=True)
class OfflineDemoConfig:
    """Configuration for end-to-end offline demo execution."""

    data_dir: Path
    output_dir: Path
    lookback_days: int
    horizon_days: int
    min_train_size: int
    test_size: int
    step_size: int
    annualization_days: int
    ridge_alpha: float


def build_modeling_frame(config: OfflineDemoConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Build aligned feature matrix and target vector for modeling."""
    prices = load_raw_prices(data_dir=config.data_dir)
    returns = build_log_return_matrix(prices=prices)

    # Target uses future returns; features use only history ending at the same timestamp.
    target = compute_forward_realized_variance_target(
        returns=returns,
        horizon_days=config.horizon_days,
        annualize=True,
        annualization_days=config.annualization_days,
    )
    features = build_covariance_feature_table(
        returns=returns,
        lookback_days=config.lookback_days,
        annualization_days=config.annualization_days,
    )

    aligned_frame = features.join(target.rename("target"), how="inner").dropna()
    feature_matrix = aligned_frame[features.columns]
    target_vector = aligned_frame["target"]

    return feature_matrix, target_vector


def run_offline_demo_pipeline(config: OfflineDemoConfig) -> None:
    """Run full offline pipeline and export canonical artifacts."""
    features, target = build_modeling_frame(config=config)

    # Expanding-window evaluation: train on all history before each test block.
    split_config = WalkForwardConfig(
        min_train_size=config.min_train_size,
        test_size=config.test_size,
        step_size=config.step_size,
    )

    predictions, coefficients = run_walk_forward_models(
        features=features,
        target=target,
        config=split_config,
        ridge_alpha=config.ridge_alpha,
    )

    metrics = build_metrics_table(predictions=predictions)

    export_offline_demo_artifacts(
        output_dir=config.output_dir,
        metrics=metrics,
        predictions=predictions,
        coefficients=coefficients,
        horizon_days=config.horizon_days,
        lookback_days=config.lookback_days,
    )
