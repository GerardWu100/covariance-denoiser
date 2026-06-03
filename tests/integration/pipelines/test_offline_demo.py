"""Contract tests for offline demo pipeline and docs."""

from __future__ import annotations

from pathlib import Path

from covariance_denoiser.cli import build_parser
from covariance_denoiser.pipelines.offline_demo import OfflineDemoConfig, run_offline_demo_pipeline


def test_readme_describes_offline_research_pipeline() -> None:
    """README should reflect the offline realized variance project scope."""
    readme_text = Path("README.md").read_text(encoding="utf-8").lower()

    assert "data/raw/" in readme_text
    assert "realized variance" in readme_text
    assert "feature" in readme_text
    assert "model training" in readme_text or "walk-forward" in readme_text


def test_package_cli_module_exists() -> None:
    """CLI module should be importable as covariance_denoiser.cli."""
    parser = build_parser()
    assert parser.prog == "covariance-denoiser"


def test_run_offline_demo_writes_required_artifacts(tmp_path: Path) -> None:
    """One offline run should export the canonical compact artifact set."""
    config = OfflineDemoConfig(
        data_dir=Path("data/raw"),
        output_dir=tmp_path,
        lookback_days=63,
        horizon_days=21,
        min_train_size=252,
        test_size=21,
        step_size=21,
        annualization_days=252,
        ridge_alpha=1.0,
    )

    run_offline_demo_pipeline(config=config)

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "fold_predictions.csv").exists()
    assert (tmp_path / "model_coefficients.csv").exists()
    assert (tmp_path / "summary.md").exists()
    assert len(list(tmp_path.glob("*.png"))) >= 1


def test_tracked_raw_files_stay_below_size_budget() -> None:
    """Tracked raw artifacts should remain under 100 MB total."""
    tracked_paths = [
        Path("data/raw/raw_prices.parquet"),
        Path("data/raw/raw_prices.metadata.json"),
    ]
    total_bytes = sum(path.stat().st_size for path in tracked_paths)
    assert total_bytes < 100 * 1024 * 1024
