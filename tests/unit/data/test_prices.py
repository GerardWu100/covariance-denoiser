"""Tests for local raw price loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from covariance_denoiser.data.prices import load_raw_prices
from covariance_denoiser.data.raw_cache import raw_metadata_path, raw_parquet_path


def test_default_raw_cache_paths_live_under_data_raw() -> None:
    """Default runtime paths should point to tracked files under data/raw/."""
    assert raw_parquet_path(Path("data/raw")).exists()
    assert raw_metadata_path(Path("data/raw")).exists()


def test_loader_reads_parquet_without_clickhouse_configuration() -> None:
    """Loading local parquet should succeed without database configuration."""
    prices = load_raw_prices(data_dir=Path("data/raw"))

    assert {"date", "asset", "close"}.issubset(set(prices.columns))
    assert len(prices) > 0


def test_loader_raises_actionable_error_for_missing_files(tmp_path: Path) -> None:
    """Missing raw files should raise a clear file-level error."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_raw_prices(data_dir=tmp_path)

    error_text = str(exc_info.value)
    assert "raw_prices.parquet" in error_text or "raw_prices.metadata.json" in error_text
