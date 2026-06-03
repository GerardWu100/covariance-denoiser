"""Tests for raw cache validation contract."""

from __future__ import annotations

import json
from pathlib import Path

from covariance_denoiser.data.raw_cache import REQUIRED_METADATA_KEYS, validate_raw_cache


def test_validate_raw_cache_succeeds_for_tracked_dataset() -> None:
    """Tracked parquet and metadata should satisfy the offline cache contract."""
    result = validate_raw_cache(data_dir=Path("data/raw"))

    assert result.is_valid is True


def test_metadata_contains_required_reproducibility_keys() -> None:
    """Metadata should expose all fields required by the offline contract."""
    metadata_path = Path("data/raw/raw_prices.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    for required_key in REQUIRED_METADATA_KEYS:
        assert required_key in metadata
