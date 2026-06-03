"""Raw parquet cache contract utilities.

This module validates the offline raw-cache contract used by all runtime
commands and notebooks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS: tuple[str, str, str] = ("date", "asset", "close")
REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "row_count",
    "assets",
    "start_date",
    "end_date",
    "frequency",
    "timestamp_timezone",
    "price_field",
    "is_adjusted_close",
)


def project_root_from_module() -> Path:
    """Resolve repository root from this module location."""
    return Path(__file__).resolve().parent.parent.parent.parent


def default_raw_data_dir() -> Path:
    """Return default raw data directory under the repository root."""
    return project_root_from_module() / "data" / "raw"


def _raw_data_dir(data_dir: Path | None) -> Path:
    """Resolve caller path or fall back to the tracked default raw directory."""
    return default_raw_data_dir() if data_dir is None else data_dir


def raw_parquet_path(data_dir: Path | None = None) -> Path:
    """Return expected path for tracked raw parquet file."""
    return _raw_data_dir(data_dir) / "raw_prices.parquet"


def raw_metadata_path(data_dir: Path | None = None) -> Path:
    """Return expected path for tracked raw metadata file."""
    return _raw_data_dir(data_dir) / "raw_prices.metadata.json"


@dataclass(frozen=True)
class RawCacheValidationResult:
    """Validation result object for raw cache checks.

    Parameters
    ----------
    is_valid
        Boolean flag for cache validity.
    message
        Human-readable validation outcome.
    parquet_path
        Absolute or relative parquet path used by validation.
    metadata_path
        Absolute or relative metadata path used by validation.
    """

    is_valid: bool
    message: str
    parquet_path: Path
    metadata_path: Path


def _invalid_result(
    message: str, parquet_path: Path, metadata_path: Path
) -> RawCacheValidationResult:
    """Build a failed validation result object."""
    return RawCacheValidationResult(
        is_valid=False,
        message=message,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load metadata JSON and ensure it is a dictionary.

    Raises
    ------
    ValueError
        If JSON cannot be parsed or top-level value is not an object.
    """
    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON metadata: {metadata_path}") from error

    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a JSON object: {metadata_path}")

    return metadata


def validate_raw_cache(data_dir: Path | None = None) -> RawCacheValidationResult:
    """Validate parquet and metadata under the raw data contract.

    Parameters
    ----------
    data_dir
        Raw data directory. If not provided, defaults to `data/raw/`.

    Returns
    -------
    RawCacheValidationResult
        Result object describing cache validity and file paths.
    """
    parquet_path = raw_parquet_path(data_dir)
    metadata_path = raw_metadata_path(data_dir)

    if not parquet_path.exists():
        return _invalid_result(f"Missing file: {parquet_path}", parquet_path, metadata_path)

    if not metadata_path.exists():
        return _invalid_result(f"Missing file: {metadata_path}", parquet_path, metadata_path)

    frame = pd.read_parquet(parquet_path)

    try:
        metadata = _load_metadata(metadata_path)
    except ValueError as error:
        return _invalid_result(str(error), parquet_path, metadata_path)

    # Metadata must declare every reproducibility field before we trust row counts or dates.
    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            return _invalid_result(
                f"Invalid metadata: missing key '{key}'",
                parquet_path,
                metadata_path,
            )

    actual_columns = list(frame.columns)
    if actual_columns != list(REQUIRED_COLUMNS):
        return _invalid_result(
            f"Invalid parquet columns: expected {list(REQUIRED_COLUMNS)}, got {actual_columns}",
            parquet_path,
            metadata_path,
        )

    if len(frame) != int(metadata["row_count"]):
        return _invalid_result(
            f"Invalid row_count: expected {metadata['row_count']}, got {len(frame)}",
            parquet_path,
            metadata_path,
        )

    actual_assets = sorted(frame["asset"].astype(str).unique().tolist())
    expected_assets = sorted(str(symbol) for symbol in metadata["assets"])
    if actual_assets != expected_assets:
        return _invalid_result(
            f"Invalid assets: expected {expected_assets}, got {actual_assets}",
            parquet_path,
            metadata_path,
        )

    parsed_dates = pd.to_datetime(frame["date"])
    actual_start = parsed_dates.min().strftime("%Y-%m-%d")
    actual_end = parsed_dates.max().strftime("%Y-%m-%d")

    if actual_start != str(metadata["start_date"]):
        return _invalid_result(
            f"Invalid start_date: expected {metadata['start_date']}, got {actual_start}",
            parquet_path,
            metadata_path,
        )

    if actual_end != str(metadata["end_date"]):
        return _invalid_result(
            f"Invalid end_date: expected {metadata['end_date']}, got {actual_end}",
            parquet_path,
            metadata_path,
        )

    if str(metadata["frequency"]) != "1d":
        return _invalid_result(
            f"Invalid frequency: expected '1d', got {metadata['frequency']}",
            parquet_path,
            metadata_path,
        )

    if str(metadata["price_field"]) != "close":
        return _invalid_result(
            f"Invalid price_field: expected 'close', got {metadata['price_field']}",
            parquet_path,
            metadata_path,
        )

    if metadata["is_adjusted_close"] is not True:
        return _invalid_result(
            f"Invalid is_adjusted_close: expected True, got {metadata['is_adjusted_close']}",
            parquet_path,
            metadata_path,
        )

    return RawCacheValidationResult(
        is_valid=True,
        message="Raw cache is valid.",
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )
