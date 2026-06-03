"""Optional ClickHouse-backed raw cache refresh.

This module is isolated from default runtime paths. It is only used when users
explicitly request one-time cache refresh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import clickhouse_connect
except ImportError:  # pragma: no cover
    clickhouse_connect = None


@dataclass(frozen=True)
class ClickHouseRefreshConfig:
    """Connection settings for ClickHouse raw-price refresh."""

    host: str
    port: int
    username: str
    password: str
    database: str
    table: str


def is_clickhouse_client_available() -> bool:
    """Return whether clickhouse-connect is available in the environment."""
    return clickhouse_connect is not None


def _write_raw_cache(data_dir: Path, frame: pd.DataFrame) -> None:
    """Write parquet and metadata contract to the raw cache directory."""
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "raw_prices.parquet"
    metadata_path = data_dir / "raw_prices.metadata.json"

    ordered_frame = frame.sort_values(["date", "asset"]).reset_index(drop=True)
    ordered_frame.to_parquet(parquet_path, index=False)

    parsed_dates = pd.to_datetime(ordered_frame["date"])
    metadata: dict[str, Any] = {
        "row_count": len(ordered_frame),
        "assets": sorted(ordered_frame["asset"].astype(str).unique().tolist()),
        "start_date": parsed_dates.min().strftime("%Y-%m-%d"),
        "end_date": parsed_dates.max().strftime("%Y-%m-%d"),
        "frequency": "1d",
        "timestamp_timezone": "America/New_York_market_close_proxy",
        "price_field": "close",
        "is_adjusted_close": True,
        "source": "clickhouse_refresh",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def refresh_raw_cache_from_clickhouse(
    data_dir: Path,
    config: ClickHouseRefreshConfig,
    start_date: datetime,
    end_date: datetime,
    assets: list[str],
) -> None:
    """Refresh raw parquet cache from ClickHouse.

    Parameters
    ----------
    data_dir
        Destination directory for raw cache files.
    config
        ClickHouse connection and table settings.
    start_date
        Inclusive start date.
    end_date
        Inclusive end date.
    assets
        Asset symbols to include.

    Raises
    ------
    RuntimeError
        If clickhouse-connect is unavailable in this runtime.
    """
    if clickhouse_connect is None:
        raise RuntimeError("clickhouse-connect is not installed. Cannot refresh raw cache.")

    quoted_assets = ", ".join(f"'{symbol}'" for symbol in assets)
    sql_query = (
        "SELECT date, asset, close "
        f"FROM {config.database}.{config.table} "
        f"WHERE date >= toDate('{start_date:%Y-%m-%d}') "
        f"AND date <= toDate('{end_date:%Y-%m-%d}') "
        f"AND asset IN ({quoted_assets}) "
        "ORDER BY date, asset"
    )

    client = clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
    )

    rows = client.query(sql_query).result_rows
    refreshed_prices = pd.DataFrame(rows, columns=["date", "asset", "close"])
    refreshed_prices["date"] = pd.to_datetime(refreshed_prices["date"])
    refreshed_prices["asset"] = refreshed_prices["asset"].astype(str)
    refreshed_prices["close"] = refreshed_prices["close"].astype(float)

    _write_raw_cache(data_dir=data_dir, frame=refreshed_prices)
