"""Command-line interface for the offline research pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from covariance_denoiser.data.clickhouse_refresh import (
    ClickHouseRefreshConfig,
    refresh_raw_cache_from_clickhouse,
)
from covariance_denoiser.pipelines.offline_demo import OfflineDemoConfig, run_offline_demo_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser exposing the offline demo command and optional raw refresh command.
    """
    parser = argparse.ArgumentParser(prog="covariance-denoiser")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser(
        "run-offline-demo",
        help="Run the offline realized variance research pipeline.",
    )
    demo_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw_prices.parquet and raw_prices.metadata.json.",
    )
    demo_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo"),
        help="Directory for exported offline demo artifacts.",
    )
    demo_parser.add_argument("--lookback-days", type=int, default=63)
    demo_parser.add_argument("--horizon-days", type=int, default=21)
    demo_parser.add_argument("--min-train-size", type=int, default=252)
    demo_parser.add_argument("--test-size", type=int, default=21)
    demo_parser.add_argument("--step-size", type=int, default=21)
    demo_parser.add_argument("--annualization-days", type=int, default=252)
    demo_parser.add_argument("--ridge-alpha", type=float, default=1.0)

    refresh_parser = subparsers.add_parser(
        "refresh-raw-cache",
        help="Optional one-time ClickHouse refresh for data/raw cache.",
    )
    refresh_parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    refresh_parser.add_argument("--host", type=str, required=True)
    refresh_parser.add_argument("--port", type=int, default=8123)
    refresh_parser.add_argument("--username", type=str, required=True)
    refresh_parser.add_argument("--password", type=str, required=True)
    refresh_parser.add_argument("--database", type=str, required=True)
    refresh_parser.add_argument("--table", type=str, required=True)
    refresh_parser.add_argument("--start-date", type=str, required=True)
    refresh_parser.add_argument("--end-date", type=str, required=True)
    refresh_parser.add_argument(
        "--assets",
        nargs="+",
        required=True,
        help="Space-separated asset symbols to query.",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch selected command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-offline-demo":
        config = OfflineDemoConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            lookback_days=args.lookback_days,
            horizon_days=args.horizon_days,
            min_train_size=args.min_train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            annualization_days=args.annualization_days,
            ridge_alpha=args.ridge_alpha,
        )
        run_offline_demo_pipeline(config)
    elif args.command == "refresh-raw-cache":
        refresh_config = ClickHouseRefreshConfig(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            database=args.database,
            table=args.table,
        )
        refresh_raw_cache_from_clickhouse(
            data_dir=args.data_dir,
            config=refresh_config,
            start_date=datetime.fromisoformat(args.start_date),
            end_date=datetime.fromisoformat(args.end_date),
            assets=[str(symbol) for symbol in args.assets],
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
