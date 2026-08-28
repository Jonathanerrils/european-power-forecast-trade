"""Run this locally (not in a sandboxed environment) with ENTSOE_TOKEN set.

    export ENTSOE_TOKEN=your-token-here
    python run_ingestion.py

Pulls DE-LU day-ahead prices, load forecast, and wind/solar generation
forecast from ENTSO-E for the date range in config.yaml (default:
2019-01-01 through the latest complete month), cleans and merges them
via src/clean.py, and writes:

    data/processed/delu_hourly.parquet   -- the clean merged dataset
    data/raw/ingestion_log.jsonl         -- full ingestion metadata log

This intentionally does NOT build features, splits, or models -- it's
just steps 2-4 of the build order (ingest -> time-normalize -> merge).
Prints a summary at the end so results can be sanity-checked and shared.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.entsoe_client import EntsoeClient, get_default_date_range
from src.clean import build_clean_dataset, validate_hourly_coverage
from src.utils import load_config, setup_logging, REPO_ROOT


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    client = EntsoeClient()
    start, end = get_default_date_range(cfg)
    logger.info("Fetching DE-LU data from %s to %s", start.date(), end.date())

    t0 = time.time()

    logger.info("Fetching day-ahead prices...")
    price_df = client.fetch_day_ahead_prices(start, end)
    logger.info("  -> %d raw rows", len(price_df))

    logger.info("Fetching load forecast...")
    load_df = client.fetch_load_forecast(start, end)
    logger.info("  -> %d raw rows", len(load_df))

    logger.info("Fetching wind/solar forecast...")
    ws_df = client.fetch_wind_solar_forecast(start, end)
    logger.info("  -> %d raw rows", len(ws_df))

    client.save_ingestion_log()

    logger.info("Cleaning and merging...")
    clean_df, reports = build_clean_dataset(
        price_df, load_df, ws_df,
        local_tz=cfg["market"]["timezone_local"],
        market_design_cutover=cfg["market_design"]["fifteen_min_mtu_start"],
        start_utc=start,
        end_utc=end,
    )

    import pandas as pd
    missing_hours = validate_hourly_coverage(
        clean_df,
        expected_start=pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC"),
        expected_end=pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC"),
    )
    if missing_hours:
        logger.warning("%d expected UTC hours are absent from ALL sources", len(missing_hours))

    out_path = REPO_ROOT / cfg["data"]["processed_dir"] / "delu_hourly.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output:  {out_path}")
    print(f"Rows:    {len(clean_df)}")
    print(f"Date range: {clean_df['timestamp_utc'].min()} -> {clean_df['timestamp_utc'].max()}")
    print("\nCleaning reports:")
    for r in reports:
        print(f"  {r.dataset_name:24s} raw={r.raw_row_count:>7} clean={r.clean_row_count:>7} "
              f"missing={r.missing_count} duplicates_removed={r.duplicate_count}")
    print(f"\nHours missing from ALL sources: {len(missing_hours)}")
    print("\nColumn summary:")
    print(clean_df.dtypes)
    print("\nMissing values per column:")
    print(clean_df.isna().sum())
    print("\nFirst 3 rows:")
    print(clean_df.head(3).to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
