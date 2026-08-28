"""Run locally: python diagnose_price_duplicates.py

Reproduces the day-ahead price fetch + resolution selection exactly as
run_ingestion.py does, then inspects the *remaining* duplicate
timestamps (there shouldn't be many after select_price_resolution).
Prints the actual colliding rows so we can see the real cause instead
of guessing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> repo root, for `src` imports

import pandas as pd
from src.entsoe_client import EntsoeClient, get_default_date_range, select_price_resolution
from src.utils import load_config


def main():
    cfg = load_config()
    client = EntsoeClient()
    start, end = get_default_date_range(cfg)

    # Fetch raw (pre-selection) so we can see resolution_min per row
    raw = client._fetch_generic(
        start, end,
        params_extra={
            "documentType": "A44",
            "in_Domain": client.eic_code,
            "out_Domain": client.eic_code,
        },
        data_type="day_ahead_price",
        unit="EUR/MWh",
        value_tag="price.amount",
        value_col="price_eur_mwh",
        keep_resolution=True,
    )
    print(f"Raw rows (pre-selection): {len(raw)}")
    print("Resolution breakdown:")
    print(raw["resolution_min"].value_counts())
    print()

    selected = select_price_resolution(raw, cutover=cfg["market_design"]["fifteen_min_mtu_start"])
    print(f"Rows after resolution selection: {len(selected)}")

    dupe_mask = selected.duplicated(subset=["timestamp_utc"], keep=False)
    dupes = selected[dupe_mask].sort_values("timestamp_utc")
    print(f"Duplicate timestamps remaining: {dupes['timestamp_utc'].nunique()} unique timestamps, "
          f"{len(dupes)} total rows involved ({len(dupes) / len(selected):.1%} of selected rows)")
    print()

    if len(dupes) == 0:
        print("No duplicates found -- selection is clean.")
        return

    print("First 20 duplicate timestamps with their values:")
    print(dupes.head(20).to_string())
    print()

    print("Distribution of duplicate timestamps by year (helps spot if it's concentrated in one period):")
    dupes_by_year = dupes.copy()
    dupes_by_year["year"] = pd.to_datetime(dupes_by_year["timestamp_utc"]).dt.year
    print(dupes_by_year.groupby("year").size())
    print()

    print("Distribution of duplicate timestamps by minute-of-hour (helps spot resolution mixing):")
    dupes_by_minute = dupes.copy()
    dupes_by_minute["minute"] = pd.to_datetime(dupes_by_minute["timestamp_utc"]).dt.minute
    print(dupes_by_minute.groupby("minute").size())
    print()

    # Check: are the duplicate values actually identical, or genuinely different prices?
    grouped = dupes.groupby("timestamp_utc")["price_eur_mwh"].agg(["nunique", "count", "min", "max"])
    identical = (grouped["nunique"] == 1).sum()
    different = (grouped["nunique"] > 1).sum()
    print(f"Of {len(grouped)} duplicated timestamps: {identical} have IDENTICAL prices across copies, "
          f"{different} have DIFFERENT prices across copies.")
    print()
    print("Sample of timestamps where duplicate copies have DIFFERENT prices:")
    print(grouped[grouped["nunique"] > 1].head(10))


if __name__ == "__main__":
    main()
