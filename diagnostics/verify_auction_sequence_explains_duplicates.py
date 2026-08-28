"""Run locally: python diagnostics/verify_auction_sequence_explains_duplicates.py

Tests a specific hypothesis about the "~3x more price rows than
expected" / duplicate-timestamp findings from diagnose_price_duplicates.py
and inspect_price_xml.py: those scripts predate the auction_sequence
discovery and only ever apply select_price_resolution(), never
select_primary_auction_sequence() -- unlike the real production
fetch_day_ahead_prices(), which applies BOTH (sequence selection first,
then resolution selection).

Confirmed against the structural audit's own finding (5,538 Period
rows, zero exceptions): pre-cutover, ONLY sequence 1 has PT60M rows
(sequence 2 is PT15M pre-cutover too, so resolution-filtering alone
already excludes it cleanly). Post-cutover, BOTH sequences are PT15M --
resolution-filtering alone lets both through, producing a genuine
duplicate timestamp for every post-cutover quarter-hour.

Prediction this script tests directly against real cached data:
  1. Resolution-selection-only duplicates should be heavily or
     exclusively concentrated in the post-2025-10-01 window.
  2. Applying select_primary_auction_sequence() before
     select_price_resolution() (matching real production) should
     eliminate those duplicates entirely.

If both hold, the "duplicate rows" / "~3x too many rows" finding from
the earlier diagnostic scripts is very likely fully explained by the
auction_sequence duality already discovered and correctly handled in
production -- not a new, separate bug. If either doesn't hold, there is
something ELSE going on that these earlier scripts correctly flagged
and that still needs investigating.

Reads from the existing ENTSO-E cache (no new API calls if this
window is already ingested) via the same low-level path
diagnose_price_duplicates.py uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.entsoe_client import (
    EntsoeClient,
    get_default_date_range,
    select_price_resolution,
    select_primary_auction_sequence,
)
from src.clean import local_delivery_date_to_utc
from src.utils import load_config, setup_logging


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    client = EntsoeClient()
    start, end = get_default_date_range(cfg)
    cutover = local_delivery_date_to_utc(cfg["market_design"]["fifteen_min_mtu_start"])

    logger.info("Fetching raw price data (both sequences, both resolutions, from cache)...")
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
        keep_auction_sequence=True,
    )
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    print(f"\nRaw rows (unfiltered): {len(raw)}")

    # --- Reproduces diagnose_price_duplicates.py exactly: resolution selection ONLY ---
    resolution_only = select_price_resolution(raw, cutover=cfg["market_design"]["fifteen_min_mtu_start"])
    dupe_mask = resolution_only.duplicated(subset=["timestamp_utc"], keep=False)
    dupes = resolution_only[dupe_mask].copy()

    print("\n" + "=" * 78)
    print("STEP 1: resolution-selection ONLY (matches diagnose_price_duplicates.py)")
    print("=" * 78)
    print(f"Rows after resolution selection: {len(resolution_only)}")
    print(f"Duplicate timestamps: {dupes['timestamp_utc'].nunique()} unique, {len(dupes)} total rows "
          f"({len(dupes) / max(len(resolution_only), 1):.1%} of selected rows)")

    if len(dupes) > 0:
        pre_count = (dupes["timestamp_utc"] < cutover).sum()
        post_count = (dupes["timestamp_utc"] >= cutover).sum()
        print(f"\nConcentration check (this is the falsifiable prediction):")
        print(f"  Duplicates BEFORE cutover: {pre_count}")
        print(f"  Duplicates FROM/AFTER cutover: {post_count}")
        if pre_count == 0:
            print("  -> PREDICTION HELD: zero pre-cutover duplicates, all concentrated post-cutover.")
        else:
            print("  -> PREDICTION DID NOT FULLY HOLD: pre-cutover duplicates exist too -- "
                  "the auction_sequence explanation alone may not cover everything found here. "
                  "Investigate the pre-cutover duplicate rows directly, don't assume this is "
                  "already-understood territory.")

    # --- Real production order: sequence selection THEN resolution selection ---
    production_equivalent = select_price_resolution(
        select_primary_auction_sequence(raw), cutover=cfg["market_design"]["fifteen_min_mtu_start"]
    )
    dupe_mask_prod = production_equivalent.duplicated(subset=["timestamp_utc"], keep=False)

    print("\n" + "=" * 78)
    print("STEP 2: sequence selection THEN resolution selection (matches real production")
    print("fetch_day_ahead_prices())")
    print("=" * 78)
    print(f"Rows after both selections: {len(production_equivalent)}")
    print(f"Duplicate timestamps: {dupe_mask_prod.sum()}")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    if len(dupes) > 0 and dupe_mask_prod.sum() == 0 and (dupes["timestamp_utc"] < cutover).sum() == 0:
        print("The auction_sequence selection step fully explains the duplicate/row-count finding")
        print("from the earlier diagnostic scripts. This is the SAME phenomenon already discovered")
        print("and correctly handled in production fetch_day_ahead_prices() -- not a new, separate bug.")
    elif dupe_mask_prod.sum() > 0:
        print("Duplicates REMAIN even after applying both selections -- this is NOT fully explained")
        print("by the auction_sequence duality. There is something else going on; investigate the")
        print("remaining duplicate rows directly (print them, check their auction_sequence/resolution_min")
        print("values, don't assume this is already-understood territory).")
    else:
        print("No duplicates found even with resolution-selection alone -- the original finding may")
        print("have been specific to a data range not covered by this run, or already resolved by")
        print("an unrelated change. Worth re-checking against the exact original observation if you")
        print("still have it.")
    print("=" * 78)


if __name__ == "__main__":
    main()
