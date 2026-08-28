"""Run locally: python scripts/verify_against_smard.py <auction_sequence_run_version> <output_run_version>

Example:
  python scripts/verify_against_smard.py a03fix_v1 smard_verification_v1

Cross-checks sequence 1 vs sequence 2 disagreeing intervals (from
verify_auction_sequence.py's saved all_disagreeing_intervals.csv)
against SMARD -- the Bundesnetzagentur's (Germany's Federal Network
Agency) official electricity market transparency platform -- as an
independent reference for the true DE-LU day-ahead price.

API CONTRACT, verified before writing this script (not assumed):
  - filter=4169 is documented in bundesAPI/smard-api's openapi.yaml as
    "Marktpreis: Deutschland/Luxemburg" -- confirmed against the raw
    YAML source, not a secondhand summary.
  - Two-step fetch: GET .../index_quarterhour.json returns available
    chunk-start timestamps; GET .../{filter}_{region}_quarterhour_
    {chunk_timestamp}.json returns that chunk's [ms_epoch, price] series.
  - TIMESTAMP CONVENTION VERIFIED against a real, independently
    reported data point (bundesAPI/smard-api issue #21): ms-epoch
    1733094000000 is stated to correspond to "Mon Dec 02 2024 00:00:00
    GMT+0100" -- converting that value as a genuine UTC epoch and then
    to Europe/Berlin reproduces exactly that local time. SMARD's
    timestamps are real UTC instants, not local-time-mislabeled-as-UTC;
    standard pd.to_datetime(..., unit="ms", utc=True) is correct here.

INDEPENDENCE, and its limits: SMARD ultimately reflects the same
real-world EPEX SDAC auction result as ENTSO-E's Transparency Platform
feed -- it is not a second, different auction. But it is retrieved
through a genuinely separate publication pipeline than the ENTSO-E
Transparency Platform XML documents this whole project's data comes
from, and SMARD publishes ONE number per interval, not two competing
classificationSequence candidates. Whichever ENTSO-E sequence matches
SMARD's published number is strong evidence for which one reflects the
real auction outcome -- a meaningful check, even though it is not a
second independent auction.

STANDING CAVEAT: never touches 2026 (only ever queries the same
Oct-Dec 2025 window verify_auction_sequence.py already audits).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from src.utils import REPO_ROOT

SMARD_BASE_URL = "https://www.smard.de/app"
SMARD_FILTER_DELU_PRICE = 4169
SMARD_REGION = "DE-LU"
SMARD_RESOLUTION = "quarterhour"
PRICE_MATCH_TOLERANCE_EUR_MWH = 0.01  # rounding-level tolerance, not a real disagreement


def fetch_smard_chunk_index(
    filter_id: int = SMARD_FILTER_DELU_PRICE,
    region: str = SMARD_REGION,
    resolution: str = SMARD_RESOLUTION,
    session=None,
) -> List[pd.Timestamp]:
    """Returns the available chunk-start timestamps (UTC) for this
    filter/region/resolution. SMARD serves data in chunks, not one
    continuous series -- every chunk overlapping the target window
    must be fetched and concatenated separately.
    """
    session = session or requests
    url = f"{SMARD_BASE_URL}/chart_data/{filter_id}/{region}/index_{resolution}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return sorted(pd.Timestamp(ts, unit="ms", tz="UTC") for ts in data["timestamps"])


def fetch_smard_chunk(
    chunk_start_utc: pd.Timestamp,
    filter_id: int = SMARD_FILTER_DELU_PRICE,
    region: str = SMARD_REGION,
    resolution: str = SMARD_RESOLUTION,
    session=None,
) -> pd.DataFrame:
    """Fetches ONE chunk's price series. Returns
    (timestamp_utc, price_eur_mwh). Rows with a null price (SMARD
    publishes null for not-yet-available intervals -- see
    bundesAPI/smard-api issue #21) are dropped, not silently coerced
    to 0 or forward-filled.
    """
    session = session or requests
    chunk_ms = int(chunk_start_utc.timestamp() * 1000)
    url = f"{SMARD_BASE_URL}/chart_data/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{chunk_ms}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["series"], columns=["ts_ms", "price_eur_mwh"])
    df["timestamp_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df.dropna(subset=["price_eur_mwh"])[["timestamp_utc", "price_eur_mwh"]]


def select_relevant_chunks(
    chunk_starts: List[pd.Timestamp], start_utc: pd.Timestamp, end_utc: pd.Timestamp
) -> List[pd.Timestamp]:
    """Given all available chunk-start timestamps (sorted ascending),
    returns the subset needed to cover [start_utc, end_utc): the last
    chunk starting at or before start_utc (since that chunk's data
    extends forward past its own start), plus every chunk starting
    strictly after start_utc and before end_utc.
    """
    before_or_at_start = [c for c in chunk_starts if c <= start_utc]
    strictly_between = [c for c in chunk_starts if start_utc < c < end_utc]
    selected = ([before_or_at_start[-1]] if before_or_at_start else []) + strictly_between
    return selected


def fetch_smard_day_ahead_price(
    start_utc: pd.Timestamp, end_utc: pd.Timestamp, session=None
) -> pd.DataFrame:
    """Orchestrates the full fetch: index -> select relevant chunks ->
    fetch each -> concatenate -> clip to [start_utc, end_utc).
    """
    chunk_starts = fetch_smard_chunk_index(session=session)
    selected = select_relevant_chunks(chunk_starts, start_utc, end_utc)
    if not selected:
        raise ValueError(f"No SMARD chunks found covering [{start_utc}, {end_utc}).")
    frames = [fetch_smard_chunk(c, session=session) for c in selected]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp_utc"])
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
    mask = (combined["timestamp_utc"] >= start_utc) & (combined["timestamp_utc"] < end_utc)
    return combined[mask].reset_index(drop=True)


def classify_match(
    smard_price: float, price_seq1: float, price_seq2: float, tolerance: float = PRICE_MATCH_TOLERANCE_EUR_MWH
) -> str:
    """Classifies ONE row: does SMARD's price match sequence 1,
    sequence 2, both (within tolerance -- shouldn't happen for rows
    that were flagged as disagreeing, but guarded rather than assumed),
    or neither?
    """
    if pd.isna(smard_price):
        return "no_smard_data"
    diff1 = abs(smard_price - price_seq1)
    diff2 = abs(smard_price - price_seq2)
    match1 = diff1 <= tolerance
    match2 = diff2 <= tolerance
    if match1 and match2:
        return "ambiguous_both_match"
    if match1:
        return "matches_seq1"
    if match2:
        return "matches_seq2"
    return "matches_neither"


def compare_against_sequences(
    smard_df: pd.DataFrame, disagreeing_df: pd.DataFrame, tolerance: float = PRICE_MATCH_TOLERANCE_EUR_MWH
) -> pd.DataFrame:
    merged = disagreeing_df.merge(
        smard_df.rename(columns={"price_eur_mwh": "price_smard"}), on="timestamp_utc", how="left"
    )
    merged["match"] = merged.apply(
        lambda row: classify_match(row["price_smard"], row["price_seq1"], row["price_seq2"], tolerance), axis=1
    )
    return merged


def resolve_run_args(args: list) -> tuple:
    if len(args) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python scripts/verify_against_smard.py <auction_sequence_run_version> <output_run_version>\n\n"
            "Example:\n"
            "  python scripts/verify_against_smard.py a03fix_v1 smard_verification_v1"
        )
    return args[0], args[1]


def main():
    auction_sequence_run_version, output_run_version = resolve_run_args(sys.argv[1:])

    disagreeing_path = (
        REPO_ROOT / "outputs" / "auction_sequence_verification" / auction_sequence_run_version
        / "all_disagreeing_intervals.csv"
    )
    if not disagreeing_path.exists():
        raise FileNotFoundError(
            f"No disagreeing-intervals file at {disagreeing_path}. Run "
            f"'python scripts/verify_auction_sequence.py {auction_sequence_run_version}' first."
        )

    out_dir = REPO_ROOT / "outputs" / "auction_sequence_verification" / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already contains results. Pass a new output_run_version.")

    disagreeing = pd.read_csv(disagreeing_path)
    disagreeing["timestamp_utc"] = pd.to_datetime(disagreeing["timestamp_utc"], utc=True)

    start_utc = disagreeing["timestamp_utc"].min()
    end_utc = disagreeing["timestamp_utc"].max() + pd.Timedelta(minutes=15)
    print(f"Fetching SMARD DE-LU day-ahead prices for [{start_utc}, {end_utc})...")

    smard_df = fetch_smard_day_ahead_price(start_utc, end_utc)
    print(f"Fetched {len(smard_df)} SMARD price rows.")

    result = compare_against_sequences(smard_df, disagreeing)

    print("\n" + "=" * 78)
    print("SMARD CROSS-CHECK RESULTS")
    print("=" * 78)
    counts = result["match"].value_counts()
    print(counts.to_string())
    print()

    n_seq1 = int(counts.get("matches_seq1", 0))
    n_seq2 = int(counts.get("matches_seq2", 0))
    n_neither = int(counts.get("matches_neither", 0))
    n_no_data = int(counts.get("no_smard_data", 0))
    n_total_classified = n_seq1 + n_seq2 + n_neither

    if n_total_classified > 0:
        print(f"Of {n_total_classified} intervals with a SMARD reference price:")
        print(f"  matches sequence 1: {n_seq1} ({n_seq1/n_total_classified*100:.1f}%)")
        print(f"  matches sequence 2: {n_seq2} ({n_seq2/n_total_classified*100:.1f}%)")
        print(f"  matches neither:    {n_neither} ({n_neither/n_total_classified*100:.1f}%)")
    if n_no_data:
        print(f"\n{n_no_data} interval(s) had no SMARD data available -- excluded from the above.")

    print("\n" + "=" * 78)
    print("HARD RULE (per README):")
    if n_total_classified == 0:
        print("  No classifiable intervals -- cannot apply the rule.")
    elif n_seq1 / n_total_classified > 0.95 and n_seq2 / n_total_classified < 0.05:
        print("  SMARD consistently matches sequence 1 -> CLOSE the auction_sequence==1 assumption.")
    elif n_seq2 / n_total_classified > 0.95 and n_seq1 / n_total_classified < 0.05:
        print("  SMARD consistently matches sequence 2 -> STOP. Target requires reconstruction.")
    else:
        print("  MIXED result -- do not guess. Investigate the matches_neither and split cases directly.")
    print("=" * 78)

    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "smard_cross_check_results.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "auction_sequence_run_version": auction_sequence_run_version,
        "output_run_version": output_run_version,
        "smard_filter": SMARD_FILTER_DELU_PRICE,
        "smard_region": SMARD_REGION,
        "smard_resolution": SMARD_RESOLUTION,
        "price_match_tolerance_eur_mwh": PRICE_MATCH_TOLERANCE_EUR_MWH,
        "window_start_utc": str(start_utc),
        "window_end_utc": str(end_utc),
        "match_counts": counts.to_dict(),
        "holdout_used": False,
    }
    with open(out_dir / "smard_verification_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved results + manifest to {out_dir}")


if __name__ == "__main__":
    main()
