"""Run locally: python scripts/verify_auction_sequence.py [n_sample_dates]

Verifies the open, documented assumption in
entsoe_client.py::select_primary_auction_sequence(): that
classificationSequence position 1 (not 2) is the correct primary
DE-LU SDAC day-ahead auction result from the 2025-10-01 delivery day
onward. Until this audit, the pipeline kept sequence 1 as a reasoned
but externally unverified assumption.

WHAT THIS SCRIPT CAN AND CANNOT DO
-----------------------------------
It CAN: pull both sequence-1 and sequence-2 PT15M price series through
the normal ENTSO-E client/cache path and identify exactly which
intervals actually differ. Already-cached request windows generate no
new API calls; cache misses are fetched from ENTSO-E by the normal
client and then cached. Only intervals where sequence 1 and sequence 2
disagree can discriminate between the two hypotheses, so the script
focuses the manual audit on those rows rather than asking you to scan
thousands of identical prices.

It deliberately DOES NOT automate the EPEX-side verification through
unofficial or reverse-engineered endpoints. The external reference is
EPEX SPOT's official public Market Results page for DE-LU SDAC
15-minute auction prices. Reliable licensed programmatic access exists
through paid EPEX/EEX data products, but this project does not need to
purchase it merely to resolve this one historical assumption.

The workflow is therefore:
  1. identify all ENTSO-E intervals where sequence 1 != sequence 2;
  2. select a temporally distributed sample of affected delivery dates,
     forcing in any DST-transition date that contains disagreements and
     including both large and small sequence differences where possible;
  3. print the exact official EPEX DE-LU SDAC 15-minute Market Results
     URL for each sampled date;
  4. manually compare the published EPEX quarter-hour price with
     price_seq1 (kept by the pipeline) and price_seq2 (dropped).

The manual step is intentional: for this audit, an official browser
reference is stronger evidence than an unofficial automated re-scrape.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.entsoe_client import EntsoeClient
from src.clean import local_delivery_date_to_utc
from src.utils import setup_logging, load_config, REPO_ROOT


LOCAL_TZ = "Europe/Berlin"
PRICE_DIFF_TOLERANCE = 0.005  # EUR/MWh; suppress float noise only


def find_sequence2_only_intervals(merged: pd.DataFrame) -> pd.DataFrame:
    """Intervals where sequence 2 exists but sequence 1 is absent --
    a DIFFERENT and more serious problem than "both present but
    disagree": the production selector (select_primary_auction_sequence,
    keeps position==1) would silently produce NO price at all for these
    intervals, a gap rather than a possibly-wrong value.
    """
    return merged[merged["price_seq1"].isna()][["timestamp_utc", "price_seq2"]].copy()


def assert_unique_sequence_timestamps(df: pd.DataFrame, sequence_name: str) -> None:
    """Fail loudly rather than allow a many-to-many merge. A duplicate
    timestamp within a single sequence's own series would silently
    fan out into multiple rows per interval when merged against the
    other sequence -- exactly the kind of collision this project has
    repeatedly found and fixed elsewhere (dual PT60M/PT15M price
    products, the two-auction-sequence collision itself). This script
    exists to audit a collision; it should not itself be vulnerable
    to a different, undetected one.
    """
    dupes = df["timestamp_utc"].duplicated(keep=False)
    if dupes.any():
        raise AssertionError(
            f"{sequence_name} contains {dupes.sum()} rows at duplicate timestamps -- "
            f"cannot perform auction-sequence verification safely. Investigate the raw "
            f"cached ENTSO-E data before proceeding; a many-to-many merge here would "
            f"silently corrupt the comparison this script exists to make trustworthy."
        )


def fetch_both_sequences(client: EntsoeClient, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch day-ahead prices WITHOUT filtering to sequence == 1.

    This uses the same ENTSO-E request/cache mechanism as the production
    client. Cached request windows are read locally; cache misses are
    fetched from ENTSO-E and cached by EntsoeClient._request().
    """
    return client._fetch_generic(
        start,
        end,
        params_extra={
            "documentType": "A44",
            "in_Domain": client.eic_code,
            "out_Domain": client.eic_code,
        },
        data_type="day_ahead_price_both_sequences",
        unit="EUR/MWh",
        value_tag="price.amount",
        value_col="price_eur_mwh",
        keep_resolution=True,
        keep_auction_sequence=True,
    )


def build_epex_url(delivery_date: str) -> str:
    """Official EPEX Market Results page for DE-LU SDAC 15-minute prices.

    The audit is specifically about the competing ENTSO-E PT15M auction
    sequences, so the EPEX reference must also be the 15-minute SDAC
    auction result, not the 60-minute index/product.
    """
    return (
        "https://www.epexspot.com/en/market-results"
        f"?auction=MRC&market_area=DE-LU&delivery_date={delivery_date}"
        "&modality=Auction&sub_modality=DayAhead&product=15&data_mode=table"
    )


def add_delivery_interval_position(seq1_df: pd.DataFrame, local_tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Assigns delivery_interval_position (1, 2, 3, ...) within each local
    delivery date, derived from the COMPLETE sequence-1 PT15M series --
    not from any disagreeing/sampled subset. Deriving it from a filtered
    subset would silently redefine "position 3" as "the third
    disagreement" rather than "the third market interval of the day",
    which defeats the purpose: this column exists specifically to give
    an unambiguous index when reading EPEX's rendered table, especially
    around a DST day's repeated local hour where clock labels alone are
    awkward to match by eye.

    Position is computed from ELAPSED TIME SINCE LOCAL DELIVERY-DAY
    MIDNIGHT, not row rank (cumcount) within the group. Row rank is only
    correct if the day's series has no gaps -- if even one interval is
    missing (a real possibility: this is live-fetched ENTSO-E data, not
    a synthetic guaranteed-complete series), cumcount silently shifts
    every later position in that day backward by one, which is exactly
    the kind of silent misalignment this column exists to prevent.
    Elapsed-time-based position instead SKIPS the missing position and
    leaves every other row correctly labelled -- verified by a dedicated
    regression test. Reuses local_delivery_date_to_utc() (the project's
    one canonical local-midnight-to-UTC conversion) rather than
    re-deriving day boundaries here.
    """
    df = seq1_df.copy()
    local = df["timestamp_utc"].dt.tz_convert(local_tz)
    df["local_delivery_date"] = local.dt.date

    unique_dates = df["local_delivery_date"].unique()
    day_start_utc = {d: local_delivery_date_to_utc(d.isoformat(), local_tz=local_tz) for d in unique_dates}
    df["_day_start_utc"] = df["local_delivery_date"].map(day_start_utc)

    elapsed_minutes = (df["timestamp_utc"] - df["_day_start_utc"]).dt.total_seconds() / 60.0
    df["delivery_interval_position"] = (elapsed_minutes / 15.0).round().astype(int) + 1
    return df[["timestamp_utc", "delivery_interval_position"]]


def find_dst_transition_dates(timestamps_utc: pd.Series, local_tz: str = LOCAL_TZ) -> set:
    """Return local delivery dates on which the UTC offset changes.

    Using the observed timestamps rather than hard-coding calendar dates
    keeps this audit valid for both spring-forward and autumn-fall-back
    transitions in any covered year.
    """
    ts = pd.to_datetime(timestamps_utc, utc=True)
    local = ts.dt.tz_convert(local_tz)
    frame = pd.DataFrame(
        {
            "local_delivery_date": local.dt.date,
            "utc_offset": local.map(lambda x: x.utcoffset()),
        }
    )
    counts = frame.groupby("local_delivery_date")["utc_offset"].nunique()
    return set(counts[counts > 1].index)


def select_sample_dates(
    disagreeing: pd.DataFrame,
    n_sample_dates: int,
    dst_transition_dates: set | None = None,
) -> list:
    """Choose an auditable, temporally distributed set of delivery dates.

    Priority is given to:
      - affected DST-transition dates;
      - earliest and latest affected dates;
      - dates with the largest and smallest non-zero max sequence gap;
      - evenly spaced dates across the remaining history.

    The final list is chronological and contains at most n_sample_dates
    unique dates. This avoids the old "first N dates in October" bias.
    """
    if n_sample_dates <= 0:
        raise ValueError("n_sample_dates must be >= 1")

    stats = (
        disagreeing.groupby("local_delivery_date")
        .agg(max_abs_diff=("abs_diff", "max"), n_disagreeing=("abs_diff", "size"))
        .sort_index()
    )
    all_dates = list(stats.index)
    if len(all_dates) <= n_sample_dates:
        return all_dates

    selected: list = []

    def add(d):
        if d in stats.index and d not in selected and len(selected) < n_sample_dates:
            selected.append(d)

    # Force in any affected DST transition first, since local clock labels
    # are intrinsically ambiguous on those dates without an offset.
    for d in sorted(dst_transition_dates or set()):
        add(d)

    add(all_dates[0])
    add(all_dates[-1])
    add(stats["max_abs_diff"].idxmax())
    add(stats["max_abs_diff"].idxmin())

    # Add dates spread roughly evenly across the entire affected period.
    slots = min(n_sample_dates, len(all_dates))
    if slots == 1:
        add(all_dates[len(all_dates) // 2])
    else:
        for i in range(slots):
            idx = round(i * (len(all_dates) - 1) / (slots - 1))
            add(all_dates[idx])

    # If duplicate priorities left spare capacity, fill with the largest
    # remaining disagreements first, then any remaining dates.
    for d in stats.sort_values("max_abs_diff", ascending=False).index:
        add(d)
    for d in all_dates:
        add(d)

    return sorted(selected)


def resolve_run_args(args: list) -> tuple:
    """run_version is MANDATORY (matching run_models.py/run_eda.py's
    established pattern), n_sample_dates is optional. An earlier
    version used a fixed, unversioned output path
    (outputs/auction_sequence_verification/), so re-running this audit
    against corrected data would silently overwrite the pre-A03-fix
    baseline -- exactly the mistake already caught and fixed for
    run_eda.py. Versioning this script's output BEFORE the corrected
    re-run means the pre-fix files (already on disk, at the old
    unversioned path) are left untouched rather than destroyed --
    unlike EDA, where the loss already happened and was unrecoverable.
    """
    if len(args) not in (1, 2):
        raise SystemExit(
            "Usage:\n"
            "  python scripts/verify_auction_sequence.py <run_version> [n_sample_dates]\n\n"
            "Example:\n"
            "  python scripts/verify_auction_sequence.py a03fix_v1 10"
        )
    run_version = args[0]
    n_sample_dates = int(args[1]) if len(args) > 1 else 10
    if n_sample_dates <= 0:
        raise SystemExit("n_sample_dates must be >= 1")
    return run_version, n_sample_dates


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    run_version, n_sample_dates = resolve_run_args(sys.argv[1:])
    out_dir = REPO_ROOT / "outputs" / "auction_sequence_verification" / run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Pass a new run_version instead of "
            f"overwriting it, e.g.\n"
            f"  python scripts/verify_auction_sequence.py {run_version}_v2"
        )

    cutover = local_delivery_date_to_utc("2025-10-01")
    # Audit ONLY the post-cutover DEVELOPMENT period (Oct-Dec 2025).
    # 2026 remains completely untouched by this script -- not just by
    # any forecasting metric, but by this audit too. The sampling
    # routine deliberately surfaces earliest/latest dates and the
    # largest/smallest discrepancies; if 2026 were in scope, that would
    # mean visually inspecting real 2026 target-price patterns before
    # the holdout evaluation, which is exactly the kind of holdout
    # peeking this project has been careful to avoid everywhere else.
    # Three months of real post-cutover data (Oct-Dec 2025) is already
    # enough to determine whether sequence 1 matches EPEX's published
    # result. 2026 stays out of scope until uncertainty, strategy, and
    # risk work are frozen and the final holdout evaluation begins.
    window_end = local_delivery_date_to_utc("2026-01-01")

    client = EntsoeClient()
    logger.info("Fetching BOTH auction sequences (not filtered) for %s -> %s", cutover, window_end)
    raw = fetch_both_sequences(client, cutover, window_end)

    if raw.empty:
        print(
            "\nNo ENTSO-E price data were returned for this window. Check your token, "
            "network/cache state, and whether the requested post-2025-10-01 range is available.\n"
        )
        return

    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)

    # Only PT15M rows are relevant to the sequence-1-vs-2 collision.
    raw = raw[raw["resolution_min"] == 15].copy()

    seq1 = raw[raw["auction_sequence"] == 1][["timestamp_utc", "price_eur_mwh"]].rename(
        columns={"price_eur_mwh": "price_seq1"}
    )
    seq2 = raw[raw["auction_sequence"] == 2][["timestamp_utc", "price_eur_mwh"]].rename(
        columns={"price_eur_mwh": "price_seq2"}
    )
    assert_unique_sequence_timestamps(seq1, "sequence 1")
    assert_unique_sequence_timestamps(seq2, "sequence 2")
    merged = seq1.merge(seq2, on="timestamp_utc", how="outer").sort_values("timestamp_utc")

    if merged.empty:
        print("\nNo PT15M rows with classificationSequence were found for this window.\n")
        return

    n_both = merged.dropna(subset=["price_seq1", "price_seq2"]).shape[0]
    n_seq1_only = merged["price_seq2"].isna().sum()
    n_seq2_only = merged["price_seq1"].isna().sum()

    print("\n" + "=" * 78)
    print("AUCTION SEQUENCE COVERAGE (PT15M rows, post-2025-10-01)")
    print("=" * 78)
    print(f"Intervals with BOTH sequence 1 and 2 present: {n_both}")
    print(f"Intervals with ONLY sequence 1:                {n_seq1_only}")
    print(f"Intervals with ONLY sequence 2:                {n_seq2_only}")

    # sequence-2-only intervals are a DIFFERENT and more serious problem
    # than "both present but disagree": the production selector
    # (select_primary_auction_sequence, keeps position==1) would discard
    # sequence 2 here and leave NO selected price for that interval at
    # all -- a silent gap, not just a possibly-wrong value. This does
    # NOT block the rest of the both-present audit below (that
    # comparison is still valid and useful on its own), but it must be
    # surfaced loudly and resolved separately before "sequence 1 is the
    # complete, correct primary series" can be declared verified.
    if n_seq2_only:
        seq2_only_intervals = find_sequence2_only_intervals(merged)
        out_dir.mkdir(parents=True, exist_ok=True)
        seq2_only_path = out_dir / "sequence2_only_intervals.csv"
        seq2_only_intervals.to_csv(seq2_only_path, index=False)
        print("\n" + "!" * 78)
        print(f"WARNING: {n_seq2_only} interval(s) have sequence 2 but NOT sequence 1.")
        print("The production selector (select_primary_auction_sequence) would silently")
        print("produce NO price for these intervals -- a gap, not just a possibly-wrong")
        print(f"value. Saved to: {seq2_only_path}")
        print("These must be investigated and resolved BEFORE declaring the sequence-1")
        print("assumption externally verified, regardless of how the both-present")
        print("comparison below turns out.")
        print("!" * 78)

    both = merged.dropna(subset=["price_seq1", "price_seq2"]).copy()
    both["abs_diff"] = (both["price_seq1"] - both["price_seq2"]).abs()
    disagreeing = both[both["abs_diff"] > PRICE_DIFF_TOLERANCE].copy()

    print(
        f"\nOf those {n_both} both-present intervals, {len(disagreeing)} DISAGREE "
        f"by more than {PRICE_DIFF_TOLERANCE:.3f} EUR/MWh."
    )

    if disagreeing.empty:
        print(
            "\nNo materially disagreeing intervals were found. Either the two sequences "
            "were identical throughout this window or sequence 2 was not simultaneously "
            "available often enough to adjudicate the assumption.\n"
        )
        return

    local = disagreeing["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    disagreeing = disagreeing.assign(
        local_delivery_date=local.dt.date,
        local_time_with_offset=local.dt.strftime("%H:%M %z"),
    ).sort_values("timestamp_utc")

    # Derived from the COMPLETE sequence-1 series (see
    # add_delivery_interval_position docstring) -- an unambiguous index
    # into the day's market intervals, independent of local-clock labels,
    # for reading off EPEX's rendered table around a DST day's repeated hour.
    positions = add_delivery_interval_position(seq1)
    disagreeing = disagreeing.merge(positions, on="timestamp_utc", how="left")

    # Detect DST transition dates from the full sequence-1 PT15M series, not
    # just the disagreeing subset, then force any affected transition date
    # into the manual audit sample.
    dst_dates = find_dst_transition_dates(seq1["timestamp_utc"]) if not seq1.empty else set()
    affected_dst_dates = dst_dates.intersection(set(disagreeing["local_delivery_date"].unique()))

    sample_dates = select_sample_dates(
        disagreeing,
        n_sample_dates=n_sample_dates,
        dst_transition_dates=affected_dst_dates,
    )

    print("\n" + "=" * 78)
    print(f"DISAGREEING INTERVALS (manual sample: {len(sample_dates)} delivery dates)")
    print("=" * 78)
    if affected_dst_dates:
        print(f"Affected DST-transition date(s) forced into sample: {sorted(affected_dst_dates)}")
        print("On these dates, use delivery_interval_position (not the local clock label) "
              "to match rows against EPEX's rendered table -- local labels repeat or skip "
              "around the transition hour and are not a reliable index on their own.")

    sampled_rows = []
    for d in sample_dates:
        day_rows = disagreeing[disagreeing["local_delivery_date"] == d].sort_values("timestamp_utc")
        print(f"\n--- {d} ({len(day_rows)} disagreeing interval(s) this day) ---")
        print(
            day_rows[
                [
                    "delivery_interval_position",
                    "timestamp_utc",
                    "local_time_with_offset",
                    "price_seq1",
                    "price_seq2",
                    "abs_diff",
                ]
            ].to_string(index=False)
        )
        sampled_rows.append(day_rows)
        epex_url = build_epex_url(str(d))
        print(f"  Official EPEX check: open in a browser -> {epex_url}")
        print(
            "  Compare the published DE-LU SDAC 15-minute price at each interval "
            "above with price_seq1 (kept) versus price_seq2 (dropped)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    all_out_path = out_dir / "all_disagreeing_intervals.csv"
    sample_out_path = out_dir / "sampled_disagreeing_intervals.csv"
    template_out_path = out_dir / "manual_epex_verification_template.csv"

    # delivery_interval_position first -- it's the unambiguous index a
    # reviewer should be matching against EPEX's table, especially on
    # DST-affected days where local clock labels alone aren't reliable.
    front_cols = ["delivery_interval_position", "local_delivery_date", "timestamp_utc", "local_time_with_offset"]
    col_order = front_cols + [c for c in disagreeing.columns if c not in front_cols]
    disagreeing = disagreeing[col_order]

    disagreeing.to_csv(all_out_path, index=False)
    sampled = pd.concat(sampled_rows, ignore_index=True)[col_order]
    sampled.to_csv(sample_out_path, index=False)

    manual_template = sampled.copy()
    # epex_source_url populated per-row (not just printed to console) so
    # each completed row is self-contained evidence, reproducible even if
    # the console output isn't kept.
    manual_template["epex_source_url"] = manual_template["local_delivery_date"].map(lambda d: build_epex_url(str(d)))
    manual_template["epex_official_price"] = pd.NA
    manual_template["matches_sequence"] = pd.NA  # enter 1, 2, both, or neither
    manual_template["verified_at_utc"] = pd.NA  # when you actually checked EPEX's page
    manual_template["evidence_filename"] = pd.NA  # screenshot/saved-page filename, kept alongside this CSV
    manual_template["reviewer_notes"] = pd.NA
    manual_template.to_csv(template_out_path, index=False)

    print("\n" + "=" * 78)
    print(f"Saved ALL disagreeing intervals to:      {all_out_path}")
    print(f"Saved sampled audit intervals to:        {sample_out_path}")
    print(f"Saved manual EPEX verification template: {template_out_path}")
    print("The remaining step is explicit manual verification against the official")
    print("EPEX DE-LU SDAC 15-minute Market Results pages printed above.")
    print("Record the official EPEX price and whether it matches sequence 1 or 2;")
    print("only then should the README limitation be confirmed or removed.")
    print("Match rows using delivery_interval_position, not just the local clock")
    print("label -- it's unambiguous even on the DST-affected sample date(s).")
    print("Recommended: save a screenshot or the rendered EPEX page for each audited")
    print("date alongside the completed CSV. The CSV records your conclusion; the")
    print("saved page/screenshot records what you actually observed, in case the")
    print("conclusion is ever questioned later.")
    if n_seq2_only:
        print("\nREMINDER: sequence2_only_intervals.csv also needs resolution -- the")
        print("sequence-1 assumption cannot be declared verified while it's outstanding,")
        print("independent of how the both-present comparison above turns out.")
    print("=" * 78)


if __name__ == "__main__":
    main()
