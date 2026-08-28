"""Time-normalization and cleaning layer (spec sections 4, 5, 6).

Core rules enforced here:
    - UTC is the canonical storage timezone. Local time (Europe/Berlin)
      is *derived* via tz_convert, never assumed via fixed offsets --
      this is what makes DST transitions safe. pandas' tz_convert uses
      the IANA tzdata rules, so spring/autumn transitions are handled
      correctly without special-casing in this code; we still add
      explicit tests for both transitions (tests/test_dst.py).
    - Before 2025-10-01: source data is already hourly. Pass through.
    - From 2025-10-01: source data is quarter-hourly.
        * Price (EUR/MWh, an intensity, not summed energy) -> mean of
          the four 15-minute prices, matching EPEX's own 60-minute
          price index methodology.
        * MW quantities (load, wind, solar) -> mean, never sum, because
          MW is a power level, not an energy-over-interval quantity.
    - Everything is tagged with post_15min_mtu so the regime change can
      be tested as a structural break later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import pandas as pd

logger = logging.getLogger("power_forecast.clean")


@dataclass
class CleaningReport:
    dataset_name: str
    raw_row_count: int
    clean_row_count: int
    missing_count: int
    duplicate_count: int


def local_delivery_date_to_utc(
    delivery_date: str,
    local_tz: str = "Europe/Berlin",
) -> pd.Timestamp:
    """Convert midnight at the start of a local market delivery date to
    UTC. Market-design cutovers (like the 2025-10-01 move to 15-minute
    MTUs) are defined by *delivery day*, not UTC calendar midnight --
    for DE-LU, 2025-10-01 00:00 local is 2025-09-30T22:00:00Z (CEST,
    UTC+2), not 2025-10-01T00:00:00Z. Getting this wrong misclassifies
    the first two hours of the new delivery day as still belonging to
    the old regime. All cutover-comparison logic in this module and in
    entsoe_client.py must go through this helper rather than each
    re-implementing its own (previously inconsistent) interpretation.
    """
    return pd.Timestamp(delivery_date).tz_localize(local_tz).tz_convert("UTC")


def _ensure_utc(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    return df


def dedupe_timestamps(
    df: pd.DataFrame,
    ts_col: str = "timestamp_utc",
    max_allowed_fraction: float = 0.01,
) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate timestamps, keeping the first. Returns
    (deduped_df, n_duplicates_removed).

    This is a last-resort safety net, NOT the place to resolve genuine
    multi-product data (e.g. ENTSO-E publishing both a PT60M and PT15M
    DE-LU price product under the same query -- see
    entsoe_client.select_price_resolution, which must run first and
    should leave zero collisions here). If more than
    max_allowed_fraction of rows are duplicated, that's a sign an
    upstream selection step is missing or broken, not a few stray
    repeats worth silently dropping -- raise instead of masking it.
    """
    before = len(df)
    if before > 0:
        dupe_fraction = df.duplicated(subset=[ts_col]).sum() / before
        if dupe_fraction > max_allowed_fraction:
            raise ValueError(
                f"{dupe_fraction:.1%} of rows share a duplicate '{ts_col}' -- "
                f"this is too high to be incidental and likely means an upstream "
                f"multi-product/multi-resolution selection step is missing "
                f"(e.g. entsoe_client.select_price_resolution). Investigate before "
                f"deduping, don't silently drop this volume of rows."
            )
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="first")
    n_dupes = before - len(df)
    if n_dupes:
        logger.warning("Removed %d duplicate timestamps", n_dupes)
    return df, n_dupes


def aggregate_to_hourly(
    df: pd.DataFrame,
    value_cols: List[str],
    ts_col: str = "timestamp_utc",
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate quarter-hourly rows to hourly (mean) unconditionally,
    across the entire date range.

    Use this for load and wind/solar generation FORECASTS: these are
    TSO reporting data published at 15-minute granularity for the whole
    2019-2026 history, independent of the day-ahead PRICE product's own
    market-design regime (discovered empirically -- raw load/wind rows
    came back at ~265k for the full period, matching quarter-hourly
    throughout, not the ~88k hourly-then-quarter-hourly pattern price
    has). Do NOT use aggregate_quarter_hour_to_hourly (the
    cutover-conditional version) for these; that function is specific
    to price, where pre-cutover data genuinely is hourly-only because
    of the auction structure itself, not a reporting-granularity choice.
    """
    df = _ensure_utc(df, ts_col)
    if df.empty:
        return df

    indexed = df.set_index(ts_col)
    hourly = indexed[value_cols].resample("1h", label="left", closed="left").agg(agg)
    hourly = hourly.reset_index()

    other_cols = [c for c in indexed.columns if c not in value_cols]
    if other_cols:
        others = indexed[other_cols].resample("1h", label="left", closed="left").first().reset_index()
        hourly = hourly.merge(others, on=ts_col, how="left")

    return hourly.sort_values(ts_col).reset_index(drop=True)


def aggregate_quarter_hour_to_hourly(
    df: pd.DataFrame,
    value_cols: List[str],
    ts_col: str = "timestamp_utc",
    market_design_cutover: str = "2025-10-01",
    agg: str = "mean",
) -> pd.DataFrame:
    """Split a UTC-indexed dataframe at the 15-minute-MTU cutover date and
    aggregate only the post-cutover quarter-hourly rows into hourly rows
    (mean, per spec section 4). Pre-cutover rows are assumed already
    hourly and pass through unchanged.

    This is specific to the day-ahead PRICE series, where the auction
    product itself was hourly-only before the cutover (a genuine
    market-structure fact) and quarter-hourly after. It is NOT correct
    for load/wind/solar forecasts, which are quarter-hourly throughout
    the whole history -- use aggregate_to_hourly() for those instead.
    """
    df = _ensure_utc(df, ts_col)
    cutover = local_delivery_date_to_utc(market_design_cutover)

    pre = df[df[ts_col] < cutover].copy()
    post = df[df[ts_col] >= cutover].copy()

    if not post.empty:
        post = post.set_index(ts_col)
        hourly_post = post[value_cols].resample("1h", label="left", closed="left").agg(agg)
        hourly_post = hourly_post.reset_index()
        # carry through any non-value columns (e.g. psr_type) via first()
        other_cols = [c for c in post.columns if c not in value_cols]
        if other_cols:
            others = post[other_cols].resample("1h", label="left", closed="left").first().reset_index()
            hourly_post = hourly_post.merge(others, on=ts_col, how="left")
        post = hourly_post

    out = pd.concat([pre, post], ignore_index=True).sort_values(ts_col).reset_index(drop=True)
    return out


def add_market_design_flag(
    df: pd.DataFrame,
    ts_col: str = "timestamp_utc",
    cutover: str = "2025-10-01",
) -> pd.DataFrame:
    df = _ensure_utc(df, ts_col)
    cutover_ts = local_delivery_date_to_utc(cutover)
    df["post_15min_mtu"] = (df[ts_col] >= cutover_ts).astype(int)
    return df


def add_local_time_columns(
    df: pd.DataFrame,
    ts_col: str = "timestamp_utc",
    local_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Derive local-time calendar features from UTC via tz_convert.
    This is DST-safe: pandas resolves the IANA rule for Europe/Berlin at
    each specific UTC instant, so ambiguous/nonexistent local times
    around DST transitions are handled correctly rather than assumed.
    """
    df = _ensure_utc(df, ts_col)
    local = df[ts_col].dt.tz_convert(local_tz)
    df["delivery_timestamp_local"] = local
    df["delivery_date"] = local.dt.date
    df["hour_local"] = local.dt.hour
    df["dow_local"] = local.dt.dayofweek  # Monday=0
    df["month_local"] = local.dt.month
    df["weekend"] = (df["dow_local"] >= 5).astype(int)
    return df


def merge_sources(
    price_df: pd.DataFrame,
    load_df: pd.DataFrame,
    wind_solar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Outer-merge the three ENTSO-E sources on timestamp_utc. Outer join
    on purpose: a missing fundamental for an hour should surface as NaN
    (and get reported), not silently drop the price row.
    """
    for d in (price_df, load_df, wind_solar_df):
        if "timestamp_utc" in d.columns:
            d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)

    merged = price_df.merge(load_df, on="timestamp_utc", how="outer")
    merged = merged.merge(wind_solar_df, on="timestamp_utc", how="outer")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    return merged


def validate_hourly_coverage(
    df: pd.DataFrame,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    ts_col: str = "timestamp_utc",
) -> List[pd.Timestamp]:
    """Detect UTC hours that are missing from ALL sources at once.

    The outer merge in merge_sources() catches a hour that's missing
    from one or two sources (it just shows up as NaN in those columns).
    It cannot catch an hour that's missing from every source, because
    then there's simply no row for it at all. This checks the merged
    dataframe's timestamps against the full expected UTC hourly grid
    and returns whatever's absent, so it can be logged rather than
    silently passing through as if the dataset were complete.
    """
    expected = pd.date_range(
        start=expected_start, end=expected_end, freq="1h", inclusive="left", tz="UTC",
    )
    observed = pd.DatetimeIndex(pd.to_datetime(df[ts_col], utc=True))
    missing = expected.difference(observed)
    return list(missing)


def clip_to_range(
    df: pd.DataFrame,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    ts_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """Enforce the declared sample boundary explicitly: keep only rows
    with start_utc <= timestamp_utc < end_utc.

    ENTSO-E's API does not clip its response to exactly the requested
    window -- it returns whatever local delivery-day periods overlap
    the query, which can extend past either boundary. Without this
    step, "0 expected hours missing" (validate_hourly_coverage) doesn't
    imply "the row count is correct" -- it only proves no gaps, not
    that the range matches what was declared. Call this as the last
    step before freezing a dataset.
    """
    df = _ensure_utc(df, ts_col)
    mask = (df[ts_col] >= start_utc) & (df[ts_col] < end_utc)
    dropped = int((~mask).sum())
    if dropped:
        logger.info(
            "Clipped %d row(s) outside the declared range [%s, %s)", dropped, start_utc, end_utc
        )
    return df[mask].reset_index(drop=True)


def build_clean_dataset(
    price_df: pd.DataFrame,
    load_df: pd.DataFrame,
    wind_solar_df: pd.DataFrame,
    local_tz: str = "Europe/Berlin",
    market_design_cutover: str = "2025-10-01",
    start_utc: pd.Timestamp = None,
    end_utc: pd.Timestamp = None,
) -> tuple[pd.DataFrame, List[CleaningReport]]:
    """End-to-end clean.py entry point: dedupe -> aggregate to hourly
    (post-cutover only for price; always for load/wind/solar) -> merge
    -> add calendar/regime flags -> clip to [start_utc, end_utc) if
    given. Returns (clean_df, [CleaningReport, ...]) for the data
    contract logging required by spec section 22.

    Pass start_utc/end_utc (e.g. from entsoe_client.get_default_date_range)
    to enforce the declared sample boundary explicitly -- without this,
    ENTSO-E's tendency to return whole overlapping delivery-day periods
    can leave rows outside the range implied by config.yaml's
    start_date/end_date.
    """
    reports = []

    # Capture true raw counts BEFORE dedupe, so the report's raw_row_count
    # actually means "what we received", not "what survived cleaning".
    price_raw_n, load_raw_n, ws_raw_n = len(price_df), len(load_df), len(wind_solar_df)

    price_df, price_dupes = dedupe_timestamps(price_df)
    load_df, load_dupes = dedupe_timestamps(load_df)
    ws_df, ws_dupes = dedupe_timestamps(wind_solar_df)

    price_df = aggregate_quarter_hour_to_hourly(
        price_df, value_cols=["price_eur_mwh"], market_design_cutover=market_design_cutover
    )
    # Load and wind/solar forecasts are quarter-hourly TSO reporting data
    # for the whole 2019-2026 history, unrelated to the price product's
    # own market-design regime -- always aggregate, not cutover-gated.
    load_df = aggregate_to_hourly(load_df, value_cols=["load_forecast_mw"])
    ws_value_cols = [c for c in ws_df.columns if c.endswith("_mw")]
    ws_df = aggregate_to_hourly(ws_df, value_cols=ws_value_cols)

    merged = merge_sources(price_df, load_df, ws_df)
    merged = add_market_design_flag(merged, cutover=market_design_cutover)
    merged = add_local_time_columns(merged, local_tz=local_tz)

    if start_utc is not None and end_utc is not None:
        merged = clip_to_range(merged, start_utc, end_utc)

    for name, raw_n, dupes, col in [
        ("day_ahead_price", price_raw_n, price_dupes, "price_eur_mwh"),
        ("load_forecast", load_raw_n, load_dupes, "load_forecast_mw"),
        ("wind_solar_forecast", ws_raw_n, ws_dupes, "wind_onshore_forecast_mw"),
    ]:
        missing = int(merged[col].isna().sum()) if col in merged.columns else None
        reports.append(
            CleaningReport(
                dataset_name=name,
                raw_row_count=raw_n,
                clean_row_count=len(merged),
                missing_count=missing,
                duplicate_count=dupes,
            )
        )

    return merged, reports
