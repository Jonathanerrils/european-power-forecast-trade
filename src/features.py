"""Feature construction with an explicit point-in-time information set
(spec sections 2, 6, 7, 8).

The simulated analyst sits at D-1, 11:45 Europe/Berlin, before the
day-ahead auction for delivery day D clears. Different leakage
arguments apply depending on the data type -- and they've been
verified against the actual EU Transparency Regulation
((EU) No 543/2013), not just assumed from "day-ahead" naming:

    - LOAD FORECAST at (D, h): Tier 1. Article 6(2)(b) requires
      publication no later than TWO HOURS BEFORE day-ahead gate closure
      (~noon D-1), which is before our 11:45 cutoff -- so the
      REGULATORY publication deadline is compatible with the cutoff.
      This is not the same claim as "individually vintage-verified":
      exact historical revision vintage still cannot be reconstructed
      for any Tier, load included (see the standing limitation in
      README.md). "Compatible with the cutoff" is the precise claim;
      "provably safe" overstates it.

    - WIND/SOLAR FORECAST at (D, h): Tier 2, not compatible with the
      cutoff even at the level of the regulatory deadline.
      Article 14(2)(d) only requires publication by 17:00 D-1 -- nearly
      5 hours AFTER our decision cutoff -- with updates continuing
      through intraday trading. We cannot reconstruct historical
      publication timestamps to prove any specific historical value
      predates 11:45. This is a real, verified point-in-time gap, not
      a hypothetical one; see FEATURE_AVAILABILITY_TIER below and the
      README's point-in-time section for the full account. Wind/solar
      and everything derived from them (residual load, renewable
      share) are therefore Tier 2: used as features, but their
      point-in-time validity is an open, documented question rather
      than a proven claim.

    - PRICE at (D', h) for D' < D: safe to use once D' is a calendar
      date strictly before D, REGARDLESS of which hour h within D' it's
      tagged to. This is because day-ahead price for delivery date D'
      is entirely fixed and published the moment D''s own auction
      clears -- which happens on D'-1, i.e. before D' has even started.
      So by the time D-1 11:45 arrives, every hour of every day up to
      and including D-1 already has a known, published price. A lag of
      >=24 hours from any hour on day D always lands on a calendar date
      <= D-1, so lag/rolling price features built with >=24h lag are
      safe by construction -- checked structurally in
      assert_information_set_valid() using explicit construction
      provenance, not inferred from the column name (a rolling column
      named "..._168h" is NOT built from data 168h back -- it's built
      from a 24h-lagged series over a 168h window, so its newest
      underlying data point is ~24h back, not 168h; the guard checks
      the real value, not the name).

    - PRICE at (D, h) itself: never a feature. It's the target.

Known limitation (documented, not hidden -- spec section 8): even for
Tier 1 (load), we don't have historical forecast *revision* vintages,
only ENTSO-E's current publication. Where a vintage can't be
reconstructed exactly, we use the published forecast as a forecast-time
proxy.
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Dict, Iterable, List

import numpy as np


# Point-in-time availability tiers for fundamentals, per the EU
# Transparency Regulation research documented in the module docstring.
# Tier 1: publication deadline is provably before the 11:45 D-1 cutoff.
# Tier 2: publication deadline is AFTER the cutoff (or derived from a
# Tier 2 column) -- used as a feature, but not proven point-in-time safe.
FEATURE_AVAILABILITY_TIER: Dict[str, int] = {
    "load_forecast_mw": 1,
    "wind_onshore_forecast_mw": 2,
    "wind_offshore_forecast_mw": 2,
    "solar_forecast_mw": 2,
    "renewables_forecast_mw": 2,
    "residual_load_forecast_mw": 2,
    "renewable_share_forecast": 2,
}
import pandas as pd

logger = logging.getLogger("power_forecast.features")

DEFAULT_PRICE_LAGS_HOURS = (24, 48, 168)
DEFAULT_ROLLING_WINDOWS_HOURS = (24, 168)
DECISION_CUTOFF_LOCAL_TIME = "11:45"
DECISION_LOCAL_TZ = "Europe/Berlin"
MIN_SAFE_PRICE_LAG_HOURS = 24  # anything shorter cannot be proven safe by the calendar-date argument


# ---------------------------------------------------------------------
# Decision-time helpers
# ---------------------------------------------------------------------
def decision_cutoff_utc(delivery_date: date, local_tz: str = DECISION_LOCAL_TZ) -> pd.Timestamp:
    """The simulated decision cutoff for delivery day D: 11:45 local
    time on D-1, expressed in UTC. DST-safe via tz_localize (handles
    the local UTC offset correctly for any date, not a fixed +1/+2).
    """
    d_minus_1 = pd.Timestamp(delivery_date) - pd.Timedelta(days=1)
    local_dt = pd.Timestamp(
        f"{d_minus_1.date()} {DECISION_CUTOFF_LOCAL_TIME}"
    ).tz_localize(local_tz)
    return local_dt.tz_convert("UTC")


def local_calendar_date(ts_utc: pd.Series, local_tz: str = DECISION_LOCAL_TZ) -> pd.Series:
    """Local (Europe/Berlin) calendar date for a UTC timestamp series."""
    ts_utc = pd.to_datetime(ts_utc, utc=True)
    return ts_utc.dt.tz_convert(local_tz).dt.date


# ---------------------------------------------------------------------
# Fundamentals (day-ahead forecasts) -- same-delivery-hour features;
# point-in-time validity depends on availability tier (see
# FEATURE_AVAILABILITY_TIER and the module docstring: load is Tier 1,
# wind/solar and everything derived from them are Tier 2)
# ---------------------------------------------------------------------
def add_fundamentals_features(df: pd.DataFrame) -> pd.DataFrame:
    """Residual load and renewable share, per spec section 7.

    ResidualLoad_t = LoadForecast_t - (Wind_t + Solar_t)
    RenewableShare_t = (Wind_t + Solar_t) / Load_t

    Built from forecast columns for the SAME delivery timestamp as the
    row -- no temporal lag is applied to the forecasts themselves. This
    does NOT by itself establish point-in-time availability: load is
    Tier 1 (regulatory deadline compatible with the 11:45 cutoff),
    while wind/solar and everything derived from them here (residual
    load, renewable share) are Tier 2 (regulatory deadline is NOT
    compatible with the cutoff) -- see the module-level documentation
    and FEATURE_AVAILABILITY_TIER.
    """
    df = df.copy()
    # Wind: do NOT fillna(0) -- a missing forecast is not the same as a
    # zero-MW forecast, and silently treating it as zero would inflate
    # residual load artificially. Require BOTH onshore and offshore to be
    # present; propagate NaN otherwise (min_count=2 forces this).
    wind_total = df[["wind_onshore_forecast_mw", "wind_offshore_forecast_mw"]].sum(axis=1, min_count=2)
    # Solar forecast is frequently NaN overnight (ENTSO-E appears to
    # omit rather than publish an explicit 0 -- see README data-status
    # notes). Treating NaN as 0 here is a deliberate, documented choice
    # for solar specifically, not a general missing-data policy.
    solar = df["solar_forecast_mw"].fillna(0)

    df["renewables_forecast_mw"] = wind_total + solar
    df["residual_load_forecast_mw"] = df["load_forecast_mw"] - df["renewables_forecast_mw"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["renewable_share_forecast"] = np.where(
            df["load_forecast_mw"] > 0,
            df["renewables_forecast_mw"] / df["load_forecast_mw"],
            np.nan,
        )
    return df


# ---------------------------------------------------------------------
# Calendar features -- always safe, deterministic, no market info
# ---------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic calendar encoding, plus one deliberate interaction term.

    EDA (fig3_seasonality_hour_weekday.png) showed weekday and weekend
    intraday price profiles are NOT simple vertical shifts of each
    other -- the weekend midday decline is visibly deeper. A purely
    additive hour_sin + hour_cos + weekend structure can only give
    weekends a different average level, not a different intraday
    SHAPE. weekend_hour_sin/cos let a linear model (ElasticNet) capture
    that shape difference without opening the door to unbounded
    feature engineering -- these two interaction terms plus month
    cyclic encoding are the last calendar features added before the
    set is frozen (see README's pre-registered hypotheses).
    """
    df = df.copy()
    hour = df["hour_local"].astype(float)
    dow = df["dow_local"].astype(float)
    month = df["month_local"].astype(float)
    weekend = df["weekend"].astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    # Month cyclic encoding: raw month=1..12 would treat December and
    # January as eleven units apart instead of adjacent.
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    # Weekend x hour-of-day interaction: lets a linear model give
    # weekends a different intraday SHAPE, not just a different level.
    df["weekend_hour_sin"] = weekend * df["hour_sin"]
    df["weekend_hour_cos"] = weekend * df["hour_cos"]
    return df


# ---------------------------------------------------------------------
# Price lag / rolling features -- must cross a calendar-day boundary
# ---------------------------------------------------------------------
def _regularize_hourly_grid(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    """Reindex onto a complete, gap-free hourly UTC grid before any
    shift/rolling operation. Without this, a handful of genuinely
    missing hours (see README data-status: ~280 missing price hours)
    would silently misalign row-position-based lags -- e.g. a 24-row
    shift would no longer correspond to a 24-hour lag once a row is
    missing. Missing hours become explicit NaN rows instead.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()  # set_index(name) drops the column, avoiding a dupe on reset
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    df = df.reindex(full_index)
    df.index.name = ts_col
    df = df.reset_index()
    return df


def add_price_lag_and_rolling_features(
    df: pd.DataFrame,
    lags_hours: Iterable[int] = DEFAULT_PRICE_LAGS_HOURS,
    rolling_windows_hours: Iterable[int] = DEFAULT_ROLLING_WINDOWS_HOURS,
    ts_col: str = "timestamp_utc",
    price_col: str = "price_eur_mwh",
    local_tz: str = DECISION_LOCAL_TZ,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Add price_lag_{n}h and price_rolling_{mean,vol}_{n}h columns.
    Returns (df, provenance) -- provenance maps each new column to its
    actual newest_source_lag_hours (and window_hours for rolling
    columns), so assert_information_set_valid can check what was
    ACTUALLY built rather than infer it from the column name (see that
    function's docstring for why the name alone is insufficient for
    rolling columns).

    Every lag here must be >= MIN_SAFE_PRICE_LAG_HOURS (24h) -- shorter
    lags cannot be proven safe by the calendar-date argument in the
    module docstring.

    A fixed N-hour UTC shift is USUALLY but not ALWAYS enough to cross
    into an earlier local calendar date: on the autumn DST fall-back
    day, the local day has 25 hours, so a 24h shift can land on the
    SAME local calendar date for the last couple of hours of that day
    (confirmed against real data: 7 rows total, one per year, exactly
    on the fall-back transition). Rather than loosen the safety check,
    we explicitly invalidate (set NaN) any row where the shift-based
    lag can't be proven safe, and build rolling stats from that
    already-invalidated series so unsafe points never enter a rolling
    window either. This affects a handful of rows per year and is the
    conservative, honest choice over silently using an unproven value.
    """
    for lag in lags_hours:
        if lag < MIN_SAFE_PRICE_LAG_HOURS:
            raise ValueError(
                f"Price lag of {lag}h is below the minimum safe lag of "
                f"{MIN_SAFE_PRICE_LAG_HOURS}h and cannot be proven leakage-free "
                f"by the calendar-date argument this module relies on."
            )

    df = df.copy()
    if not df[ts_col].is_monotonic_increasing:
        df = df.sort_values(ts_col).reset_index(drop=True)

    price = df[price_col]
    delivery_local_date = local_calendar_date(df[ts_col], local_tz)
    provenance: Dict[str, Dict[str, int]] = {}

    def _unsafe_mask(lag_hours: int) -> pd.Series:
        source_ts = pd.to_datetime(df[ts_col], utc=True) - pd.Timedelta(hours=lag_hours)
        source_local_date = local_calendar_date(source_ts, local_tz)
        return source_local_date >= delivery_local_date

    def _safe_shift(series: pd.Series, lag_hours: int, unsafe: pd.Series = None) -> pd.Series:
        shifted = series.shift(lag_hours)
        if unsafe is None:
            unsafe = _unsafe_mask(lag_hours)
        if unsafe.any():
            logger.info(
                "%d row(s) at %dh lag fall on a DST-affected boundary where the "
                "shift can't be proven to cross a local calendar date; setting to NaN "
                "rather than using an unproven value.", int(unsafe.sum()), lag_hours,
            )
        return shifted.where(~unsafe.values)

    for lag in lags_hours:
        col = f"price_lag_{lag}h"
        df[col] = _safe_shift(price, lag)
        provenance[col] = {"newest_source_lag_hours": lag}

    base_lag = min(lags_hours) if lags_hours else MIN_SAFE_PRICE_LAG_HOURS
    base_unsafe = _unsafe_mask(base_lag)
    lagged_for_rolling = _safe_shift(price, base_lag, unsafe=base_unsafe)
    for window in rolling_windows_hours:
        # Default pandas behavior (min_periods=window) requires EVERY
        # point in the window to be non-null. With ~0.4% of price hours
        # missing scattered across the series, that default propagates
        # much further than it looks like it should: each missing hour
        # poisons up to `window` subsequent rolling results. Confirmed
        # against real data: with min_periods=window, price_rolling_mean_168h
        # was NaN for ~47% of rows. Requiring 80% of the window instead
        # is a documented tradeoff -- slightly less strict about
        # completeness, in exchange for not discarding half the dataset
        # for a one-week rolling feature over sparse, small, real gaps.
        min_periods = max(1, int(window * 0.8))
        rolling_mean = lagged_for_rolling.rolling(window, min_periods=min_periods).mean()
        rolling_vol = lagged_for_rolling.rolling(window, min_periods=min_periods).std()

        # A DST-boundary row is unsafe at the nominal lag point even if
        # min_periods tolerance lets rolling() compute a numeric value
        # from the window's OTHER (older, still-safe) points. To stay
        # consistent with assert_information_set_valid's boundary check
        # -- and to avoid depending on a more complex proof about which
        # exact window points were used -- explicitly invalidate these
        # rows here too, rather than let a technically-different but
        # harder-to-verify argument stand.
        mean_col = f"price_rolling_mean_{window}h"
        vol_col = f"price_rolling_vol_{window}h"
        df[mean_col] = rolling_mean.where(~base_unsafe.values)
        df[vol_col] = rolling_vol.where(~base_unsafe.values)
        # The rolling window's newest possible source point is base_lag
        # hours back -- NOT `window` hours back. A column named
        # "..._168h" only tells you the WINDOW SIZE, not how recent its
        # newest data is; that's exactly the gap a name-based leakage
        # check would miss. See assert_information_set_valid.
        provenance[mean_col] = {"newest_source_lag_hours": base_lag, "window_hours": window}
        provenance[vol_col] = {"newest_source_lag_hours": base_lag, "window_hours": window}

    return df, provenance


# ---------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------
def assert_information_set_valid(
    df: pd.DataFrame,
    ts_col: str = "timestamp_utc",
    price_col: str = "price_eur_mwh",
    provenance: Dict[str, Dict[str, int]] = None,
    local_tz: str = DECISION_LOCAL_TZ,
) -> None:
    """Structural leakage check, not just a description of intent.

    For every price-derived column, verify that the underlying raw
    price data it was ACTUALLY built from carries a LOCAL CALENDAR DATE
    strictly earlier than the row's own delivery date.

    provenance (from add_price_lag_and_rolling_features) is REQUIRED
    for rolling columns and strongly preferred for everything else.
    Inferring the lag from a column name (e.g. parsing "168" out of
    "price_rolling_mean_168h") is wrong for rolling columns: that
    number is the WINDOW SIZE, not how recent the newest data in it is
    -- a rolling mean over a 168h window built from a 24h-lagged series
    has its newest data point ~24h back, not 168h. Checking against the
    wrong number can pass even when the real construction changes
    (e.g. someone removes the internal shift), which is exactly the
    kind of regression this guard exists to catch. If provenance is
    omitted, this falls back to name-parsing ONLY for price_lag_*
    columns (where the name directly IS the construction) and raises
    for any price_rolling_* column, rather than silently trusting an
    unverified number.

    Raises AssertionError with a specific, actionable message on
    failure. Call this after every feature-construction step that
    touches price, before the feature matrix is used for anything.
    """
    candidate_cols = [c for c in df.columns if c.startswith("price_lag_") or c.startswith("price_rolling_")]
    if not candidate_cols:
        return

    delivery_local_date = local_calendar_date(df[ts_col], local_tz)

    for col in candidate_cols:
        if provenance is not None and col in provenance:
            lag_hours = provenance[col]["newest_source_lag_hours"]
        elif col.startswith("price_lag_"):
            digits = "".join(ch for ch in col.split("_")[-1] if ch.isdigit())
            if not digits:
                continue
            lag_hours = int(digits)
        else:
            raise AssertionError(
                f"Column '{col}' looks like a rolling price feature but no provenance "
                f"was supplied for it. Its window size is NOT a safe proxy for how "
                f"recent its underlying data is -- pass the provenance dict returned "
                f"by add_price_lag_and_rolling_features() rather than let this guard "
                f"guess from the column name."
            )

        if lag_hours < MIN_SAFE_PRICE_LAG_HOURS:
            raise AssertionError(
                f"Column '{col}' has newest_source_lag_hours={lag_hours}, below the "
                f"minimum safe lag of {MIN_SAFE_PRICE_LAG_HOURS}h. This cannot be "
                f"proven leakage-free."
            )

        source_ts = pd.to_datetime(df[ts_col], utc=True) - pd.Timedelta(hours=lag_hours)
        source_local_date = local_calendar_date(source_ts, local_tz)

        # Only check POPULATED rows -- a NaN here means the value was
        # deliberately invalidated (e.g. a DST-boundary row where the
        # shift couldn't be proven safe; see add_price_lag_and_rolling_features),
        # not a leakage violation. Checking NaN rows would either
        # false-positive on intentional gaps or mask a real bug that
        # produces NaN for an unrelated reason -- so we check what's
        # actually present, which is the only thing that can leak.
        populated = df[col].notna()
        violation = (source_local_date >= delivery_local_date) & populated
        n_violations = int(violation.sum())
        if n_violations > 0:
            bad_rows = df.loc[violation, ts_col].head(5).tolist()
            raise AssertionError(
                f"LEAKAGE: column '{col}' has {n_violations} row(s) where the "
                f"source data's local calendar date is NOT strictly before the "
                f"delivery date. Example timestamps: {bad_rows}. This means the "
                f"feature could reference information not yet public at the "
                f"D-1 {DECISION_CUTOFF_LOCAL_TIME} decision cutoff."
            )

    logger.info("Leakage guard passed: %d price-derived columns checked, 0 violations.", len(candidate_cols))


# ---------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame,
    lags_hours: Iterable[int] = DEFAULT_PRICE_LAGS_HOURS,
    rolling_windows_hours: Iterable[int] = DEFAULT_ROLLING_WINDOWS_HOURS,
) -> pd.DataFrame:
    """clean_df -> feature matrix, in the order: regularize grid ->
    fundamentals -> calendar -> price lags/rolling -> leakage guard.

    Note: fundamentals (load, wind, solar, and everything derived from
    them) are NOT all point-in-time proven -- see
    FEATURE_AVAILABILITY_TIER and the module docstring. They're
    included as features regardless, but their availability tier is an
    explicit, checkable fact about this dataset, not a hidden assumption.
    """
    df = _regularize_hourly_grid(df)
    df = add_fundamentals_features(df)
    df = add_calendar_features(df)
    df, provenance = add_price_lag_and_rolling_features(
        df, lags_hours=lags_hours, rolling_windows_hours=rolling_windows_hours
    )
    assert_information_set_valid(df, provenance=provenance)
    return df
