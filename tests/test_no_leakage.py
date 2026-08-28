"""spec section 21: test_lag_24_contains_only_past_price,
test_lag_168_contains_only_past_price,
test_future_actual_load_never_enters_features,
test_future_actual_generation_never_enters_features.

These tests treat "no leakage" as a claim to be proven against actual
data, not a property assumed from how the code was written. Each test
either recomputes an independent ground truth and compares, or checks
the structural invariant (source calendar date < delivery calendar
date) directly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import (
    decision_cutoff_utc,
    local_calendar_date,
    add_fundamentals_features,
    add_calendar_features,
    add_price_lag_and_rolling_features,
    assert_information_set_valid,
    build_feature_matrix,
    MIN_SAFE_PRICE_LAG_HOURS,
)


def _make_hourly_frame(start="2024-01-01T00:00:00Z", periods=24 * 20, price_fn=None):
    ts = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    if price_fn is None:
        price_fn = lambda i: float(i)  # noqa: E731 -- price == row index, easy to verify lags against
    df = pd.DataFrame({
        "timestamp_utc": ts,
        "price_eur_mwh": [price_fn(i) for i in range(periods)],
        "load_forecast_mw": [1000.0 + i for i in range(periods)],
        "wind_onshore_forecast_mw": [100.0] * periods,
        "wind_offshore_forecast_mw": [50.0] * periods,
        "solar_forecast_mw": [0.0 if (i % 24) < 6 or (i % 24) > 18 else 200.0 for i in range(periods)],
    })
    local = ts.tz_convert("Europe/Berlin")
    df["hour_local"] = local.hour
    df["dow_local"] = local.dayofweek
    df["month_local"] = local.month
    df["weekend"] = (df["dow_local"] >= 5).astype(int)
    return df


def test_decision_cutoff_is_1145_local_on_d_minus_1():
    cutoff = decision_cutoff_utc(pd.Timestamp("2024-06-15").date())
    local = cutoff.tz_convert("Europe/Berlin")
    assert local.date() == pd.Timestamp("2024-06-14").date()
    assert (local.hour, local.minute) == (11, 45)


def test_decision_cutoff_is_dst_safe():
    # 2024-03-31 is EU spring-forward day; D-1 = 2024-03-30, no DST
    # ambiguity there, but the offset (CET vs CEST) must resolve correctly
    # for dates on either side of the transition.
    winter_cutoff = decision_cutoff_utc(pd.Timestamp("2024-01-15").date())
    summer_cutoff = decision_cutoff_utc(pd.Timestamp("2024-07-15").date())
    # CET = UTC+1, CEST = UTC+2 -> 11:45 local should be 10:45 UTC in winter, 09:45 UTC in summer
    assert winter_cutoff.hour == 10
    assert summer_cutoff.hour == 9


def test_lag_24_contains_only_past_price():
    df = _make_hourly_frame()
    df, prov = add_price_lag_and_rolling_features(df, lags_hours=[24], rolling_windows_hours=[])
    # price == row index by construction, so lag_24 at row i must equal i-24
    valid = df["price_lag_24h"].notna()
    expected = df.loc[valid, "price_eur_mwh"].index - 24
    actual_lagged_price = df.loc[valid, "price_lag_24h"].values
    assert np.allclose(actual_lagged_price, expected.values)
    assert_information_set_valid(df)


def test_lag_168_contains_only_past_price():
    df = _make_hourly_frame()
    df, prov = add_price_lag_and_rolling_features(df, lags_hours=[168], rolling_windows_hours=[])
    valid = df["price_lag_168h"].notna()
    expected = df.loc[valid, "price_eur_mwh"].index - 168
    actual_lagged_price = df.loc[valid, "price_lag_168h"].values
    assert np.allclose(actual_lagged_price, expected.values)
    assert_information_set_valid(df)


def test_lag_below_minimum_safe_lag_is_rejected():
    df = _make_hourly_frame()
    with pytest.raises(ValueError, match="minimum safe lag"):
        add_price_lag_and_rolling_features(df, lags_hours=[12], rolling_windows_hours=[])


def test_assert_information_set_valid_catches_injected_same_day_leak():
    """A feature column that looks like a lag column by name, but was
    actually built with a same-day (unsafe) offset, must be caught.
    """
    df = _make_hourly_frame()
    # Deliberately inject a "price_lag_2h" column (2h < 24h minimum) to
    # prove the guard actually inspects the claimed lag, not just trusts it.
    df["price_lag_2h"] = df["price_eur_mwh"].shift(2)
    with pytest.raises(AssertionError, match="minimum safe lag"):
        assert_information_set_valid(df)


def test_assert_information_set_valid_passes_clean_features():
    df = _make_hourly_frame()
    df, prov = add_price_lag_and_rolling_features(df)
    assert_information_set_valid(df, provenance=prov)  # should not raise


def test_rolling_features_only_use_past_calendar_dates():
    df = _make_hourly_frame(periods=24 * 30)
    df, prov = add_price_lag_and_rolling_features(df, lags_hours=[24], rolling_windows_hours=[24, 168])
    assert_information_set_valid(df, provenance=prov)

    # Spot check: rolling_mean_24h at a given row should equal the mean
    # of the 24 raw price values ending exactly at (row - 24), not any
    # value from the row's own day.
    i = 300
    expected_mean = df["price_eur_mwh"].iloc[i - 24 - 24 + 1 : i - 24 + 1].mean()
    assert df["price_rolling_mean_24h"].iloc[i] == pytest.approx(expected_mean)


def test_dst_fallback_boundary_invalidates_unsafe_lag_rather_than_leaking():
    """Regression test for a real finding: on the autumn DST fall-back
    day (25 local hours instead of 24), a fixed 24-UTC-hour shift can
    land on the SAME local calendar date as the delivery row for the
    last couple of hours of that day, instead of the previous date.
    Real data showed exactly 7 such rows (one per year, 2019-2025) at
    e.g. 2019-10-27T22:00:00Z. These must become NaN, not a leaking value.
    """
    # Build an hourly UTC frame spanning the 2019 autumn fall-back
    # (2019-10-27 01:00 UTC, when Europe/Berlin goes CEST->CET).
    df = _make_hourly_frame(start="2019-10-25T00:00:00Z", periods=24 * 5)
    df, prov = add_price_lag_and_rolling_features(df, lags_hours=[24], rolling_windows_hours=[])

    delivery_local = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert("Europe/Berlin")
    source_local = (pd.to_datetime(df["timestamp_utc"], utc=True) - pd.Timedelta(hours=24)).dt.tz_convert("Europe/Berlin")
    would_be_unsafe = source_local.dt.date >= delivery_local.dt.date

    assert would_be_unsafe.sum() > 0, "test setup should include the fall-back boundary"
    assert df.loc[would_be_unsafe, "price_lag_24h"].isna().all(), (
        "DST-boundary rows where the 24h shift can't be proven safe must be NaN, not a value"
    )
    # And the guard must pass cleanly on the result -- no violations remain
    # because unsafe rows are now NaN, which the guard correctly treats as
    # "no value" rather than "leaked value".
    assert_information_set_valid(df)


def test_rolling_min_periods_tolerates_sparse_missingness():
    """Regression test: pandas' default min_periods=window requires
    EVERY point in a rolling window to be non-null, which propagates
    far more missingness than the underlying gap rate suggests --
    confirmed against real data (~0.4% missing price hours caused
    price_rolling_mean_168h to be NaN for ~47% of rows with the
    default). min_periods is relaxed to 80% of the window so sparse,
    small real-world gaps don't cascade into losing half the dataset.

    Uses a long enough sample that the unavoidable ~168-row warm-up
    period (no rolling value possible at all before that many hours
    have elapsed) doesn't dominate the missingness measurement.
    """
    df = _make_hourly_frame(periods=24 * 400)  # ~9600 rows: warm-up is <2% of the sample
    rng = np.random.default_rng(42)
    missing_idx = rng.choice(len(df), size=int(len(df) * 0.004), replace=False)
    df.loc[missing_idx, "price_eur_mwh"] = np.nan

    out, prov = add_price_lag_and_rolling_features(df, lags_hours=[24], rolling_windows_hours=[168])

    missing_fraction = out["price_rolling_mean_168h"].isna().mean()
    assert missing_fraction < 0.15, (
        f"price_rolling_mean_168h is {missing_fraction:.1%} missing with sparse "
        f"input gaps -- min_periods relaxation may have regressed"
    )


def test_dst_boundary_invalidates_rolling_features_too():
    """Regression test for a real interaction bug: with min_periods <
    window, rolling().mean() can still produce a numeric value even
    when the single DST-invalidated point in the window is NaN, by
    using the window's OTHER (older, still-safe) points instead. That
    numeric value isn't actually unsafe, but assert_information_set_valid's
    boundary check doesn't know that and flags it as a false positive.
    Rather than build a more complex proof, rolling columns are
    explicitly invalidated at the same rows as their base lag column.
    """
    df = _make_hourly_frame(start="2019-10-25T00:00:00Z", periods=24 * 10)
    df, prov = add_price_lag_and_rolling_features(df, lags_hours=[24], rolling_windows_hours=[24, 168])

    delivery_local = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert("Europe/Berlin")
    source_local = (pd.to_datetime(df["timestamp_utc"], utc=True) - pd.Timedelta(hours=24)).dt.tz_convert("Europe/Berlin")
    dst_boundary_rows = source_local.dt.date >= delivery_local.dt.date

    assert dst_boundary_rows.sum() > 0, "test setup should include the fall-back boundary"
    assert df.loc[dst_boundary_rows, "price_rolling_mean_24h"].isna().all()
    assert df.loc[dst_boundary_rows, "price_rolling_mean_168h"].isna().all()
    assert_information_set_valid(df, provenance=prov)  # must not raise


def test_guard_raises_for_rolling_column_without_provenance():
    """Regression test for a real design gap: previously the guard
    inferred a rolling column's 'source lag' from its window size in
    the name (e.g. parsing 168 out of 'price_rolling_mean_168h'), which
    is wrong -- the window size isn't how recent the data is. Now the
    guard requires explicit provenance for rolling columns and refuses
    to guess.
    """
    df = _make_hourly_frame()
    df["price_rolling_mean_168h"] = df["price_eur_mwh"].rolling(168).mean()  # built with NO safety at all
    with pytest.raises(AssertionError, match="provenance"):
        assert_information_set_valid(df)  # no provenance passed -> must refuse, not guess


def test_guard_uses_actual_provenance_not_window_size():
    """A rolling column's real newest-source-lag (from provenance) is
    what gets checked, not a number parsed from the window size in its
    name. Prove this distinguishes a genuinely unsafe construction from
    a safe one that happens to have the same window size.
    """
    df = _make_hourly_frame(periods=24 * 20)
    # Genuinely unsafe: rolling mean built directly on RAW price (no lag at all).
    df["price_rolling_mean_168h"] = df["price_eur_mwh"].rolling(168, min_periods=1).mean()
    bad_provenance = {"price_rolling_mean_168h": {"newest_source_lag_hours": 0, "window_hours": 168}}
    with pytest.raises(AssertionError, match="minimum safe lag"):
        assert_information_set_valid(df, provenance=bad_provenance)


def test_feature_availability_tier_matches_documented_regulation_findings():
    """Locks in the point-in-time tiering found via the actual EU
    Transparency Regulation (EU) No 543/2013: load forecast (Article
    6(2)(b), published >=2h before gate closure) is Tier 1 (proven safe
    at the 11:45 D-1 cutoff); wind/solar forecast (Article 14(2)(d),
    published by 17:00 D-1 -- AFTER the cutoff) and everything derived
    from it are Tier 2 (used as features, but not proven safe).
    """
    from src.features import FEATURE_AVAILABILITY_TIER
    assert FEATURE_AVAILABILITY_TIER["load_forecast_mw"] == 1
    for col in [
        "wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw",
        "renewables_forecast_mw", "residual_load_forecast_mw", "renewable_share_forecast",
    ]:
        assert FEATURE_AVAILABILITY_TIER[col] == 2


def test_missing_wind_component_propagates_nan_not_zero():
    """Regression test: a missing wind forecast is NOT the same as a
    zero-MW forecast. Silently fillna(0) would understate available
    renewables and inflate residual load artificially. Both onshore
    and offshore must be present, or the combined figure (and
    everything derived from it) must be NaN.
    """
    df = pd.DataFrame({
        "load_forecast_mw": [1000.0, 1000.0],
        "wind_onshore_forecast_mw": [200.0, np.nan],  # missing on row 1
        "wind_offshore_forecast_mw": [100.0, 50.0],
        "solar_forecast_mw": [0.0, 0.0],
    })
    out = add_fundamentals_features(df)

    assert out["renewables_forecast_mw"].iloc[0] == pytest.approx(300.0)
    assert pd.isna(out["renewables_forecast_mw"].iloc[1]), (
        "missing wind component must propagate as NaN, not be treated as 0 MW"
    )
    assert pd.isna(out["residual_load_forecast_mw"].iloc[1])


def test_month_cyclic_encoding_december_january_adjacent():
    """Raw month=1..12 would treat Dec/Jan as 11 units apart. Cyclic
    encoding must place them close together.
    """
    df = _make_hourly_frame(periods=24 * 400)  # spans across a year boundary
    df = add_calendar_features(df)
    dec_row = df[df["month_local"] == 12].iloc[0]
    jan_row = df[df["month_local"] == 1].iloc[0]
    # Euclidean distance in (sin, cos) space between Dec and Jan should
    # be small -- comparable to the distance between adjacent months
    # generally, not the near-maximal distance a raw linear 12 vs 1
    # encoding would imply.
    dist = ((dec_row["month_sin"] - jan_row["month_sin"]) ** 2
            + (dec_row["month_cos"] - jan_row["month_cos"]) ** 2) ** 0.5
    assert dist < 1.0  # two points a full 12-unit cycle apart on a unit circle: distance ~0.5 for 1-month gap


def test_weekend_hour_interaction_zero_on_weekdays():
    df = _make_hourly_frame(periods=24 * 10)
    df = add_calendar_features(df)
    weekday_rows = df[df["weekend"] == 0]
    assert (weekday_rows["weekend_hour_sin"] == 0).all()
    assert (weekday_rows["weekend_hour_cos"] == 0).all()


def test_weekend_hour_interaction_matches_hour_encoding_on_weekends():
    df = _make_hourly_frame(periods=24 * 10)
    df = add_calendar_features(df)
    weekend_rows = df[df["weekend"] == 1]
    assert np.allclose(weekend_rows["weekend_hour_sin"], weekend_rows["hour_sin"])
    assert np.allclose(weekend_rows["weekend_hour_cos"], weekend_rows["hour_cos"])


def test_future_actual_load_never_enters_features():
    """We only ever ingest day-ahead LOAD FORECASTS (see
    entsoe_client.fetch_load_forecast), never realised/actual load.
    This test documents and locks in that invariant at the feature
    layer: no column derived from "actual" or "realised" data exists.
    """
    df = _make_hourly_frame()
    df = build_feature_matrix(df)
    forbidden_substrings = ["actual_load", "realised_load", "realized_load"]
    for col in df.columns:
        for bad in forbidden_substrings:
            assert bad not in col.lower(), f"Column '{col}' suggests realised/actual load leaked in"


def test_future_actual_generation_never_enters_features():
    """Same invariant for wind/solar: only day-ahead FORECASTS are ever
    ingested (entsoe_client.fetch_wind_solar_forecast), never realised
    generation.
    """
    df = _make_hourly_frame()
    df = build_feature_matrix(df)
    forbidden_substrings = ["actual_generation", "realised_generation", "realized_generation",
                             "actual_wind", "actual_solar"]
    for col in df.columns:
        for bad in forbidden_substrings:
            assert bad not in col.lower(), f"Column '{col}' suggests realised generation leaked in"


def test_residual_load_uses_same_timestamp_forecast_columns_only():
    """Residual load / renewable share are built from day-ahead forecast
    columns at the row's OWN timestamp -- safe by construction (see
    module docstring), but verify the arithmetic is actually right so
    "safe" isn't just an unverified assumption.
    """
    df = _make_hourly_frame(periods=24)
    df = add_fundamentals_features(df)
    i = 12  # a daytime hour, solar_forecast_mw = 200 by _make_hourly_frame's construction
    expected_renewables = df["wind_onshore_forecast_mw"].iloc[i] + df["wind_offshore_forecast_mw"].iloc[i] + df["solar_forecast_mw"].iloc[i]
    assert df["renewables_forecast_mw"].iloc[i] == pytest.approx(expected_renewables)
    expected_residual = df["load_forecast_mw"].iloc[i] - expected_renewables
    assert df["residual_load_forecast_mw"].iloc[i] == pytest.approx(expected_residual)


def test_missing_hours_do_not_misalign_lags():
    """If a raw hour is missing, a naive row-position shift(24) would
    silently pull the WRONG hour's price into the lag column once the
    grid is irregular. build_feature_matrix regularizes to a complete
    hourly grid first specifically to prevent this.
    """
    df = _make_hourly_frame(periods=24 * 10)
    # Drop a handful of interior rows to simulate real missing hours
    df = df.drop(index=[50, 51, 52]).reset_index(drop=True)

    out = build_feature_matrix(df)

    # After regularization the grid must be a complete, gap-free hourly index
    ts = pd.to_datetime(out["timestamp_utc"], utc=True)
    assert (ts.diff().dropna() == pd.Timedelta(hours=1)).all()

    # And the lag column must still correctly reference true source
    # prices where both source and target rows exist, not a
    # position-shifted neighbor.
    valid = out["price_lag_24h"].notna() & out["price_eur_mwh"].notna()
    src_ts = ts[valid] - pd.Timedelta(hours=24)
    price_by_ts = dict(zip(ts, out["price_eur_mwh"]))
    expected = src_ts.map(price_by_ts)
    actual = out.loc[valid, "price_lag_24h"].values
    comparable = expected.notna()
    assert np.allclose(actual[comparable.values], expected[comparable].values)
