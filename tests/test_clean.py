import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clean import (
    aggregate_quarter_hour_to_hourly,
    aggregate_to_hourly,
    dedupe_timestamps,
    add_market_design_flag,
    local_delivery_date_to_utc,
    validate_hourly_coverage,
    build_clean_dataset,
    clip_to_range,
)


def test_hourly_price_aggregation():
    """Post-2025-10-01 quarter-hour prices should average to the hourly
    price, matching EPEX's own 60-minute price index methodology.
    """
    ts = pd.date_range("2025-10-01T00:00:00Z", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts, "price_eur_mwh": [100.0, 120.0, 80.0, 140.0]})

    out = aggregate_quarter_hour_to_hourly(df, value_cols=["price_eur_mwh"])

    assert len(out) == 1
    assert out.iloc[0]["price_eur_mwh"] == pytest.approx((100 + 120 + 80 + 140) / 4)
    assert out.iloc[0]["timestamp_utc"] == pd.Timestamp("2025-10-01T00:00:00Z")


def test_mw_aggregation_uses_mean_not_sum():
    """MW is a power level, not summed energy -- spec section 4 is explicit
    that this must never be a sum.
    """
    ts = pd.date_range("2025-10-01T00:00:00Z", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts, "load_forecast_mw": [1000.0, 1000.0, 1000.0, 1000.0]})

    out = aggregate_quarter_hour_to_hourly(df, value_cols=["load_forecast_mw"])

    assert out.iloc[0]["load_forecast_mw"] == pytest.approx(1000.0)
    assert out.iloc[0]["load_forecast_mw"] != pytest.approx(4000.0)


def test_pre_cutover_hourly_data_passes_through_unchanged():
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts, "price_eur_mwh": [50.0, 60.0, 70.0]})

    out = aggregate_quarter_hour_to_hourly(df, value_cols=["price_eur_mwh"])

    assert len(out) == 3
    assert list(out["price_eur_mwh"]) == [50.0, 60.0, 70.0]


def test_duplicate_timestamps_rejected():
    """A small, incidental duplicate rate gets cleaned up quietly."""
    n = 500
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC").tolist()
    ts.append(ts[0])  # one stray duplicate -> 1/501 ~ 0.2%, well under the threshold
    df = pd.DataFrame({"timestamp_utc": ts, "price_eur_mwh": [10.0] * n + [999.0]})

    out, n_dupes = dedupe_timestamps(df)

    assert n_dupes == 1
    assert len(out) == n
    assert out["timestamp_utc"].is_unique


def test_high_duplicate_fraction_raises_instead_of_silently_dropping():
    """Regression guard for the real dual-resolution price bug: a high
    duplicate rate must raise, not get silently deduped away, because it
    usually means an upstream multi-product selection step is missing
    (see entsoe_client.select_price_resolution).
    """
    ts = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]
    )
    df = pd.DataFrame({"timestamp_utc": ts, "price_eur_mwh": [10.0, 999.0, 20.0]})

    with pytest.raises(ValueError, match="duplicate"):
        dedupe_timestamps(df)


def test_aggregate_to_hourly_always_aggregates_regardless_of_date():
    """Load/wind/solar forecasts are quarter-hourly for the WHOLE
    2019-2026 history, unlike price which is only quarter-hourly after
    the 2025-10-01 cutover. This must aggregate everywhere, including
    dates well before the price cutover.
    """
    ts = pd.date_range("2019-01-01T00:00:00Z", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts, "load_forecast_mw": [1000.0, 1000.0, 1000.0, 1000.0]})

    out = aggregate_to_hourly(df, value_cols=["load_forecast_mw"])

    assert len(out) == 1
    assert out.iloc[0]["load_forecast_mw"] == pytest.approx(1000.0)


def test_build_clean_dataset_price_not_mostly_missing_pre_cutover():
    """Regression test for a real bug: pre-2025 load/wind data is
    quarter-hourly while price is hourly-only. If load/wind aren't
    aggregated to hourly for the WHOLE history (not just post-cutover),
    the outer merge produces a row grid finer than price's actual
    granularity, and price ends up mostly NaN. A real ingestion run
    showed 177,754 / 243,953 (73%) of price values missing because of
    exactly this.
    """
    hours = pd.date_range("2019-01-01T00:00:00Z", periods=48, freq="1h", tz="UTC")
    price_df = pd.DataFrame({"timestamp_utc": hours, "price_eur_mwh": range(len(hours))})

    quarter_hours = pd.date_range("2019-01-01T00:00:00Z", periods=48 * 4, freq="15min", tz="UTC")
    load_df = pd.DataFrame({"timestamp_utc": quarter_hours, "load_forecast_mw": [500.0] * len(quarter_hours)})
    ws_df = pd.DataFrame({
        "timestamp_utc": quarter_hours,
        "solar_forecast_mw": [0.0] * len(quarter_hours),
        "wind_onshore_forecast_mw": [100.0] * len(quarter_hours),
        "wind_offshore_forecast_mw": [50.0] * len(quarter_hours),
    })

    clean_df, reports = build_clean_dataset(price_df, load_df, ws_df)

    missing_fraction = clean_df["price_eur_mwh"].isna().mean()
    assert missing_fraction < 0.05, (
        f"price_eur_mwh is {missing_fraction:.1%} missing -- load/wind aggregation "
        f"is likely not running unconditionally across the whole date range"
    )
    assert len(clean_df) == len(hours)  # merged grid should be hourly, not quarter-hourly


def test_clip_to_range_removes_out_of_bounds_rows():
    """Regression test for a real finding: ENTSO-E does not clip its
    response to exactly the requested window -- it returns whatever
    local delivery-day periods overlap the query, which can extend
    past either boundary. A real ingestion run's frozen dataset started
    at 2018-12-31T23:00Z even though the declared config start_date was
    2019-01-01 -- one hour of unintended data leaked in from outside
    the declared sample range.
    """
    ts = pd.date_range("2018-12-31T22:00:00Z", "2019-01-02T02:00:00Z", freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts, "price_eur_mwh": range(len(ts))})

    start = pd.Timestamp("2019-01-01T00:00:00Z")
    end = pd.Timestamp("2019-01-02T00:00:00Z")
    out = clip_to_range(df, start, end)

    assert out["timestamp_utc"].min() == start
    assert out["timestamp_utc"].max() == end - pd.Timedelta(hours=1)
    assert (out["timestamp_utc"] >= start).all()
    assert (out["timestamp_utc"] < end).all()


def test_build_clean_dataset_clips_to_declared_range_when_given():
    ts = pd.date_range("2018-12-31T22:00:00Z", "2019-01-03T02:00:00Z", freq="1h", tz="UTC")
    price_df = pd.DataFrame({"timestamp_utc": ts, "price_eur_mwh": range(len(ts))})
    load_df = pd.DataFrame({"timestamp_utc": ts, "load_forecast_mw": [500.0] * len(ts)})
    ws_df = pd.DataFrame({
        "timestamp_utc": ts,
        "solar_forecast_mw": [0.0] * len(ts),
        "wind_onshore_forecast_mw": [10.0] * len(ts),
        "wind_offshore_forecast_mw": [5.0] * len(ts),
    })

    start = pd.Timestamp("2019-01-01T00:00:00Z")
    end = pd.Timestamp("2019-01-02T00:00:00Z")
    clean_df, _ = build_clean_dataset(price_df, load_df, ws_df, start_utc=start, end_utc=end)

    assert clean_df["timestamp_utc"].min() == start
    assert clean_df["timestamp_utc"].max() == end - pd.Timedelta(hours=1)
    assert len(clean_df) == 24  # exactly one declared day


def test_market_design_flag_uses_local_delivery_day_not_utc_midnight():
    """The 2025-10-01 cutover is defined by LOCAL delivery day, not UTC
    calendar midnight. For DE-LU (CEST, UTC+2 in early October),
    2025-10-01 00:00 Europe/Berlin is 2025-09-30T22:00:00Z -- two hours
    earlier than naive UTC midnight. Getting this wrong misclassifies
    the first two hours of the new delivery day as still pre-cutover.
    """
    ts = pd.to_datetime([
        "2025-09-30T21:45:00Z",  # 23:45 Sep 30 Berlin -> still old regime
        "2025-09-30T22:00:00Z",  # 00:00 Oct 1 Berlin -> new regime starts here
        "2025-09-30T23:00:00Z",  # 01:00 Oct 1 Berlin -> new regime
    ])
    df = pd.DataFrame({"timestamp_utc": ts})
    out = add_market_design_flag(df)
    assert list(out["post_15min_mtu"]) == [0, 1, 1]


def test_local_delivery_date_to_utc_de_lu_october_cutover():
    # DE-LU is on CEST (UTC+2) through late October, so local midnight
    # on 2025-10-01 is 2025-09-30T22:00:00Z, not 2025-10-01T00:00:00Z.
    result = local_delivery_date_to_utc("2025-10-01")
    assert result == pd.Timestamp("2025-09-30T22:00:00Z")


def test_completely_missing_hour_detected():
    """An hour missing from ALL sources can't be caught by the outer
    merge (there's simply no row for it), so it needs its own check
    against the expected UTC grid.
    """
    ts = pd.date_range("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts.delete(5)})  # drop one hour entirely

    missing = validate_hourly_coverage(
        df, expected_start=pd.Timestamp("2024-01-01T00:00:00Z"),
        expected_end=pd.Timestamp("2024-01-02T00:00:00Z"),
    )

    assert len(missing) == 1
    assert missing[0] == ts[5]


def test_complete_hourly_grid_has_no_missing_hours():
    ts = pd.date_range("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", freq="1h", tz="UTC", inclusive="left")
    df = pd.DataFrame({"timestamp_utc": ts})

    missing = validate_hourly_coverage(
        df, expected_start=pd.Timestamp("2024-01-01T00:00:00Z"),
        expected_end=pd.Timestamp("2024-01-02T00:00:00Z"),
    )

    assert missing == []
