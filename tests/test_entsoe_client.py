"""Regression test for the dual-resolution DE-LU price bug: ENTSO-E
returns BOTH a PT60M (standard hourly) and PT15M (quarter-hour) price
TimeSeries per day, going back to 2019 -- these are genuinely different
prices, not the same auction at different granularity. A real ingestion
run against ENTSO-E surfaced this: 264,877 raw price rows for 2019
alone (~3x expected), with 88,070 "duplicate" timestamps that were
actually two different published prices for the same instant.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.entsoe_client import (
    select_price_resolution,
    select_primary_auction_sequence,
    get_default_date_range,
    EntsoeClient,
    IngestionRecord,
)


def _row(ts, resolution_min, price):
    return {"timestamp_utc": pd.Timestamp(ts, tz="UTC"), "resolution_min": resolution_min, "price_eur_mwh": price}


# ---------------------------------------------------------------------
# curveType decoding (_expand_period_points / _parse_timeseries)
#
# Regression tests for a real production bug: an earlier version of
# _parse_timeseries() only emitted rows for explicit <Point> elements.
# Post-2025-10-01 DE-LU prices use curveType=A03 ("variable sized
# blocks"), where ENTSO-E only publishes a Point when the price CHANGES
# -- an unpublished position means "same as the previous block", not
# "missing". The old parser silently undercounted real published prices
# as if they were gaps. Verified against ENTSO-E's own official
# specification ("The Introduction of Different Time Series
# Possibilities (CurveType) within ENTSO-E Electronic Documents", v1.4,
# section 4.3) before this fix was made, not assumed.
# ---------------------------------------------------------------------
def _make_client():
    # EntsoeClient.__init__ needs a valid config + token; tests exercise
    # the pure parsing methods directly, so a minimal stand-in avoids
    # needing a real token/network for logic that has none of its own
    # dependency on either.
    client = EntsoeClient.__new__(EntsoeClient)
    return client


def test_a03_expansion_matches_entsoe_official_spec_worked_example():
    """This is ENTSO-E's OWN worked example from the official curveType
    specification document (section 4.3, Figure 4): a 24h period at
    PT4H resolution (6 positions), with explicit Points only at
    positions {1: 50, 2: 100, 4: 150, 5: 50} (positions 3 and 6 omitted
    because the price doesn't change). The spec's own stated result:
    position 3 inherits block 2's value (100), position 6 inherits
    block 5's value (50) and extends to the end of the period.
    """
    client = _make_client()
    points_by_position = {1: 50.0, 2: 100.0, 4: 150.0, 5: 50.0}
    expanded = client._expand_period_points(points_by_position, curve_type="A03", n_positions=6)

    expected_values = {1: 50.0, 2: 100.0, 3: 100.0, 4: 150.0, 5: 50.0, 6: 50.0}
    expected_synthesized = {1: False, 2: False, 3: True, 4: False, 5: False, 6: True}
    for pos in range(1, 7):
        value, is_synthesized = expanded[pos]
        assert value == expected_values[pos], f"position {pos}: expected {expected_values[pos]}, got {value}"
        assert is_synthesized == expected_synthesized[pos], f"position {pos}: is_synthesized mismatch"


def test_a01_requires_every_position_no_forward_fill():
    """A01 (sequential fixed size blocks, also the default when
    curveType is omitted) requires every position to be explicitly
    provided per spec -- no forward-fill logic applies. A genuinely
    missing position under A01 is malformed and must fail closed rather
    than emit a null that a downstream aggregation could silently ignore.
    """
    client = _make_client()
    points_by_position = {1: 50.0, 2: 100.0, 3: 100.0}  # all 3 explicitly provided
    expanded = client._expand_period_points(points_by_position, curve_type="A01", n_positions=3)
    assert expanded == {1: (50.0, False), 2: (100.0, False), 3: (100.0, False)}


def test_a01_missing_position_fails_closed():
    client = _make_client()
    points_by_position = {1: 50.0, 3: 100.0}  # position 2 genuinely absent
    with pytest.raises(ValueError, match="A01 requires an explicit Point for every interval"):
        client._expand_period_points(points_by_position, curve_type="A01", n_positions=3)


def test_a03_must_start_at_position_one():
    client = _make_client()
    # A03 can omit unchanged later positions, but a continuous Period cannot
    # begin without a block value. Genuine gaps belong in disjoint Periods.
    with pytest.raises(ValueError, match="A03 must start with an explicit Point at position 1"):
        client._expand_period_points({2: 50.0}, curve_type="A03", n_positions=4)


def test_expand_rejects_out_of_range_point_position():
    client = _make_client()
    with pytest.raises(ValueError, match="outside valid range"):
        client._expand_period_points({1: 50.0, 5: 60.0}, curve_type="A03", n_positions=4)


def test_expand_rejects_explicit_null_value():
    client = _make_client()
    with pytest.raises(ValueError, match="missing a numeric value"):
        client._expand_period_points({1: None}, curve_type="A03", n_positions=4)


def test_curve_type_none_defaults_to_a01_per_spec():
    """Spec: 'If the CurveType attribute is omitted... a default value
    of sequential fixed size blocks [A01] shall be understood.'
    """
    client = _make_client()
    points_by_position = {1: 50.0, 2: 100.0, 3: 100.0}
    expanded_none = client._expand_period_points(points_by_position, curve_type=None, n_positions=3)
    expanded_a01 = client._expand_period_points(points_by_position, curve_type="A01", n_positions=3)
    assert expanded_none == expanded_a01


def test_a02_points_are_not_expanded():
    """A02 (points) is explicitly sparse by design -- 'no relational
    significance between each reading'. Must NOT be forward-filled.
    """
    client = _make_client()
    points_by_position = {1: 50.0, 2: 100.0, 5: 150.0}  # positions 3, 4 genuinely have no reading
    expanded = client._expand_period_points(points_by_position, curve_type="A02", n_positions=6)
    assert set(expanded.keys()) == {1, 2, 5}, "A02 must only contain explicitly provided positions"


def test_unrecognized_curve_type_fails_closed():
    """A04/A05 use linear interpolation between breakpoints -- a
    fundamentally different reconstruction this project doesn't
    implement. Must raise, not silently mishandle.
    """
    client = _make_client()
    with pytest.raises(NotImplementedError, match="A04"):
        client._expand_period_points({1: 50.0}, curve_type="A04", n_positions=6)


def test_parse_timeseries_full_a03_document_reconstructs_all_positions():
    """End-to-end: a real-shaped Publication_MarketDocument with
    curveType=A03 and sparse Points must produce one row per quarter-hour
    position, not one row per explicit XML Point.
    """
    client = _make_client()
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
    <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
      <mRID>doc-1</mRID>
      <TimeSeries>
        <mRID>1</mRID>
        <curveType>A03</curveType>
        <classificationSequence_AttributeInstanceComponent.position>1</classificationSequence_AttributeInstanceComponent.position>
        <Period>
          <timeInterval><start>2025-10-01T00:00Z</start><end>2025-10-01T01:00Z</end></timeInterval>
          <resolution>PT15M</resolution>
          <Point><position>1</position><price.amount>50.0</price.amount></Point>
          <Point><position>3</position><price.amount>60.0</price.amount></Point>
        </Period>
      </TimeSeries>
    </Publication_MarketDocument>"""
    rows = client._parse_timeseries(xml_text)
    assert len(rows) == 4, f"expected 4 rows (one per PT15M position in a 1h period), got {len(rows)}"
    values = {row[0]: row[2] for row in rows}  # ts_utc -> value
    is_synth = {row[0]: row[6] for row in rows}  # ts_utc -> is_synthesized
    expected_positions_utc = [pd.Timestamp(f"2025-10-01T00:{m:02d}:00Z") for m in (0, 15, 30, 45)]
    assert values[expected_positions_utc[0]] == 50.0  # position 1, explicit
    assert values[expected_positions_utc[1]] == 50.0  # position 2, forward-filled from position 1
    assert values[expected_positions_utc[2]] == 60.0  # position 3, explicit
    assert values[expected_positions_utc[3]] == 60.0  # position 4, forward-filled from position 3
    assert is_synth[expected_positions_utc[1]] is True
    assert is_synth[expected_positions_utc[3]] is True
    assert is_synth[expected_positions_utc[0]] is False
    assert is_synth[expected_positions_utc[2]] is False



def test_parse_timeseries_rejects_duplicate_point_positions():
    client = _make_client()
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
    <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
      <TimeSeries>
        <curveType>A03</curveType>
        <Period>
          <timeInterval><start>2025-10-01T00:00Z</start><end>2025-10-01T01:00Z</end></timeInterval>
          <resolution>PT15M</resolution>
          <Point><position>1</position><price.amount>50.0</price.amount></Point>
          <Point><position>1</position><price.amount>60.0</price.amount></Point>
        </Period>
      </TimeSeries>
    </Publication_MarketDocument>"""
    with pytest.raises(ValueError, match="Duplicate ENTSO-E Point position 1"):
        client._parse_timeseries(xml_text)


def test_parse_timeseries_rejects_point_without_value():
    client = _make_client()
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
    <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
      <TimeSeries>
        <curveType>A03</curveType>
        <Period>
          <timeInterval><start>2025-10-01T00:00Z</start><end>2025-10-01T01:00Z</end></timeInterval>
          <resolution>PT15M</resolution>
          <Point><position>1</position></Point>
        </Period>
      </TimeSeries>
    </Publication_MarketDocument>"""
    with pytest.raises(ValueError, match="has no 'price.amount' or 'quantity' value"):
        client._parse_timeseries(xml_text)


def test_parse_timeseries_rejects_period_not_divisible_by_resolution():
    client = _make_client()
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
    <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
      <TimeSeries>
        <curveType>A03</curveType>
        <Period>
          <timeInterval><start>2025-10-01T00:00Z</start><end>2025-10-01T00:50Z</end></timeInterval>
          <resolution>PT15M</resolution>
          <Point><position>1</position><price.amount>50.0</price.amount></Point>
        </Period>
      </TimeSeries>
    </Publication_MarketDocument>"""
    with pytest.raises(ValueError, match="not exactly divisible"):
        client._parse_timeseries(xml_text)


def test_parse_timeseries_rejects_non_positive_period_duration():
    client = _make_client()
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
    <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
      <TimeSeries>
        <curveType>A03</curveType>
        <Period>
          <timeInterval><start>2025-10-01T01:00Z</start><end>2025-10-01T01:00Z</end></timeInterval>
          <resolution>PT15M</resolution>
          <Point><position>1</position><price.amount>50.0</price.amount></Point>
        </Period>
      </TimeSeries>
    </Publication_MarketDocument>"""
    with pytest.raises(ValueError, match="non-positive duration"):
        client._parse_timeseries(xml_text)

def test_price_resolution_selection_pre_cutover_keeps_pt60m():
    # Same hour, two different published prices: PT60M=28.32, and the
    # PT15M series' first point at the same timestamp is a different
    # number (40.16, per real ENTSO-E data for 2019-01-01).
    df = pd.DataFrame([
        _row("2019-01-01T00:00:00Z", 60, 28.32),
        _row("2019-01-01T00:00:00Z", 15, 40.16),
        _row("2019-01-01T00:15:00Z", 15, 27.51),
        _row("2019-01-01T00:30:00Z", 15, 13.84),
        _row("2019-01-01T00:45:00Z", 15, 31.61),
        _row("2019-01-01T01:00:00Z", 60, 10.07),
    ])

    out = select_price_resolution(df, cutover="2025-10-01")

    # Only the PT60M rows survive pre-cutover; no timestamp collision left.
    assert len(out) == 2
    assert out["timestamp_utc"].is_unique
    assert set(out["price_eur_mwh"]) == {28.32, 10.07}


def test_price_resolution_selection_post_cutover_keeps_pt15m():
    df = pd.DataFrame([
        _row("2025-10-01T00:00:00Z", 60, 99.99),   # legacy hourly product, if still published
        _row("2025-10-01T00:00:00Z", 15, 100.0),
        _row("2025-10-01T00:15:00Z", 15, 120.0),
        _row("2025-10-01T00:30:00Z", 15, 80.0),
        _row("2025-10-01T00:45:00Z", 15, 140.0),
    ])

    out = select_price_resolution(df, cutover="2025-10-01")

    assert len(out) == 4
    assert out["timestamp_utc"].is_unique
    assert 99.99 not in set(out["price_eur_mwh"])
    assert set(out["price_eur_mwh"]) == {100.0, 120.0, 80.0, 140.0}


def test_select_primary_auction_sequence_drops_secondary_sequence():
    """Regression test built directly from a real collision: ENTSO-E
    published TWO PT15M price series for 2025-09-30T22:00Z with
    different prices (102.60 vs 112.10), distinguished only by
    classificationSequence_AttributeInstanceComponent.position (1 vs 2).
    """
    df = pd.DataFrame([
        {"timestamp_utc": pd.Timestamp("2025-09-30T22:00:00Z"), "price_eur_mwh": 102.60, "auction_sequence": 1},
        {"timestamp_utc": pd.Timestamp("2025-09-30T22:00:00Z"), "price_eur_mwh": 112.10, "auction_sequence": 2},
        {"timestamp_utc": pd.Timestamp("2025-09-30T22:15:00Z"), "price_eur_mwh": 92.24, "auction_sequence": 1},
        {"timestamp_utc": pd.Timestamp("2025-09-30T22:15:00Z"), "price_eur_mwh": 108.10, "auction_sequence": 2},
    ])

    out = select_primary_auction_sequence(df)

    assert len(out) == 2
    assert out["timestamp_utc"].is_unique
    assert set(out["price_eur_mwh"]) == {102.60, 92.24}
    assert "auction_sequence" not in out.columns


def test_select_primary_auction_sequence_keeps_rows_with_no_sequence_info():
    """A missing/None auction_sequence value must pass through unchanged,
    not get dropped as if "missing" meant "wrong sequence". This is NOT
    specific to pre-2025 price data -- the full historical structural
    audit (scripts/audit_price_curve_types.py) confirmed
    classificationSequence is present throughout the ENTIRE price
    history, not just from 2025-10-01 onward (an earlier version of
    this test's docstring incorrectly claimed otherwise). This case
    covers other document types (e.g. load/wind/solar forecasts) that
    never carry this field at all, or any genuinely missing value.
    """
    df = pd.DataFrame([
        {"timestamp_utc": pd.Timestamp("2019-01-01T00:00:00Z"), "price_eur_mwh": 40.16, "auction_sequence": None},
    ])
    out = select_primary_auction_sequence(df)
    assert len(out) == 1
    assert out.iloc[0]["price_eur_mwh"] == 40.16


def test_select_primary_auction_sequence_empty_input():
    df = pd.DataFrame(columns=["timestamp_utc", "price_eur_mwh", "auction_sequence"])
    out = select_primary_auction_sequence(df)
    assert out.empty


def test_price_resolution_selection_uses_local_delivery_day_boundary():
    """At 2025-09-30T22:00:00Z (= 00:00 Oct 1 Europe/Berlin), the new
    15-minute regime has already started even though it's still
    2025-09-30 in UTC. PT15M must win here, not PT60M.
    """
    df = pd.DataFrame([
        _row("2025-09-30T21:45:00Z", 60, 40.0),   # 23:45 Berlin -> still old regime
        _row("2025-09-30T22:00:00Z", 15, 50.0),   # 00:00 Berlin Oct 1 -> new regime
        _row("2025-09-30T22:00:00Z", 60, 999.0),  # legacy hourly product, if still published -> must be excluded
        _row("2025-09-30T22:15:00Z", 15, 55.0),
    ])

    out = select_price_resolution(df, cutover="2025-10-01")

    assert len(out) == 3
    assert 999.0 not in set(out["price_eur_mwh"])
    assert set(out["price_eur_mwh"]) == {40.0, 50.0, 55.0}


def test_get_default_date_range_uses_local_delivery_day_boundary():
    """Regression test for a real finding: naive UTC-midnight parsing of
    start_date/end_date doesn't match what ENTSO-E actually returns (it
    organizes data by local delivery day). A real ingestion run's frozen
    dataset started at 2018-12-31T23:00Z even though config declared
    start_date: '2019-01-01' -- one hour of data from outside the
    declared range leaked in silently. Using local_delivery_date_to_utc
    for both boundaries makes the declared config dates match reality.
    """
    cfg = {
        "data": {"start_date": "2019-01-01", "end_date": "2019-02-01"},
        "market": {"timezone_local": "Europe/Berlin"},
    }
    start, end = get_default_date_range(cfg)
    # 2019-01-01 00:00 Europe/Berlin (CET, UTC+1 in January) = 2018-12-31T23:00:00Z
    assert start == pd.Timestamp("2018-12-31T23:00:00Z").to_pydatetime()
    assert end == pd.Timestamp("2019-01-31T23:00:00Z").to_pydatetime()


def test_wind_solar_pivot_raises_on_duplicate_timestamp_psr_pairs(monkeypatch):
    """Regression test: aggfunc='mean' would have silently averaged
    duplicate (timestamp, psr_type) rows into a value that may never
    have existed as an actual forecast -- the same class of mistake
    already found and fixed for the price series. Verify duplicates are
    now detected and raised on, not silently smoothed over.
    """
    import os
    os.environ["ENTSOE_TOKEN"] = "dummy-for-test"
    client = EntsoeClient()

    dupe_df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-05-20T15:00:00Z", "2024-05-20T15:00:00Z"], utc=True),
        "value_mw": [21000.0, 24000.0],
        "psr_type": ["B19", "B19"],  # same timestamp, same psr_type, DIFFERENT values
    })
    monkeypatch.setattr(client, "_fetch_generic", lambda *a, **kw: dupe_df)

    with pytest.raises(ValueError, match="duplicate"):
        client.fetch_wind_solar_forecast(
            pd.Timestamp("2024-05-20", tz="UTC"), pd.Timestamp("2024-05-21", tz="UTC")
        )


def test_price_resolution_selection_empty_input():
    df = pd.DataFrame(columns=["timestamp_utc", "resolution_min", "price_eur_mwh"])
    out = select_price_resolution(df, cutover="2025-10-01")
    assert out.empty
    assert "resolution_min" not in out.columns


def test_save_ingestion_log_overwrites_not_appends(tmp_path):
    """Regression test for a real bug found by reading an actual uploaded
    ingestion_log.jsonl: an earlier version opened in append mode, so
    three re-runs during iterative debugging (all using an old,
    naive-UTC-boundary version of get_default_date_range) each appended
    their own records on top of what was already there, and a fourth,
    corrected run added its own on top of THAT -- 32 total price-chunk
    log entries where only 8 reflected current, correct behavior.
    Calling save_ingestion_log() twice must leave only the SECOND call's
    records in the file, matching delu_hourly.parquet being overwritten
    wholesale each run, not accumulated silently across every historical
    attempt.
    """
    client = _make_client()
    log_path = tmp_path / "ingestion_log.jsonl"

    client.ingestion_log = [
        IngestionRecord(
            source="ENTSO-E Transparency Platform", retrieval_timestamp_utc="2026-01-01T00:00:00Z",
            requested_start="2019-01-01T00:00:00Z", requested_end="2020-01-01T00:00:00Z",
            bidding_zone="DE-LU", data_type="day_ahead_price", unit="EUR/MWh", timezone="UTC",
            raw_row_count=111, cache_hit=True, request_params={"periodStart": "old"},
        ),
    ]
    client.save_ingestion_log(path=log_path)

    client.ingestion_log = [
        IngestionRecord(
            source="ENTSO-E Transparency Platform", retrieval_timestamp_utc="2026-01-02T00:00:00Z",
            requested_start="2018-12-31T23:00:00Z", requested_end="2019-12-31T23:00:00Z",
            bidding_zone="DE-LU", data_type="day_ahead_price", unit="EUR/MWh", timezone="UTC",
            raw_row_count=222, cache_hit=False, request_params={"periodStart": "corrected"},
        ),
    ]
    client.save_ingestion_log(path=log_path)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1, (
        f"expected exactly 1 record after the second save_ingestion_log() call "
        f"(overwritten, not appended), got {len(lines)}"
    )
    record = json.loads(lines[0])
    assert record["raw_row_count"] == 222
    assert record["request_params"]["periodStart"] == "corrected"
