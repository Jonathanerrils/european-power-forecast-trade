"""Tests for scripts/verify_auction_sequence.py -- specifically the
DST-detection, delivery-interval-position, and sample-selection logic,
per the checklist:

1. DST detection: 2025-10-26 detected, 2025-10-25 not, fall-back day
   has 100 PT15M intervals (not 96).
2. Sampling: unique dates, <= n_sample_dates, earliest/latest appear
   when capacity permits, max-gap date appears, an affected DST date
   is retained even when it isn't among the earliest disagreements.
3. Clock ambiguity: local_time_with_offset distinguishes the two local
   01:xx-02:xx occurrences on the fall-back day via UTC offset.
4. delivery_interval_position is derived from the COMPLETE sequence-1
   series, not a filtered subset, and is unambiguous even on
   DST-affected days.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scripts.verify_auction_sequence as vas


# ---------------------------------------------------------------------
# 1. DST detection
# ---------------------------------------------------------------------
def test_autumn_dst_transition_day_detected():
    ts = pd.date_range("2025-10-25T00:00:00Z", "2025-10-27T00:00:00Z", freq="15min", tz="UTC")
    dst_dates = vas.find_dst_transition_dates(pd.Series(ts))
    assert pd.Timestamp("2025-10-26").date() in dst_dates


def test_normal_day_not_flagged_as_dst_transition():
    ts = pd.date_range("2025-10-25T00:00:00Z", "2025-10-27T00:00:00Z", freq="15min", tz="UTC")
    dst_dates = vas.find_dst_transition_dates(pd.Series(ts))
    assert pd.Timestamp("2025-10-25").date() not in dst_dates


def test_fallback_day_has_100_pt15m_intervals_not_96():
    """The whole reason this day needs special handling: it has MORE
    than the usual 96 quarter-hour intervals, not the same number.
    """
    local = pd.date_range("2025-10-26T00:00:00", periods=200, freq="15min", tz="Europe/Berlin")
    fallback_day_intervals = local[local.date == pd.Timestamp("2025-10-26").date()]
    assert len(fallback_day_intervals) == 100


def test_spring_forward_day_has_92_pt15m_intervals():
    local = pd.date_range("2025-03-30T00:00:00", periods=200, freq="15min", tz="Europe/Berlin")
    spring_day_intervals = local[local.date == pd.Timestamp("2025-03-30").date()]
    assert len(spring_day_intervals) == 92


# ---------------------------------------------------------------------
# 2. Sampling
# ---------------------------------------------------------------------
def _make_disagreeing_frame(dates, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for _ in range(3):
            rows.append({"local_delivery_date": d, "abs_diff": rng.uniform(0.1, 50)})
    return pd.DataFrame(rows)


def test_sample_dates_are_unique():
    dates = pd.date_range("2025-10-01", "2025-12-31", freq="D").date
    disagreeing = _make_disagreeing_frame(dates)
    sample = vas.select_sample_dates(disagreeing, n_sample_dates=10)
    assert len(sample) == len(set(sample))


def test_sample_size_never_exceeds_requested_cap():
    dates = pd.date_range("2025-10-01", "2025-12-31", freq="D").date
    disagreeing = _make_disagreeing_frame(dates)
    for n in [1, 5, 10, 20]:
        sample = vas.select_sample_dates(disagreeing, n_sample_dates=n)
        assert len(sample) <= n


def test_earliest_and_latest_dates_included_when_capacity_permits():
    dates = pd.date_range("2025-10-01", "2025-12-31", freq="D").date
    disagreeing = _make_disagreeing_frame(dates)
    sample = vas.select_sample_dates(disagreeing, n_sample_dates=10)
    assert dates[0] in sample
    assert dates[-1] in sample


def test_max_gap_date_included():
    dates = pd.date_range("2025-10-01", "2025-10-10", freq="D").date
    disagreeing = _make_disagreeing_frame(dates)
    # Force one date to have an obviously larger max gap than all others
    spike_date = dates[4]
    disagreeing.loc[disagreeing["local_delivery_date"] == spike_date, "abs_diff"] = 999.0
    sample = vas.select_sample_dates(disagreeing, n_sample_dates=5)
    assert spike_date in sample


def test_affected_dst_date_retained_even_if_not_an_early_disagreement():
    """The whole point of forcing DST dates in: a DST date buried deep
    in the list (not among the earliest/latest/largest-gap picks)
    must still make it into the sample when explicitly flagged.
    """
    dates = pd.date_range("2025-10-01", "2025-12-31", freq="D").date
    disagreeing = _make_disagreeing_frame(dates, seed=42)
    dst_date = pd.Timestamp("2025-11-15").date()  # deliberately unremarkable: not first/last/max-gap
    # Force its abs_diff to be mid-range, not extreme, so it wouldn't be
    # picked by any of the other priority rules
    disagreeing.loc[disagreeing["local_delivery_date"] == dst_date, "abs_diff"] = 5.0
    sample = vas.select_sample_dates(disagreeing, n_sample_dates=6, dst_transition_dates={dst_date})
    assert dst_date in sample


def test_sample_dates_returned_in_chronological_order():
    dates = pd.date_range("2025-10-01", "2025-12-31", freq="D").date
    disagreeing = _make_disagreeing_frame(dates)
    sample = vas.select_sample_dates(disagreeing, n_sample_dates=8)
    assert sample == sorted(sample)


# ---------------------------------------------------------------------
# 3. Clock ambiguity on the fall-back day
# ---------------------------------------------------------------------
def test_local_time_with_offset_disambiguates_repeated_fallback_hour():
    """02:15 local occurs TWICE on 2025-10-26 (once at CEST, once at
    CET) -- the printed local_time_with_offset must differ between them
    even though the naive HH:MM label would be identical.
    """
    first_occurrence = pd.Timestamp("2025-10-26T00:15:00Z").tz_convert("Europe/Berlin")
    second_occurrence = pd.Timestamp("2025-10-26T01:15:00Z").tz_convert("Europe/Berlin")
    label1 = first_occurrence.strftime("%H:%M %z")
    label2 = second_occurrence.strftime("%H:%M %z")
    assert label1 != label2
    assert label1.startswith("02:15") and label2.startswith("02:15")  # same naive clock label
    assert "+0200" in label1  # first pass, still CEST
    assert "+0100" in label2  # second pass, now CET, after fallback


# ---------------------------------------------------------------------
# 4. delivery_interval_position derived from the COMPLETE series
# ---------------------------------------------------------------------
def test_position_derived_from_complete_series_not_a_filtered_subset():
    """Regression guard for the exact failure mode flagged in review:
    deriving position from a filtered (e.g. disagreeing-only) subset
    would silently redefine 'position 3' as 'the third disagreement'
    rather than 'the third real market interval'. This proves position
    values match true chronological rank within the full day.
    """
    ts = pd.date_range("2025-10-05T00:00:00", periods=96, freq="15min", tz="Europe/Berlin").tz_convert("UTC")
    seq1 = pd.DataFrame({"timestamp_utc": ts})
    positions = vas.add_delivery_interval_position(seq1)
    assert list(positions["delivery_interval_position"]) == list(range(1, 97))


def test_position_correct_on_fallback_dst_day_100_intervals():
    ts = pd.date_range("2025-10-26T00:00:00", periods=100, freq="15min", tz="Europe/Berlin").tz_convert("UTC")
    seq1 = pd.DataFrame({"timestamp_utc": ts})
    positions = vas.add_delivery_interval_position(seq1)
    assert list(positions["delivery_interval_position"]) == list(range(1, 101))
    assert positions["delivery_interval_position"].max() == 100


def test_position_correct_on_spring_forward_dst_day_92_intervals():
    ts = pd.date_range("2025-03-30T00:00:00", periods=92, freq="15min", tz="Europe/Berlin").tz_convert("UTC")
    seq1 = pd.DataFrame({"timestamp_utc": ts})
    positions = vas.add_delivery_interval_position(seq1)
    assert list(positions["delivery_interval_position"]) == list(range(1, 93))
    assert positions["delivery_interval_position"].max() == 92


def test_position_resets_at_each_local_delivery_date():
    ts_day1 = pd.date_range("2025-10-05T00:00:00", periods=96, freq="15min", tz="Europe/Berlin").tz_convert("UTC")
    ts_day2 = pd.date_range("2025-10-06T00:00:00", periods=96, freq="15min", tz="Europe/Berlin").tz_convert("UTC")
    seq1 = pd.DataFrame({"timestamp_utc": ts_day1.append(ts_day2)})
    positions = vas.add_delivery_interval_position(seq1)
    assert positions["delivery_interval_position"].iloc[0] == 1
    assert positions["delivery_interval_position"].iloc[95] == 96
    assert positions["delivery_interval_position"].iloc[96] == 1  # resets for day 2
    assert positions["delivery_interval_position"].iloc[-1] == 96


def test_missing_interval_does_not_shift_later_positions():
    """Regression test for a real bug: an earlier version computed
    position via groupby.cumcount(), which is only correct if the day's
    series has no gaps. Live-fetched ENTSO-E data is not guaranteed
    complete -- one missing interval silently shifted every later
    position that day backward by one (verified: position 6 became 5).
    Elapsed-time-since-local-midnight must SKIP the missing position
    instead, leaving every other row correctly labelled.
    """
    full = pd.date_range("2025-10-05T00:00:00", periods=96, freq="15min", tz="Europe/Berlin").tz_convert("UTC")
    missing = full.delete(4)  # remove true position 5 (index 4)
    seq1 = pd.DataFrame({"timestamp_utc": missing})
    out = vas.add_delivery_interval_position(seq1)

    row = out[out["timestamp_utc"] == full[5]]  # the interval that should be position 6
    assert row["delivery_interval_position"].iloc[0] == 6, (
        "position silently shifted backward after a missing interval"
    )
    # Also confirm position 5 (the value the bug would produce) is genuinely absent
    assert 5 not in out["delivery_interval_position"].values


# ---------------------------------------------------------------------
# 5. Duplicate-timestamp guard
# ---------------------------------------------------------------------
def test_duplicate_timestamps_in_a_sequence_raise():
    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2025-10-05T10:00:00Z", "2025-10-05T10:00:00Z", "2025-10-05T10:15:00Z"]),
        "price_seq1": [50.0, 52.0, 60.0],
    })
    with pytest.raises(AssertionError, match="duplicate timestamps"):
        vas.assert_unique_sequence_timestamps(df, "sequence 1")


def test_unique_timestamps_do_not_raise():
    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2025-10-05T10:00:00Z", "2025-10-05T10:15:00Z"]),
        "price_seq1": [50.0, 60.0],
    })
    vas.assert_unique_sequence_timestamps(df, "sequence 1")  # must not raise


# ---------------------------------------------------------------------
# 6. EPEX URL correctness -- pins the exact bug that was already fixed once
# ---------------------------------------------------------------------
def test_epex_url_targets_delu_sdac_15_minute_product():
    """Regression test for the actual historical bug: an earlier version
    of this script pointed at product=60 (the 60-minute aggregated
    index) instead of product=15 (the actual PT15M auction product this
    audit is about) -- verified by directly fetching both URLs, where
    product=60 returned nothing but a loading placeholder and product=15
    returned real price data. Pin the fix so it can't silently regress.
    """
    url = vas.build_epex_url("2025-10-26")
    assert "market_area=DE-LU" in url
    assert "sub_modality=DayAhead" in url
    assert "product=15" in url
    assert "product=60" not in url


# ---------------------------------------------------------------------
# 7. Audit window must never include 2026 (the untouched holdout)
# ---------------------------------------------------------------------
def test_audit_window_end_stays_within_2025():
    """Regression test for a real methodological issue: an earlier
    version's window_end reached into 2026 (the untouched final
    holdout). Even a data-construction audit that only PRINTS
    earliest/latest dates and largest/smallest discrepancies is a form
    of holdout peeking if it surfaces real 2026 price patterns before
    the holdout evaluation. This inspects main()'s source directly
    rather than running it, since window_end is a local variable, not
    an importable constant.
    """
    import inspect
    source = inspect.getsource(vas.main)
    assert '"2026-01-01"' in source, (
        "window_end must be pinned to 2026-01-01 (exclusive) -- the audit window "
        "must stop at the end of 2025, keeping all of 2026 (the untouched holdout) "
        "completely out of scope for this script"
    )
    assert '"2026-08-01"' not in source, "window_end must not reach into the holdout year"


# ---------------------------------------------------------------------
# 8. Sequence-2-only intervals (a different, more serious problem)
# ---------------------------------------------------------------------
def test_find_sequence2_only_intervals_identifies_the_right_rows():
    merged = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([
            "2025-10-05T10:00:00Z", "2025-10-05T10:15:00Z", "2025-10-05T10:30:00Z",
        ]),
        "price_seq1": [50.0, np.nan, 60.0],
        "price_seq2": [55.0, 62.0, np.nan],
    })
    seq2_only = vas.find_sequence2_only_intervals(merged)
    assert len(seq2_only) == 1
    assert seq2_only["timestamp_utc"].iloc[0] == pd.Timestamp("2025-10-05T10:15:00Z")
    assert seq2_only["price_seq2"].iloc[0] == 62.0
    assert "price_seq1" not in seq2_only.columns  # seq1 is absent for these rows by definition


def test_find_sequence2_only_intervals_empty_when_none_exist():
    merged = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2025-10-05T10:00:00Z", "2025-10-05T10:15:00Z"]),
        "price_seq1": [50.0, 55.0],
        "price_seq2": [55.0, 60.0],
    })
    seq2_only = vas.find_sequence2_only_intervals(merged)
    assert len(seq2_only) == 0


# ---------------------------------------------------------------------
# 9. run_version versioning (regression guard for the same class of bug
#    already found and fixed in run_eda.py -- a fixed, unversioned
#    output path silently overwrites prior audit results)
# ---------------------------------------------------------------------
def test_run_version_is_mandatory():
    with pytest.raises(SystemExit):
        vas.resolve_run_args([])


def test_run_version_with_default_sample_size():
    run_version, n = vas.resolve_run_args(["a03fix_v1"])
    assert run_version == "a03fix_v1"
    assert n == 10


def test_run_version_with_explicit_sample_size():
    run_version, n = vas.resolve_run_args(["a03fix_v1", "20"])
    assert run_version == "a03fix_v1"
    assert n == 20


def test_non_positive_sample_size_rejected():
    with pytest.raises(SystemExit, match="n_sample_dates must be"):
        vas.resolve_run_args(["a03fix_v1", "0"])


def test_too_many_arguments_rejected():
    with pytest.raises(SystemExit):
        vas.resolve_run_args(["a", "b", "c"])
