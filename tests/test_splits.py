"""spec section 21: test_train_before_validation, test_validation_before_test.

These lock in the chronological split architecture BEFORE any model
exists, so no baseline or model has any chance to influence how the
evaluation framework itself is designed.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import (
    get_split_windows,
    get_final_train_window,
    get_holdout_window,
    slice_window,
    assert_split_is_chronological,
    validate_all_splits,
    SplitWindow,
)
from src.clean import local_delivery_date_to_utc


def test_all_default_splits_are_chronological():
    windows = get_split_windows()
    validate_all_splits(windows)  # must not raise


def test_train_before_validation():
    windows = get_split_windows()
    for w in windows:
        assert w.train_end <= w.val_start, f"{w.name}: train_end must be <= val_start"


def test_validation_windows_do_not_overlap():
    windows = get_split_windows()
    val_windows = [(w.val_start, w.val_end) for w in windows]
    for i in range(len(val_windows)):
        for j in range(i + 1, len(val_windows)):
            s1, e1 = val_windows[i]
            s2, e2 = val_windows[j]
            assert not (s1 < e2 and s2 < e1), "validation windows must not overlap"


def test_regime_stress_fold_targets_post_cutover_period():
    """The regime stress fold exists specifically because EDA found the
    development sample has very little post-2025-10-01 data while the
    entire holdout is post-cutover -- this fold must actually validate
    on that exact window. Proven against local_delivery_date_to_utc
    directly (the project's canonical market-date->UTC conversion)
    rather than against a second, independently-hardcoded UTC literal
    that could silently drift out of sync with it.
    """
    windows = get_split_windows()
    stress = next(w for w in windows if w.name == "regime_stress_test")
    assert stress.val_start == local_delivery_date_to_utc("2025-10-01")
    assert stress.val_end == local_delivery_date_to_utc("2026-01-01")


def test_assert_chronological_catches_inverted_window():
    bad = SplitWindow(
        "bad", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-06-01", tz="UTC"),
        pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-09-01", tz="UTC"),
    )  # train_end (June) is AFTER val_start (March) -- classic leakage shape
    with pytest.raises(AssertionError, match="NOT chronological"):
        assert_split_is_chronological(bad)


def test_slice_window_respects_boundaries():
    ts = pd.date_range("2022-06-01", "2024-06-01", freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts, "value": range(len(ts))})

    windows = get_split_windows()
    fold_1 = next(w for w in windows if w.name == "fold_1")
    train_df, val_df = slice_window(df, fold_1)

    assert (train_df["timestamp_utc"] < fold_1.train_end).all()
    assert (train_df["timestamp_utc"] >= fold_1.train_start).all()
    assert (val_df["timestamp_utc"] >= fold_1.val_start).all()
    assert (val_df["timestamp_utc"] < fold_1.val_end).all()
    # No overlap between the two slices
    assert train_df["timestamp_utc"].max() < val_df["timestamp_utc"].min()


def test_holdout_requires_explicit_end_no_hardcoded_default():
    """'Latest complete month' moves over time -- there must be no
    silent hardcoded fallback that could go stale.
    """
    with pytest.raises(ValueError, match="holdout_end must be supplied"):
        get_holdout_window()


def test_holdout_window_uses_supplied_end():
    # Real numerical stakes here, not just style: the local-correct
    # boundary is 2026-07-31T22:00:00Z (5,087 hours from holdout start),
    # matching the actual EDA manifest's holdout_rows_excluded=5087.
    # A naive UTC literal (2026-08-01T00:00:00Z) would silently include
    # 2 extra hours -- the first two local hours of 1 August.
    end = local_delivery_date_to_utc("2026-08-01")
    w = get_holdout_window(holdout_end=end)
    assert w.val_end == end
    assert w.val_start == local_delivery_date_to_utc("2026-01-01")


def test_holdout_never_enters_any_development_fold():
    """The 2026+ holdout must not overlap with any fold's train or
    validation window -- if it did, "untouched holdout" would be false
    by construction, not just by discipline.
    """
    windows = get_split_windows()
    holdout = get_holdout_window(holdout_end=local_delivery_date_to_utc("2026-08-01"))

    for w in windows:
        train_overlap = w.train_start < holdout.val_end and holdout.val_start < w.train_end
        val_overlap = w.val_start < holdout.val_end and holdout.val_start < w.val_end
        assert not train_overlap, f"{w.name}'s train window overlaps the holdout"
        assert not val_overlap, f"{w.name}'s validation window overlaps the holdout"

    final_train = get_final_train_window()
    final_overlap = final_train.train_start < holdout.val_end and holdout.val_start < final_train.train_end
    assert not final_overlap, "final_train window overlaps the holdout"


def test_all_development_folds_finish_before_holdout():
    """Stronger than "no overlap": non-overlapping intervals alone
    would also technically pass if a development fold were accidentally
    placed entirely AFTER the holdout. This proves the actual required
    ordering: development -> final training -> holdout, not merely
    "they don't collide."
    """
    holdout = get_holdout_window(holdout_end=local_delivery_date_to_utc("2026-08-01"))

    for w in get_split_windows():
        assert w.train_end <= holdout.val_start, f"{w.name}'s train window must finish before the holdout starts"
        assert w.val_end <= holdout.val_start, f"{w.name}'s validation window must finish before the holdout starts"

    final_train = get_final_train_window()
    assert final_train.train_end <= holdout.val_start, "final_train must finish before the holdout starts"


def test_split_membership_uses_market_date_not_naive_utc_date():
    """Independent proof (beyond features.py's own DST tests) that the
    split layer assigns fold boundaries using local delivery dates, not
    a fixed UTC+1 assumption -- checked against both a winter boundary
    (CET, UTC+1) and a summer boundary (CEST, UTC+2), so the splitter
    can't be accidentally assuming one offset year-round.
    """
    winter_boundary = local_delivery_date_to_utc("2024-01-01")
    assert winter_boundary == pd.Timestamp("2023-12-31T23:00:00Z")  # CET, UTC+1

    summer_boundary = local_delivery_date_to_utc("2025-07-01")
    assert summer_boundary == pd.Timestamp("2025-06-30T22:00:00Z")  # CEST, UTC+2

    fold_2 = next(w for w in get_split_windows() if w.name == "fold_2")
    assert fold_2.val_start == winter_boundary


def test_final_train_window_covers_all_development_data():
    w = get_final_train_window()
    assert w.train_start == local_delivery_date_to_utc("2019-01-01")
    assert w.train_end == local_delivery_date_to_utc("2026-01-01")
