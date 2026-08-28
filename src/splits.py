"""Chronological split architecture (spec section 9), locked in BEFORE
src/models.py exists -- so no model or naive baseline has any chance to
influence how the evaluation framework itself is designed.

No shuffling, ever. Every split is strictly chronological: a
validation window's train data never contains a single timestamp at or
after that window's own start. This is checked structurally by
assert_split_is_chronological(), not just by construction.

Fold structure (per project research notes / README hypotheses):

    Fold 1:            train 2019-01-01 -> 2023-01-01, validate 2023
    Fold 2:            train 2019-01-01 -> 2024-01-01, validate 2024
    Fold 3:            train 2019-01-01 -> 2025-01-01, validate Jan-Sep 2025
    Regime stress test: train 2019-01-01 -> 2025-10-01, validate Oct-Dec 2025

The regime stress fold exists because of a real finding from EDA: the
development sample has only ~2,209 rows (about 3 months) under the
post-2025-10-01 15-minute-MTU market design, while the ENTIRE 2026+
holdout is under that same new regime. This fold is the only chance to
see how a model trained mostly on the old regime performs on the new
one, before the holdout is ever touched.

    Final model:  train 2019-01-01 -> 2026-01-01 (all development data)
    Final holdout: 2026-01-01 -> latest complete month (untouched until
                   the model is completely frozen)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .clean import local_delivery_date_to_utc


@dataclass
class SplitWindow:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp    # exclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp      # exclusive


def _boundary(date_str: str, local_tz: str) -> pd.Timestamp:
    return local_delivery_date_to_utc(date_str, local_tz=local_tz)


def get_split_windows(
    local_tz: str = "Europe/Berlin",
    sample_start: str = "2019-01-01",
) -> List[SplitWindow]:
    """Returns the chronological validation folds, in order. Does not
    include the final-model / holdout windows -- see
    get_final_train_window() and get_holdout_window() for those,
    kept separate so it's structurally impossible to accidentally
    include a validation fold's data in the final holdout check.
    """
    b = lambda d: _boundary(d, local_tz)  # noqa: E731
    start = b(sample_start)

    return [
        SplitWindow("fold_1", start, b("2023-01-01"), b("2023-01-01"), b("2024-01-01")),
        SplitWindow("fold_2", start, b("2024-01-01"), b("2024-01-01"), b("2025-01-01")),
        SplitWindow("fold_3", start, b("2025-01-01"), b("2025-01-01"), b("2025-10-01")),
        SplitWindow(
            "regime_stress_test", start, b("2025-10-01"), b("2025-10-01"), b("2026-01-01")
        ),
    ]


def get_final_train_window(
    local_tz: str = "Europe/Berlin",
    sample_start: str = "2019-01-01",
    holdout_start: str = "2026-01-01",
) -> SplitWindow:
    """All development data, once the model is frozen and only the
    final holdout evaluation remains.
    """
    b = lambda d: _boundary(d, local_tz)  # noqa: E731
    return SplitWindow(
        "final_train", b(sample_start), b(holdout_start), b(holdout_start), b(holdout_start),
    )


def get_holdout_window(
    local_tz: str = "Europe/Berlin",
    holdout_start: str = "2026-01-01",
    holdout_end: pd.Timestamp = None,
) -> SplitWindow:
    """The untouched final holdout. holdout_end should come from
    entsoe_client.get_default_date_range()'s end boundary (latest
    complete month) at the time the model is actually frozen -- not
    hardcoded here, since "latest complete month" is a moving target.
    """
    b = lambda d: _boundary(d, local_tz)  # noqa: E731
    start = b(holdout_start)
    if holdout_end is None:
        raise ValueError(
            "holdout_end must be supplied explicitly (e.g. from "
            "entsoe_client.get_default_date_range()) -- there is no safe default, "
            "since 'latest complete month' changes over time and hardcoding it here "
            "risks silently using a stale boundary."
        )
    return SplitWindow("holdout", start, start, start, holdout_end)


def slice_window(df: pd.DataFrame, window: SplitWindow, ts_col: str = "timestamp_utc"):
    """Return (train_df, val_df) for a SplitWindow. train_df is
    strictly [train_start, train_end); val_df is strictly
    [val_start, val_end). For the final-train/holdout windows, val_df
    will be empty/holdout-only as appropriate -- callers should use
    get_holdout_window()'s val_start/val_end directly for holdout data.
    """
    ts = pd.to_datetime(df[ts_col], utc=True)
    train_df = df[(ts >= window.train_start) & (ts < window.train_end)]
    val_df = df[(ts >= window.val_start) & (ts < window.val_end)]
    return train_df, val_df


def assert_split_is_chronological(window: SplitWindow) -> None:
    """Structural check, not just a naming convention: train data must
    end at or before validation data begins. Raises AssertionError
    with a specific message on failure -- this is the guard that makes
    "no shuffling, ever" an enforced property rather than a promise.
    """
    if window.train_end > window.val_start:
        raise AssertionError(
            f"Split '{window.name}' is NOT chronological: train_end "
            f"({window.train_end}) is after val_start ({window.val_start}). "
            f"Train data would include information from at or after the "
            f"validation window -- this is exactly the kind of leakage "
            f"random train_test_split(shuffle=True) would introduce, and "
            f"this guard exists specifically to prevent it structurally."
        )
    if window.val_start > window.val_end:
        raise AssertionError(
            f"Split '{window.name}' has val_start ({window.val_start}) after "
            f"val_end ({window.val_end})."
        )
    if window.train_start > window.train_end:
        raise AssertionError(
            f"Split '{window.name}' has train_start ({window.train_start}) after "
            f"train_end ({window.train_end})."
        )


def validate_all_splits(windows: List[SplitWindow]) -> None:
    """Run assert_split_is_chronological on every window, and also
    verify successive validation folds don't overlap each other (each
    fold's validation window should be disjoint from every other
    fold's validation window, since each is a distinct time period).
    """
    for w in windows:
        assert_split_is_chronological(w)

    val_windows = [(w.name, w.val_start, w.val_end) for w in windows]
    for i in range(len(val_windows)):
        name_i, start_i, end_i = val_windows[i]
        for j in range(i + 1, len(val_windows)):
            name_j, start_j, end_j = val_windows[j]
            overlap = start_i < end_j and start_j < end_i
            if overlap:
                raise AssertionError(
                    f"Validation windows '{name_i}' and '{name_j}' overlap "
                    f"({start_i}-{end_i} vs {start_j}-{end_j}). Each fold's "
                    f"validation period should be a distinct time window."
                )
