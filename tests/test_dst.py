"""Spec section 21: test_spring_dst_transition_preserved, test_autumn_dst_transition_preserved.

Storage is UTC, so hourly UTC timestamps never skip or duplicate around a
DST boundary -- there's always exactly 24 UTC hours in a UTC day. The
risk is entirely in the *derived* Europe/Berlin local-time columns: a
naive implementation could silently drop or duplicate the local
23-hour spring day or 25-hour autumn day. These tests check that the
local day lengths come out right and that no UTC rows are lost or
duplicated in the process.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clean import add_local_time_columns


def _hourly_utc_range(start, end):
    ts = pd.date_range(start, end, freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp_utc": ts, "price_eur_mwh": range(len(ts))})


def test_spring_dst_transition_preserved():
    # 2024-03-31: EU spring-forward. Local clocks jump 02:00 -> 03:00 CET->CEST.
    # That local day has 23 local hours.
    df = _hourly_utc_range("2024-03-30T00:00:00Z", "2024-04-01T00:00:00Z")
    n_utc_before = len(df)

    out = add_local_time_columns(df.copy())

    # No UTC rows lost or duplicated by the local-time derivation.
    assert len(out) == n_utc_before
    assert out["timestamp_utc"].is_unique

    counts = out.groupby("delivery_date").size()
    spring_day = pd.Timestamp("2024-03-31").date()
    assert counts.loc[spring_day] == 23, (
        f"Expected 23 local hours on the spring-forward day, got {counts.loc[spring_day]}"
    )


def test_autumn_dst_transition_preserved():
    # 2024-10-27: EU fall-back. Local clocks repeat 02:00-03:00 CEST->CET.
    # That local day has 25 local hours.
    df = _hourly_utc_range("2024-10-26T00:00:00Z", "2024-10-28T00:00:00Z")
    n_utc_before = len(df)

    out = add_local_time_columns(df.copy())

    assert len(out) == n_utc_before
    assert out["timestamp_utc"].is_unique

    counts = out.groupby("delivery_date").size()
    autumn_day = pd.Timestamp("2024-10-27").date()
    assert counts.loc[autumn_day] == 25, (
        f"Expected 25 local hours on the fall-back day, got {counts.loc[autumn_day]}"
    )


def test_utc_timestamps_are_timezone_aware():
    df = _hourly_utc_range("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
    out = add_local_time_columns(df.copy())
    assert out["timestamp_utc"].dt.tz is not None
    assert str(out["timestamp_utc"].dt.tz) == "UTC"
