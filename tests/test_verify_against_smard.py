"""Tests for scripts/verify_against_smard.py. Network calls (the
requests.get boundary) are the only thing NOT tested here -- everything
downstream of a response is real logic and gets tested against it.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scripts.verify_against_smard as smard


# ---------------------------------------------------------------------
# select_relevant_chunks
# ---------------------------------------------------------------------
def test_select_relevant_chunks_includes_the_chunk_covering_window_start():
    """The chunk containing start_utc doesn't necessarily START at
    start_utc -- it starts BEFORE it and extends forward. This is the
    core correctness property: omitting this chunk would silently
    drop the first part of the requested window.
    """
    chunks = [pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC")]
    selected = smard.select_relevant_chunks(chunks, pd.Timestamp("2025-10-15", tz="UTC"), pd.Timestamp("2025-10-20", tz="UTC"))
    assert selected == [pd.Timestamp("2025-10-01", tz="UTC")]


def test_select_relevant_chunks_spans_multiple_chunks():
    chunks = [pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC"), pd.Timestamp("2025-12-01", tz="UTC")]
    selected = smard.select_relevant_chunks(chunks, pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-12-01", tz="UTC"))
    assert selected == [pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC")]


def test_select_relevant_chunks_start_exactly_on_a_chunk_boundary():
    chunks = [pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC")]
    selected = smard.select_relevant_chunks(chunks, pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-10-15", tz="UTC"))
    assert selected == [pd.Timestamp("2025-10-01", tz="UTC")]


def test_select_relevant_chunks_no_chunk_before_start_returns_only_later_ones():
    chunks = [pd.Timestamp("2025-11-01", tz="UTC"), pd.Timestamp("2025-12-01", tz="UTC")]
    selected = smard.select_relevant_chunks(chunks, pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-12-01", tz="UTC"))
    assert selected == [pd.Timestamp("2025-11-01", tz="UTC")]


# ---------------------------------------------------------------------
# classify_match
# ---------------------------------------------------------------------
def test_classify_match_matches_seq1():
    assert smard.classify_match(smard_price=100.00, price_seq1=100.00, price_seq2=120.00) == "matches_seq1"


def test_classify_match_matches_seq2():
    assert smard.classify_match(smard_price=120.00, price_seq1=100.00, price_seq2=120.00) == "matches_seq2"


def test_classify_match_within_tolerance_counts_as_match():
    assert smard.classify_match(smard_price=100.005, price_seq1=100.00, price_seq2=120.00) == "matches_seq1"


def test_classify_match_matches_neither():
    assert smard.classify_match(smard_price=999.0, price_seq1=100.00, price_seq2=120.00) == "matches_neither"


def test_classify_match_missing_smard_data():
    assert smard.classify_match(smard_price=float("nan"), price_seq1=100.00, price_seq2=120.00) == "no_smard_data"


def test_classify_match_both_within_tolerance_is_ambiguous():
    assert smard.classify_match(smard_price=100.0, price_seq1=100.0, price_seq2=100.005) == "ambiguous_both_match"


# ---------------------------------------------------------------------
# compare_against_sequences
# ---------------------------------------------------------------------
def test_compare_against_sequences_end_to_end():
    disagreeing = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2025-10-01T00:00:00Z", "2025-10-01T00:15:00Z", "2025-10-01T00:30:00Z"]),
        "price_seq1": [100.0, 50.0, 30.0],
        "price_seq2": [110.0, 55.0, 32.0],
    })
    smard_df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2025-10-01T00:00:00Z", "2025-10-01T00:15:00Z"]),  # missing the third row on purpose
        "price_eur_mwh": [100.0, 55.0],
    })
    result = smard.compare_against_sequences(smard_df, disagreeing)
    assert list(result["match"]) == ["matches_seq1", "matches_seq2", "no_smard_data"]


def test_compare_against_sequences_preserves_row_count():
    disagreeing = pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-10-01", periods=10, freq="15min", tz="UTC"),
        "price_seq1": range(10), "price_seq2": range(10, 20),
    })
    smard_df = pd.DataFrame({"timestamp_utc": pd.date_range("2025-10-01", periods=5, freq="15min", tz="UTC"), "price_eur_mwh": range(5)})
    result = smard.compare_against_sequences(smard_df, disagreeing)
    assert len(result) == 10  # every disagreeing row present, regardless of SMARD coverage


# ---------------------------------------------------------------------
# Network-layer orchestration, tested against a mocked session matching
# the DOCUMENTED SMARD response schema exactly (bundesAPI/smard-api's
# openapi.yaml), not a guessed shape.
# ---------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


class _FakeSession:
    """Simulates the two SMARD endpoints exactly per the documented
    schema: index_quarterhour.json -> {"timestamps": [...]},
    {filter}_{region}_quarterhour_{ts}.json -> {"series": [[ts, val], ...]}.
    """
    def __init__(self, index_timestamps_ms, chunks_by_start_ms):
        self.index_timestamps_ms = index_timestamps_ms
        self.chunks_by_start_ms = chunks_by_start_ms
        self.requested_urls = []

    def get(self, url, timeout=30):
        self.requested_urls.append(url)
        if "index_" in url:
            return _FakeResponse({"timestamps": self.index_timestamps_ms})
        for start_ms, series in self.chunks_by_start_ms.items():
            if f"_{start_ms}." in url:
                return _FakeResponse({"meta_data": {"version": 1, "created": 0}, "series": series})
        raise AssertionError(f"Unexpected URL in fake session: {url}")


def test_fetch_smard_chunk_index_parses_documented_schema():
    session = _FakeSession(index_timestamps_ms=[1727740800000, 1730419200000], chunks_by_start_ms={})
    result = smard.fetch_smard_chunk_index(session=session)
    assert result == sorted([pd.Timestamp(1727740800000, unit="ms", tz="UTC"), pd.Timestamp(1730419200000, unit="ms", tz="UTC")])


def test_fetch_smard_chunk_drops_null_prices():
    """SMARD publishes null for not-yet-available intervals (see
    bundesAPI/smard-api issue #21) -- these must be dropped, not
    coerced to 0 or forward-filled.
    """
    start_ms = 1727740800000
    session = _FakeSession(
        index_timestamps_ms=[start_ms],
        chunks_by_start_ms={start_ms: [[start_ms, 100.0], [start_ms + 900000, None], [start_ms + 1800000, 105.0]]},
    )
    result = smard.fetch_smard_chunk(pd.Timestamp(start_ms, unit="ms", tz="UTC"), session=session)
    assert len(result) == 2  # the null row dropped
    assert list(result["price_eur_mwh"]) == [100.0, 105.0]


def test_fetch_smard_day_ahead_price_end_to_end_orchestration():
    """Full flow: index -> select relevant chunk -> fetch -> clip to
    the exact requested window.
    """
    chunk_start_ms = 1727740800000  # 2024-10-01T00:00:00Z
    ts0 = chunk_start_ms
    ts1 = chunk_start_ms + 900000    # +15 min
    ts2 = chunk_start_ms + 1800000   # +30 min
    ts3 = chunk_start_ms + 2700000   # +45 min, outside our requested window
    session = _FakeSession(
        index_timestamps_ms=[chunk_start_ms],
        chunks_by_start_ms={chunk_start_ms: [[ts0, 100.0], [ts1, 101.0], [ts2, 102.0], [ts3, 103.0]]},
    )
    start_utc = pd.Timestamp(ts0, unit="ms", tz="UTC")
    end_utc = pd.Timestamp(ts2, unit="ms", tz="UTC")  # exclusive -- should NOT include ts2 or ts3
    result = smard.fetch_smard_day_ahead_price(start_utc, end_utc, session=session)
    assert len(result) == 2
    assert list(result["price_eur_mwh"]) == [100.0, 101.0]


def test_fetch_smard_day_ahead_price_raises_when_no_chunks_available():
    session = _FakeSession(index_timestamps_ms=[], chunks_by_start_ms={})
    with pytest.raises(ValueError, match="No SMARD chunks found"):
        smard.fetch_smard_day_ahead_price(
            pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-10-02", tz="UTC"), session=session
        )
