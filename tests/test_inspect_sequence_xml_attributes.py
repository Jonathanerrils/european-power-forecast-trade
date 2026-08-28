"""Network-free regression tests for scripts/inspect_sequence_xml_attributes.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "scripts"))

try:
    import scripts.inspect_sequence_xml_attributes as diag
except ModuleNotFoundError:
    # Allows this file to be run from the repository's tests/ directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import scripts.inspect_sequence_xml_attributes as diag

NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"


def _period(start: str, end: str, resolution: str = "PT15M") -> str:
    return f"""
    <Period>
      <timeInterval><start>{start}</start><end>{end}</end></timeInterval>
      <resolution>{resolution}</resolution>
      <Point><position>1</position><price.amount>50.0</price.amount></Point>
    </Period>
    """


def _timeseries(seq: int, periods: list[str], *, business_type: str = "A62") -> str:
    return f"""
    <TimeSeries>
      <mRID>ts-{seq}</mRID>
      <businessType>{business_type}</businessType>
      <classificationSequence_AttributeInstanceComponent.position>{seq}</classificationSequence_AttributeInstanceComponent.position>
      <curveType>A01</curveType>
      {''.join(periods)}
    </TimeSeries>
    """


def _document(*timeseries: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <Publication_MarketDocument xmlns="{NS}">
      <mRID>doc-1</mRID>
      <type>A44</type>
      {''.join(timeseries)}
    </Publication_MarketDocument>
    """


def test_multiple_pt15m_periods_in_one_timeseries_do_not_create_false_ambiguity(capsys):
    seq1 = _timeseries(
        1,
        [
            _period("2025-09-30T22:00Z", "2025-10-01T10:00Z"),
            _period("2025-10-01T10:00Z", "2025-10-01T22:00Z"),
        ],
    )
    seq2 = _timeseries(
        2,
        [_period("2025-09-30T22:00Z", "2025-10-01T22:00Z")],
    )

    diag.dump_and_diff_timeseries(_document(seq1, seq2), "2025-10-01")
    out = capsys.readouterr().out

    assert "AMBIGUOUS" not in out
    assert "2 overlapping PT15M Period(s)" in out
    assert "OTHER STRUCTURAL DIFFERENCES" in out


def test_two_distinct_pt15m_timeseries_for_same_sequence_are_ambiguous(capsys):
    seq1_a = _timeseries(1, [_period("2025-09-30T22:00Z", "2025-10-01T22:00Z")])
    seq1_b = _timeseries(1, [_period("2025-09-30T22:00Z", "2025-10-01T22:00Z")])
    seq2 = _timeseries(2, [_period("2025-09-30T22:00Z", "2025-10-01T22:00Z")])

    diag.dump_and_diff_timeseries(_document(seq1_a, seq1_b, seq2), "2025-10-01")
    out = capsys.readouterr().out

    assert "AMBIGUOUS: more than one distinct PT15M TimeSeries block" in out
    assert "sequence 1: TimeSeries indices [0, 1]" in out


def test_unexpected_sequence_is_reported_but_excluded_from_diff(capsys):
    seq1 = _timeseries(1, [_period("2025-09-30T22:00Z", "2025-10-01T22:00Z")])
    seq2 = _timeseries(2, [_period("2025-09-30T22:00Z", "2025-10-01T22:00Z")])
    seq3 = _timeseries(3, [_period("2025-09-30T22:00Z", "2025-10-01T22:00Z")])

    diag.dump_and_diff_timeseries(_document(seq1, seq2, seq3), "2025-10-01")
    out = capsys.readouterr().out

    assert "unexpected classificationSequence=3" in out
    assert "OTHER STRUCTURAL DIFFERENCES" in out
    assert "AMBIGUOUS" not in out
