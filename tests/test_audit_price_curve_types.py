"""Regression tests for scripts/audit_price_curve_types.py.

The most important invariant is provenance: the audit must replay exact
production cache keys and must never fetch fresh ENTSO-E data.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scripts.audit_price_curve_types as audit
from src.clean import local_delivery_date_to_utc


A44_XML_A03 = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <mRID>doc-1</mRID>
  <TimeSeries>
    <mRID>1</mRID>
    <curveType>A03</curveType>
    <classificationSequence_AttributeInstanceComponent.position>1</classificationSequence_AttributeInstanceComponent.position>
    <Period>
      <timeInterval><start>2024-12-31T23:00Z</start><end>2025-01-01T00:00Z</end></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><price.amount>50.0</price.amount></Point>
      <Point><position>3</position><price.amount>60.0</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
"""


A44_XML_DISJOINT = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <mRID>doc-2</mRID>
  <TimeSeries>
    <mRID>7</mRID>
    <curveType>A01</curveType>
    <Period>
      <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T01:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>40.0</price.amount></Point>
    </Period>
    <Period>
      <timeInterval><start>2024-01-01T02:00Z</start><end>2024-01-01T03:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>45.0</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
"""


class _FakeClient:
    def __init__(self, cache_dir: Path, xml_by_key: dict[str, str] | None = None, chunk_days: int = 365):
        self.cache_dir = cache_dir
        self.eic_code = "10Y1001A1001A82H"
        self.chunk_days = chunk_days
        self.xml_by_key = xml_by_key or {}
        self.request_calls = 0

    @staticmethod
    def _fmt(dt):
        return dt.strftime("%Y%m%d%H%M")

    def _chunk_windows(self, start, end):
        cur = start
        step = timedelta(days=self.chunk_days)
        while cur < end:
            nxt = min(cur + step, end)
            yield cur, nxt
            cur = nxt

    def _cache_path(self, params):
        key = f"{params['periodStart']}_{params['periodEnd']}_{params['documentType']}.xml"
        return self.cache_dir / key

    def _request(self, params):  # pragma: no cover - any call is a test failure
        self.request_calls += 1
        raise AssertionError("audit must never call _request/network")

    def _parse_timeseries(self, xml_text, value_tag="price.amount"):
        # Use a tiny production-compatible parser result for audit orchestration tests.
        if "curveType>A03" in xml_text:
            start = pd.Timestamp("2024-12-31T23:00:00Z").to_pydatetime()
            return [
                (start, 15, 50.0, None, 1, "A03", False),
                (start + timedelta(minutes=15), 15, 50.0, None, 1, "A03", True),
                (start + timedelta(minutes=30), 15, 60.0, None, 1, "A03", False),
                (start + timedelta(minutes=45), 15, 60.0, None, 1, "A03", True),
            ]
        if "curveType>A01" in xml_text:
            a = pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime()
            b = pd.Timestamp("2024-01-01T02:00:00Z").to_pydatetime()
            return [
                (a, 60, 40.0, None, None, "A01", False),
                (b, 60, 45.0, None, None, "A01", False),
            ]
        raise ValueError("simulated malformed document")


def _write_exact_cache(client: _FakeClient, start, end, xml_text: str) -> Path:
    params = audit.build_a44_cache_params(client, start, end)
    path = client._cache_path(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml_text, encoding="utf-8")
    return path


def test_exact_production_cache_key_is_used_and_network_is_never_called(tmp_path):
    client = _FakeClient(tmp_path)
    start = local_delivery_date_to_utc("2024-01-01")
    end = start + timedelta(days=365)
    expected = _write_exact_cache(client, start, end, A44_XML_A03)

    xml, path, params = audit.read_exact_cached_a44_chunk(client, start, end)

    assert path == expected
    assert xml == A44_XML_A03
    assert params["periodStart"] == client._fmt(start)
    assert params["periodEnd"] == client._fmt(end)
    assert params["documentType"] == "A44"
    assert client.request_calls == 0


def test_missing_exact_cache_chunk_is_reported_not_downloaded(tmp_path):
    client = _FakeClient(tmp_path)
    start = local_delivery_date_to_utc("2024-01-01")
    end = start + timedelta(days=1)

    with pytest.raises(FileNotFoundError, match="cache-only"):
        audit.read_exact_cached_a44_chunk(client, start, end)
    assert client.request_calls == 0


def test_audit_uses_production_chunk_boundaries_and_marks_manifest_incomplete(tmp_path):
    client = _FakeClient(tmp_path, chunk_days=2)
    start = local_delivery_date_to_utc("2024-01-01")
    end = start + timedelta(days=4)
    chunks = list(client._chunk_windows(start, end))
    _write_exact_cache(client, *chunks[0], A44_XML_DISJOINT)
    # chunks[1] deliberately missing

    summary, parsed, structural, issues = audit.audit_cached_history(
        client, start, end, "2025-10-01", tmp_path / "out"
    )

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "MISSING_CACHE"
    assert client.request_calls == 0
    manifest = json.loads((tmp_path / "out" / "curve_type_audit_manifest.json").read_text())
    assert manifest["expected_cache_chunks"] == 2
    assert manifest["missing_cache_chunks"] == 1
    assert manifest["complete"] is False
    assert not summary.empty and not parsed.empty and not structural.empty


def test_local_delivery_year_is_derived_row_by_row_not_from_utc_year():
    client = _FakeClient(Path("."))
    parsed = audit.parse_cached_prices_with_production_decoder(client, A44_XML_A03)
    # 2024-12-31 23:00 UTC = 2025-01-01 00:00 Europe/Berlin.
    assert set(parsed["year"]) == {2025}


def test_structural_scan_counts_timeseries_periods_points_and_disjoint_gap():
    structural = audit.scan_xml_structure(A44_XML_DISJOINT, "cache.xml")
    assert len(structural) == 2
    assert structural["timeseries_uid"].nunique() == 1
    assert structural["explicit_points_xml"].sum() == 2
    assert int(structural["disjoint_gap_before"].sum()) == 1
    summary = audit.build_summary(pd.DataFrame(), structural)
    row = summary.iloc[0]
    assert row["timeseries_count"] == 1
    assert row["period_count"] == 2
    assert row["disjoint_gap_count"] == 1


def test_pre_cutover_detection_uses_actual_cutover_not_only_calendar_year():
    # A03 PT60M on 2025-09-30 must count as PRE-cutover even though it is in cutover year 2025.
    parsed = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2025-09-30T20:00:00Z", "2025-09-30T22:00:00Z"], utc=True
            ),
            "resolution_min": [60, 60],
            "curve_type": ["A03", "A01"],
            "price_eur_mwh": [50.0, 60.0],
            "is_synthesized": [False, False],
        }
    )
    curve_types = audit.pre_cutover_pt60m_curve_types(parsed, "2025-10-01")
    assert curve_types == ["A03"]


def test_parse_failure_isolated_to_one_production_cache_chunk(tmp_path):
    client = _FakeClient(tmp_path, chunk_days=1)
    start = local_delivery_date_to_utc("2024-01-01")
    end = start + timedelta(days=2)
    chunks = list(client._chunk_windows(start, end))
    _write_exact_cache(client, *chunks[0], "<not-valid-enough-for-parser>")
    _write_exact_cache(client, *chunks[1], A44_XML_DISJOINT)

    summary, parsed, structural, issues = audit.audit_cached_history(
        client, start, end, "2025-10-01", tmp_path / "out"
    )

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "PARSE_FAILURE"
    assert not summary.empty
    assert not parsed.empty
    assert not structural.empty
    assert client.request_calls == 0


# ---------------------------------------------------------------------
# run_version versioning (regression guard for the same class of bug
# already found and fixed in run_eda.py)
# ---------------------------------------------------------------------
def test_run_version_is_mandatory():
    default_start = pd.Timestamp("2019-01-01", tz="UTC")
    default_end = pd.Timestamp("2026-01-01", tz="UTC")
    with pytest.raises(SystemExit):
        audit.resolve_run_args([], default_start, default_end)


def test_run_version_with_default_dates():
    default_start = pd.Timestamp("2019-01-01", tz="UTC")
    default_end = pd.Timestamp("2026-01-01", tz="UTC")
    run_version, start, end = audit.resolve_run_args(["a03fix_v1"], default_start, default_end)
    assert run_version == "a03fix_v1"
    assert start == default_start
    assert end == default_end


def test_run_version_with_explicit_dates():
    default_start = pd.Timestamp("2019-01-01", tz="UTC")
    default_end = pd.Timestamp("2026-01-01", tz="UTC")
    run_version, start, end = audit.resolve_run_args(
        ["a03fix_v1", "2020-01-01", "2021-01-01"], default_start, default_end
    )
    assert run_version == "a03fix_v1"
    assert start == local_delivery_date_to_utc("2020-01-01")
    assert end == local_delivery_date_to_utc("2021-01-01")


def test_start_after_end_rejected():
    default_start = pd.Timestamp("2019-01-01", tz="UTC")
    default_end = pd.Timestamp("2026-01-01", tz="UTC")
    with pytest.raises(ValueError, match="must precede end"):
        audit.resolve_run_args(["a03fix_v1", "2021-01-01", "2020-01-01"], default_start, default_end)


def test_too_many_arguments_rejected():
    default_start = pd.Timestamp("2019-01-01", tz="UTC")
    default_end = pd.Timestamp("2026-01-01", tz="UTC")
    with pytest.raises(SystemExit):
        audit.resolve_run_args(["a", "b", "c", "d"], default_start, default_end)
