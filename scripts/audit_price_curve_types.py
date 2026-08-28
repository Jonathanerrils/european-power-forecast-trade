"""Run locally: python scripts/audit_price_curve_types.py

Whole-history, CACHE-ONLY audit of ENTSO-E DE-LU day-ahead price curveType
usage. The audit exists to answer whether the A03 under-parsing bug affected
only the post-2025-10-01 PT15M target regime or also the PT60M target used
before the cutover.

Provenance rule (non-negotiable)
--------------------------------
This script NEVER calls EntsoeClient._request() or _fetch_generic(). It
replays the exact production chunk boundaries produced by
client._chunk_windows(start, end), reconstructs the exact A44 cache key for
each chunk, and reads that cache file directly. If an expected cache file is
missing, the chunk is reported as MISSING_CACHE and is NOT downloaded.

Each cached XML document is then:
  1. structurally scanned for TimeSeries / Period / explicit Point counts and
     disjoint Period gaps; and
  2. passed through the production EntsoeClient._parse_timeseries() decoder,
     so A01/A02/A03 handling is exactly the same code used by ingestion.

Year labels are derived row-by-row from Europe/Berlin delivery time, not from
UTC chunk boundaries. This matters around New Year because local delivery-day
midnight can fall on the previous UTC calendar date.

Outputs
-------
outputs/curve_type_audit/curve_type_summary_by_year.csv
outputs/curve_type_audit/curve_type_structural_periods.csv
outputs/curve_type_audit/curve_type_chunk_issues.csv
outputs/curve_type_audit/curve_type_audit_manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.entsoe_client import EntsoeClient, get_default_date_range
from src.clean import local_delivery_date_to_utc
from src.utils import setup_logging, load_config, REPO_ROOT

LOCAL_TZ = "Europe/Berlin"
DEFAULT_A01_LABEL = "A01(default)"


def make_cache_only_client(cfg: dict) -> EntsoeClient:
    """Construct only the EntsoeClient state needed for cache replay/parsing.

    Deliberately bypasses EntsoeClient.__init__ so the audit does not require an
    ENTSOE_TOKEN and cannot acquire request/retry configuration as a side effect.
    The methods used below (_chunk_windows, _fmt, _cache_path, _parse_timeseries)
    need only these fields.
    """
    client = EntsoeClient.__new__(EntsoeClient)
    client.cfg = cfg
    client.eic_code = cfg["market"]["eic_code"]
    client.chunk_days = cfg["entsoe"]["chunk_days"]
    client.cache_dir = REPO_ROOT / cfg["data"]["cache_dir"]
    return client


def build_a44_cache_params(client: EntsoeClient, chunk_start: datetime, chunk_end: datetime) -> dict:
    """Reconstruct the exact params used by production A44 ingestion.

    securityToken is intentionally absent because EntsoeClient._cache_key()
    hashes request params before the token is added.
    """
    return {
        "periodStart": client._fmt(chunk_start),
        "periodEnd": client._fmt(chunk_end),
        "documentType": "A44",
        "in_Domain": client.eic_code,
        "out_Domain": client.eic_code,
    }


def read_exact_cached_a44_chunk(
    client: EntsoeClient, chunk_start: datetime, chunk_end: datetime
) -> tuple[str, Path, dict]:
    """Read the exact production-cache A44 XML. Never fetch from network."""
    params = build_a44_cache_params(client, chunk_start, chunk_end)
    cache_path = client._cache_path(params)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Expected production cache file is missing: {cache_path} "
            f"({chunk_start.isoformat()} -> {chunk_end.isoformat()}). "
            "Audit is cache-only; refusing to fetch fresh ENTSO-E data."
        )
    return cache_path.read_text(), cache_path, params


def _parse_utc_text(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _resolution_minutes(text: str | None) -> int | None:
    return {"PT15M": 15, "PT30M": 30, "PT60M": 60}.get(text)


def scan_xml_structure(xml_text: str, cache_name: str) -> pd.DataFrame:
    """Scan raw XML structure without reimplementing price reconstruction.

    Returns one record per Period. `disjoint_gap_before` is True when the
    current Period starts strictly after the prior Period in the same
    TimeSeries. Adjacent Periods are not gaps. Overlaps are also surfaced.
    """
    root = ET.fromstring(xml_text)
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns_map = {"ns": ns} if ns else {}
    ts_path = "ns:TimeSeries" if ns else "TimeSeries"

    records: list[dict] = []
    for ts_index, ts in enumerate(root.findall(ts_path, ns_map)):
        prefix = "ns:" if ns else ""
        curve_el = ts.find(f"{prefix}curveType", ns_map)
        curve_raw = curve_el.text.strip() if curve_el is not None and curve_el.text else None
        curve_type = curve_raw or DEFAULT_A01_LABEL

        seq_el = ts.find(
            f"{prefix}classificationSequence_AttributeInstanceComponent.position", ns_map
        )
        auction_sequence = int(seq_el.text) if seq_el is not None and seq_el.text else None

        mrid_el = ts.find(f"{prefix}mRID", ns_map)
        mrid = mrid_el.text.strip() if mrid_el is not None and mrid_el.text else None
        ts_uid = f"{cache_name}:ts{ts_index}"

        period_path = f".//{prefix}Period"
        period_infos: list[dict] = []
        for period_index, period in enumerate(ts.findall(period_path, ns_map)):
            start_el = period.find(f"{prefix}timeInterval/{prefix}start", ns_map)
            end_el = period.find(f"{prefix}timeInterval/{prefix}end", ns_map)
            res_el = period.find(f"{prefix}resolution", ns_map)
            if (
                start_el is None
                or end_el is None
                or res_el is None
                or not start_el.text
                or not end_el.text
                or not res_el.text
            ):
                raise ValueError(
                    f"TimeSeries {ts_index} Period {period_index} in {cache_name} "
                    "is missing start/end/resolution metadata."
                )

            start_utc = _parse_utc_text(start_el.text)
            end_utc = _parse_utc_text(end_el.text)
            resolution_text = res_el.text.strip()
            resolution_min = _resolution_minutes(resolution_text)
            point_path = f"{prefix}Point"
            n_explicit_points = len(period.findall(point_path, ns_map))

            period_infos.append(
                {
                    "cache_file": cache_name,
                    "timeseries_uid": ts_uid,
                    "timeseries_index": ts_index,
                    "timeseries_mrid": mrid,
                    "auction_sequence": auction_sequence,
                    "curve_type": curve_type,
                    "period_index": period_index,
                    "period_start_utc": start_utc,
                    "period_end_utc": end_utc,
                    "resolution": resolution_text,
                    "resolution_min": resolution_min,
                    "explicit_points_xml": n_explicit_points,
                }
            )

        period_infos.sort(key=lambda r: (r["period_start_utc"], r["period_end_utc"]))
        prev_end: pd.Timestamp | None = None
        for rec in period_infos:
            rec["disjoint_gap_before"] = bool(prev_end is not None and rec["period_start_utc"] > prev_end)
            rec["overlap_with_previous"] = bool(prev_end is not None and rec["period_start_utc"] < prev_end)
            # A later overlapping period should not move the coverage frontier backward.
            if prev_end is None or rec["period_end_utc"] > prev_end:
                prev_end = rec["period_end_utc"]

            local_start = rec["period_start_utc"].tz_convert(LOCAL_TZ)
            rec["year"] = int(local_start.year)
            records.append(rec)

    return pd.DataFrame(records)


def parse_cached_prices_with_production_decoder(
    client: EntsoeClient, xml_text: str
) -> pd.DataFrame:
    """Parse one cached document using the production TimeSeries decoder."""
    rows = client._parse_timeseries(xml_text, value_tag="price.amount")
    cols = [
        "timestamp_utc",
        "resolution_min",
        "price_eur_mwh",
        "psr_type",
        "auction_sequence",
        "curve_type",
        "is_synthesized",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["curve_type"] = df["curve_type"].fillna(DEFAULT_A01_LABEL)
    df["is_synthesized"] = df["is_synthesized"].astype(bool)
    df["year"] = df["timestamp_utc"].dt.tz_convert(LOCAL_TZ).dt.year.astype(int)
    return df


def build_summary(parsed: pd.DataFrame, structural: pd.DataFrame) -> pd.DataFrame:
    keys = ["year", "resolution_min", "curve_type"]

    if parsed.empty:
        parsed_summary = pd.DataFrame(
            columns=keys
            + ["effective_intervals", "explicit_intervals", "synthesized_intervals", "pct_synthesized"]
        )
    else:
        parsed_summary = (
            parsed.groupby(keys, dropna=False)
            .agg(
                effective_intervals=("price_eur_mwh", "size"),
                explicit_intervals=("is_synthesized", lambda s: int((~s).sum())),
                synthesized_intervals=("is_synthesized", lambda s: int(s.sum())),
            )
            .reset_index()
        )
        parsed_summary["pct_synthesized"] = (
            parsed_summary["synthesized_intervals"]
            / parsed_summary["effective_intervals"]
            * 100
        ).round(2)

    if structural.empty:
        structural_summary = pd.DataFrame(
            columns=keys
            + [
                "timeseries_count",
                "period_count",
                "explicit_points_xml",
                "disjoint_gap_count",
                "overlapping_period_count",
            ]
        )
    else:
        structural_summary = (
            structural.groupby(keys, dropna=False)
            .agg(
                timeseries_count=("timeseries_uid", "nunique"),
                period_count=("period_index", "size"),
                explicit_points_xml=("explicit_points_xml", "sum"),
                disjoint_gap_count=("disjoint_gap_before", "sum"),
                overlapping_period_count=("overlap_with_previous", "sum"),
            )
            .reset_index()
        )

    summary = pd.merge(structural_summary, parsed_summary, on=keys, how="outer")
    return summary.sort_values(keys).reset_index(drop=True)


def pre_cutover_pt60m_curve_types(parsed: pd.DataFrame, cutover_date: str) -> list[str]:
    """Return curve types actually observed in PT60M rows before local cutover."""
    if parsed.empty:
        return []
    cutover_utc = pd.Timestamp(local_delivery_date_to_utc(cutover_date))
    subset = parsed[
        (parsed["timestamp_utc"] < cutover_utc)
        & (parsed["resolution_min"] == 60)
    ]
    return sorted(subset["curve_type"].dropna().astype(str).unique().tolist())


def audit_cached_history(
    client: EntsoeClient,
    start: datetime,
    end: datetime,
    cutover_date: str,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Audit exact production cache chunks. No network path exists here."""
    parsed_frames: list[pd.DataFrame] = []
    structural_frames: list[pd.DataFrame] = []
    issues: list[dict] = []
    chunks = list(client._chunk_windows(start, end))

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
        params = build_a44_cache_params(client, chunk_start, chunk_end)
        cache_path = client._cache_path(params)
        issue_base = {
            "chunk_index": chunk_index,
            "chunk_start": chunk_start.isoformat(),
            "chunk_end": chunk_end.isoformat(),
            "cache_file": cache_path.name,
        }

        try:
            xml_text, cache_path, _ = read_exact_cached_a44_chunk(client, chunk_start, chunk_end)
        except FileNotFoundError as exc:
            issues.append({**issue_base, "issue_type": "MISSING_CACHE", "error": str(exc)})
            continue

        try:
            structural = scan_xml_structure(xml_text, cache_path.name)
            parsed = parse_cached_prices_with_production_decoder(client, xml_text)
        except (ValueError, NotImplementedError, ET.ParseError) as exc:
            issues.append({**issue_base, "issue_type": "PARSE_FAILURE", "error": str(exc)})
            continue

        if not structural.empty:
            structural_frames.append(structural)
        if not parsed.empty:
            parsed["cache_file"] = cache_path.name
            parsed_frames.append(parsed)

    parsed_all = pd.concat(parsed_frames, ignore_index=True) if parsed_frames else pd.DataFrame()
    structural_all = (
        pd.concat(structural_frames, ignore_index=True) if structural_frames else pd.DataFrame()
    )
    summary = build_summary(parsed_all, structural_all)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "curve_type_summary_by_year.csv", index=False)
    structural_all.to_csv(out_dir / "curve_type_structural_periods.csv", index=False)
    issue_cols = ["chunk_index", "chunk_start", "chunk_end", "cache_file", "issue_type", "error"]
    pd.DataFrame(issues, columns=issue_cols).to_csv(
        out_dir / "curve_type_chunk_issues.csv", index=False
    )

    manifest = {
        "cache_only": True,
        "network_fetch_allowed": False,
        "audit_start": start.isoformat(),
        "audit_end": end.isoformat(),
        "expected_cache_chunks": len(chunks),
        "successful_cache_chunks": len(chunks) - len(issues),
        "missing_cache_chunks": sum(i["issue_type"] == "MISSING_CACHE" for i in issues),
        "parse_failed_chunks": sum(i["issue_type"] == "PARSE_FAILURE" for i in issues),
        "complete": len(issues) == 0,
        "pre_cutover_pt60m_curve_types": pre_cutover_pt60m_curve_types(parsed_all, cutover_date),
    }
    (out_dir / "curve_type_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return summary, parsed_all, structural_all, issues


def _print_results(
    summary: pd.DataFrame,
    parsed: pd.DataFrame,
    issues: list[dict],
    cutover_date: str,
    out_dir: Path,
) -> None:
    if issues:
        print("\n" + "!" * 78)
        print(f"AUDIT INCOMPLETE: {len(issues)} CACHE CHUNK ISSUE(S)")
        print("No fresh ENTSO-E data were fetched. Missing/failed chunks are excluded.")
        print("!" * 78)
        for issue in issues:
            print(
                f"  [{issue['issue_type']}] {issue['chunk_start']} -> {issue['chunk_end']} "
                f"{issue['cache_file']}: {issue['error']}"
            )
    else:
        print("\nAUDIT COMPLETE: every expected production cache chunk was found and parsed.")

    print("\n" + "=" * 78)
    print("CURVE TYPE USAGE BY LOCAL DELIVERY YEAR / RESOLUTION")
    print("=" * 78)
    if summary.empty:
        print("  <no successfully parsed data>")
    else:
        print(summary.to_string(index=False))

    curve_types = pre_cutover_pt60m_curve_types(parsed, cutover_date)
    print("\n" + "=" * 78)
    print(f"PRE-{cutover_date} PT60M CURVE TYPES")
    print("=" * 78)
    if not curve_types:
        print("  No pre-cutover PT60M rows were available in successfully parsed cache chunks.")
    else:
        print(f"  curveType(s) observed: {curve_types}")
        if "A03" in curve_types:
            print("  *** A03 IS present in pre-cutover PT60M target data.")
            print("  *** Therefore the old under-parsing bug affected target history before the cutover too.")
        else:
            print("  A03 not observed in pre-cutover PT60M target data in the audited cache.")

    print(f"\nSaved audit outputs to {out_dir}")


def resolve_run_args(args: list, default_start, default_end) -> tuple:
    """run_version is MANDATORY (matching the project's established
    pattern), start/end remain optional and default to the full
    configured history. An earlier version used a fixed, unversioned
    output path (outputs/curve_type_audit/), so re-running this audit
    against corrected/re-ingested data would silently overwrite the
    pre-A03-fix baseline. Versioning this BEFORE the corrected re-run
    means the pre-fix files (already on disk, at the old unversioned
    path) are left untouched rather than destroyed.
    """
    if len(args) not in (1, 2, 3):
        raise SystemExit(
            "Usage:\n"
            "  python scripts/audit_price_curve_types.py <run_version> [start] [end]\n\n"
            "Example:\n"
            "  python scripts/audit_price_curve_types.py a03fix_v1"
        )
    run_version = args[0]
    start = local_delivery_date_to_utc(args[1]) if len(args) > 1 else default_start
    end = local_delivery_date_to_utc(args[2]) if len(args) > 2 else default_end
    if start >= end:
        raise ValueError(f"Audit start must precede end: {start} >= {end}")
    return run_version, start, end


def main() -> None:
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])

    default_start, default_end = get_default_date_range(cfg)
    run_version, start, end = resolve_run_args(sys.argv[1:], default_start, default_end)

    client = make_cache_only_client(cfg)
    out_dir = REPO_ROOT / "outputs" / "curve_type_audit" / run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Pass a new run_version instead of "
            f"overwriting it, e.g.\n"
            f"  python scripts/audit_price_curve_types.py {run_version}_v2"
        )
    summary, parsed, _structural, issues = audit_cached_history(
        client=client,
        start=start,
        end=end,
        cutover_date=cfg["market_design"]["fifteen_min_mtu_start"],
        out_dir=out_dir,
    )
    _print_results(
        summary=summary,
        parsed=parsed,
        issues=issues,
        cutover_date=cfg["market_design"]["fifteen_min_mtu_start"],
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
