"""Run locally: python scripts/inspect_sequence_xml_attributes.py [delivery_date]

The auction-sequence audit found something much bigger than a rare
edge-case collision. Independently verified against the real uploaded
CSVs: 96.8% of ALL PT15M intervals across the entire Oct 1 - Dec 31 2025
window disagree between sequence 1 and sequence 2 (8,553 of 8,836
possible intervals), median gap EUR 7.04/MWh, mean EUR 11.28/MWh, max
EUR 278.69/MWh (2025-10-01 18:15 local: 200.00 vs 478.69). That pattern
is not consistent with "two near-identical resubmissions of the same
auction" -- it looks like sequence 1 and sequence 2 may be structurally
DIFFERENT things entirely (two different auction products/mechanisms),
not two candidate values for the same one.

_parse_timeseries() in entsoe_client.py currently extracts only a
handful of fields per TimeSeries block (psr_type, classificationSequence
position, timestamps, values). ENTSO-E's A44 schema carries more
TimeSeries-level attributes that might explain what actually
distinguishes these two sequences -- businessType, curveType,
Auction.mRID/type, contract_MarketAgreement.type, etc. This script dumps
every field for the requested delivery day's TimeSeries blocks, then
prints an explicit diff of which fields actually differ between
sequence 1 and sequence 2, so the answer isn't hidden in a wall of
identical repeated output.

CRITICAL PROVENANCE REQUIREMENT: this reads ONLY the exact cached XML
that verify_auction_sequence.py's audit already used to produce the
disagreement findings above -- never a fresh API request. With
config.yaml's chunk_days=365, the audit's ~92-day window
(2025-10-01 to 2026-01-01) was ONE chunk under ONE cache key; a
single-day request has different periodStart/periodEnd and therefore a
DIFFERENT cache key, which would silently pull fresh data that could
theoretically reflect a different publication/revision state on
ENTSO-E's servers than what actually produced this finding. This script
hard-pins the request window to match the audit exactly and refuses to
proceed (raises, does not silently fetch) if that exact cache file
isn't present.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.entsoe_client import EntsoeClient
from src.clean import local_delivery_date_to_utc
from src.utils import setup_logging, load_config

# Hard-pinned to match verify_auction_sequence.py's audit window EXACTLY.
# Do not parametrize this to an arbitrary date range -- the whole point
# is reading the identical cached XML the audit already used.
AUDIT_WINDOW_START = "2025-10-01"
AUDIT_WINDOW_END = "2026-01-01"


def get_audit_cache_xml(client: EntsoeClient) -> str:
    """Reads ONLY the exact cache file the audit's fetch_both_sequences()
    call produced. Raises FileNotFoundError rather than silently making
    a new API request -- provenance matters here specifically because
    we want to inspect the XML that produced the disagreement findings,
    not whatever ENTSO-E returns right now.
    """
    start = local_delivery_date_to_utc(AUDIT_WINDOW_START)
    end = local_delivery_date_to_utc(AUDIT_WINDOW_END)
    params_extra = {
        "documentType": "A44",
        "in_Domain": client.eic_code,
        "out_Domain": client.eic_code,
    }
    # Replicates _fetch_generic's exact param construction for the single
    # chunk this window produces (chunk_days=365 > ~92 day window).
    chunks = list(client._chunk_windows(start, end))
    if len(chunks) != 1:
        raise AssertionError(
            f"Expected exactly one chunk for the audit window with chunk_days="
            f"{client.chunk_days}, got {len(chunks)}. The audit window or "
            f"chunk_days config may have changed -- re-verify before proceeding."
        )
    chunk_start, chunk_end = chunks[0]
    params = {
        "periodStart": client._fmt(chunk_start),
        "periodEnd": client._fmt(chunk_end),
        **params_extra,
    }
    cache_path = client._cache_path(params)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Exact audit cache file not found: {cache_path}\n"
            f"Run verify_auction_sequence.py first (against the real ENTSO-E data) -- "
            f"this diagnostic must read the SAME cached XML the audit used, not a "
            f"fresh request, and will not fetch on its own."
        )
    print(f"Reading exact audit cache file: {cache_path.name}")
    return cache_path.read_text()


def _flatten_element(el, prefix=""):
    """Returns a list of (path, text) pairs -- NOT a dict -- so repeated
    XML element names (which a dict comprehension would silently
    overwrite, keeping only the last one) are all preserved. Order
    matches document order.
    """
    tag = el.tag.split("}")[-1]
    path = f"{prefix}.{tag}" if prefix else tag
    pairs = []
    children = list(el)
    if not children:
        if el.text and el.text.strip():
            pairs.append((path, el.text.strip()))
        return pairs
    for child in children:
        pairs.extend(_flatten_element(child, prefix=path))
    return pairs


def dump_and_diff_timeseries(xml_text: str, target_date: str) -> None:
    root = ET.fromstring(xml_text)
    ns = root.tag.split("}")[0].strip("{")
    ns_map = {"ns": ns}

    # DST-safe local delivery-day boundaries -- NOT start + timedelta(days=1),
    # which is wrong on the 23h/25h DST-transition days.
    day_start = local_delivery_date_to_utc(target_date)
    next_date = (pd.Timestamp(target_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    day_end = local_delivery_date_to_utc(next_date)

    print("\n" + "#" * 78)
    print("DOCUMENT-LEVEL FIELDS")
    print("#" * 78)
    for child in root:
        tag = child.tag.split("}")[-1]
        if tag == "TimeSeries":
            continue
        for path, text in _flatten_element(child):
            print(f"  {path}: {text}")

    # Grouped by sequence, each entry keeps ALL matching *distinct* TimeSeries
    # blocks -- never collapsed to "the last one seen". A single TimeSeries may
    # legitimately contain multiple Period elements, so Period count must NOT be
    # mistaken for TimeSeries count. This project also has a documented history
    # of PT60M/PT15M dual-product collisions (select_price_resolution() exists
    # specifically because of one), so ONLY TimeSeries with at least one
    # overlapping PT15M Period are eligible for the sequence-1-vs-2 diff.
    matched_by_seq: dict = {}
    for i, ts in enumerate(root.findall("ns:TimeSeries", ns_map)):
        seq_el = ts.find("ns:classificationSequence_AttributeInstanceComponent.position", ns_map)
        seq = int(seq_el.text) if seq_el is not None else None

        # This diagnostic is explicitly about the observed sequence-1/sequence-2
        # collision. Unexpected or missing sequence identifiers are surfaced but
        # never allowed to contaminate that comparison.
        if seq not in (1, 2):
            print(
                f"\nNOTE: TimeSeries #{i} has unexpected classificationSequence={seq}; "
                f"not used in the sequence-1-vs-sequence-2 diff."
            )
            continue

        eligible_periods = []
        for period in ts.findall(".//ns:Period", ns_map):
            res_el = period.find("ns:resolution", ns_map)
            resolution = res_el.text if res_el is not None else None
            p_start_el = period.find("ns:timeInterval/ns:start", ns_map)
            p_end_el = period.find("ns:timeInterval/ns:end", ns_map)

            if p_start_el is None or p_end_el is None or not p_start_el.text or not p_end_el.text:
                print(
                    f"\nNOTE: TimeSeries #{i} (sequence {seq}) has a Period with missing "
                    f"timeInterval boundaries -- SKIPPED for this diagnostic."
                )
                continue

            p_start_text = p_start_el.text
            p_end_text = p_end_el.text
            p_start = pd.Timestamp(p_start_text.replace("Z", "+00:00"))
            p_end = pd.Timestamp(p_end_text.replace("Z", "+00:00"))
            overlaps = not (p_end <= day_start or p_start >= day_end)
            if not overlaps:
                continue

            if resolution != "PT15M":
                print(
                    f"\nNOTE: TimeSeries #{i} (sequence {seq}) has a {resolution} Period "
                    f"overlapping {target_date} -- SKIPPED. Only PT15M blocks are eligible "
                    f"for the sequence-1-vs-2 comparison; comparing across resolutions would "
                    f"contaminate the diff."
                )
                continue

            eligible_periods.append((period, p_start_text, p_end_text))

        if not eligible_periods:
            continue

        print(f"\n{'=' * 78}")
        print(
            f"TimeSeries #{i} -- classificationSequence position = {seq} "
            f"({len(eligible_periods)} overlapping PT15M Period(s))"
        )
        print("=" * 78)
        for j, (period, p_start_text, p_end_text) in enumerate(eligible_periods, start=1):
            n_points = len(period.findall("ns:Point", ns_map))
            print(
                f"  Eligible PT15M Period #{j}: {p_start_text} -> {p_end_text}, "
                f"n_points={n_points}"
            )

        # Extract TimeSeries-level metadata ONCE per distinct TimeSeries. Multiple
        # Periods inside the same TimeSeries are not multiple candidate series.
        fields = []  # list of (path, value) -- repeats preserved, NOT collapsed to a dict
        for child in ts:
            tag = child.tag.split("}")[-1]
            if tag == "Period":
                continue  # per-interval prices already covered by the normal pipeline
            for path, value in _flatten_element(child):
                print(f"  {path}: {value}")
                fields.append((path, value))

        matched_by_seq.setdefault(seq, []).append((i, fields))

    # If more than one DISTINCT PT15M TimeSeries block matches the same sequence
    # for this day, do NOT silently pick one. Print their indices and stop; this
    # is a genuine ambiguity, unlike multiple Periods within one TimeSeries.
    ambiguous = {seq: blocks for seq, blocks in matched_by_seq.items() if len(blocks) > 1}
    if ambiguous:
        print("\n" + "!" * 78)
        print(
            "AMBIGUOUS: more than one distinct PT15M TimeSeries block matches the same "
            "sequence for this day. Refusing to guess which is correct -- listed above. "
            "Resolve this manually before trusting any diff below."
        )
        for seq, blocks in ambiguous.items():
            print(f"  sequence {seq}: TimeSeries indices {[i for i, _ in blocks]}")
        print("!" * 78)
        return

    if 1 in matched_by_seq and 2 in matched_by_seq:
        _print_diff(matched_by_seq[1][0][1], matched_by_seq[2][0][1])
    elif 1 in matched_by_seq:
        print(
            f"\nOnly sequence 1 has a PT15M TimeSeries block overlapping {target_date} "
            f"in this cached XML -- cannot diff sequence 1 vs sequence 2 for this date."
        )
    elif 2 in matched_by_seq:
        print(
            f"\nOnly sequence 2 has a PT15M TimeSeries block overlapping {target_date} "
            f"in this cached XML -- cannot diff sequence 1 vs sequence 2 for this date."
        )
    else:
        print(f"\nNo eligible sequence-1/sequence-2 PT15M TimeSeries blocks overlap {target_date} in this cached XML.")


def _print_diff(fields_seq1: list, fields_seq2: list) -> None:
    """Compares two (path, value) lists as MULTIMAPS (path -> list of
    values), not as dicts -- a dict would silently drop a repeated path's
    earlier value, exactly the collapsing bug this function exists to
    avoid. classificationSequence is reported separately from every other
    field: it's EXPECTED to differ (that's the whole point of there being
    two sequences), so burying it among "other structural differences"
    would make every date's diff spuriously non-empty and hide whether
    anything ELSE actually differs.
    """
    def to_multimap(fields):
        mm = defaultdict(list)
        for path, value in fields:
            mm[path].append(value)
        return dict(mm)

    seq1_mm = to_multimap(fields_seq1)
    seq2_mm = to_multimap(fields_seq2)
    all_keys = sorted(set(seq1_mm.keys()) | set(seq2_mm.keys()))

    seq_field = "classificationSequence_AttributeInstanceComponent.position"
    other_diffs = {}
    for key in all_keys:
        if key == seq_field:
            continue
        v1 = seq1_mm.get(key, ["<absent>"])
        v2 = seq2_mm.get(key, ["<absent>"])
        if v1 != v2:
            other_diffs[key] = (v1, v2)

    print("\n" + "#" * 78)
    print("EXPECTED DIFFERENCE (this is supposed to differ)")
    print("#" * 78)
    print(f"  {seq_field}:")
    print(f"    sequence 1: {seq1_mm.get(seq_field, ['<absent>'])}")
    print(f"    sequence 2: {seq2_mm.get(seq_field, ['<absent>'])}")

    print("\n" + "#" * 78)
    print("OTHER STRUCTURAL DIFFERENCES")
    print("(if this is empty, sequence 1 and 2 are IDENTICAL except for the")
    print("sequence designation itself -- the sequence field's real-world")
    print("meaning becomes the key thing to research in ENTSO-E documentation)")
    print("#" * 78)
    if not other_diffs:
        print("  <none -- every other field is identical between sequence 1 and 2>")
    else:
        for key, (v1, v2) in other_diffs.items():
            print(f"  {key}:")
            print(f"    sequence 1: {v1}")
            print(f"    sequence 2: {v2}")


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    delivery_date = sys.argv[1] if len(sys.argv) > 1 else "2025-10-01"

    client = EntsoeClient()
    xml_text = get_audit_cache_xml(client)  # raises if the exact cache isn't present -- never fetches fresh
    dump_and_diff_timeseries(xml_text, delivery_date)


if __name__ == "__main__":
    main()
