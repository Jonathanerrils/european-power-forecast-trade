"""Run locally: python scripts/inspect_2025_price_xml.py

The 2019 chunk showed a clean PT15M/PT60M pair per day. Post-cutover
data shows TWO PT15M series colliding with DIFFERENT prices for the
same timestamp. This inspects the raw cached XML for the chunk
covering 2025-09-30 -> 2025-10-01 and prints every distinguishing
attribute (businessType, curveType, contract type, mRID, revision,
any other identifying element) for every TimeSeries whose period
touches that window, so we can see what actually separates the two
colliding series.
"""
import json
import hashlib
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "data" / "raw" / "ingestion_log.jsonl"
CACHE_DIR = REPO_ROOT / "data" / "raw" / "cache"


def cache_key(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def main():
    records = [json.loads(l) for l in open(LOG_PATH)]
    price_records = [r for r in records if r["data_type"] == "day_ahead_price"]

    # Find the chunk whose window covers 2025-09-30
    target = pd.Timestamp("2025-09-30", tz="UTC")
    chunk = None
    for r in price_records:
        start = pd.Timestamp(r["requested_start"])
        end = pd.Timestamp(r["requested_end"])
        if start <= target < end:
            chunk = r
            break

    if chunk is None:
        print("Could not find a chunk covering 2025-09-30 in the ingestion log.")
        print("Available chunks:")
        for r in price_records:
            print(f"  {r['requested_start']} -> {r['requested_end']}")
        return

    params = chunk["request_params"]
    path = CACHE_DIR / f"{cache_key(params)}.xml"
    print(f"Inspecting: {path}")
    print(f"Chunk window: {chunk['requested_start']} -> {chunk['requested_end']}\n")

    xml_text = path.read_text()
    root = ET.fromstring(xml_text)
    ns = root.tag.split("}")[0].strip("{")
    ns_map = {"ns": ns}

    # Print document-level metadata first
    print("Document-level fields:")
    for tag in ["mRID", "revisionNumber", "type", "createdDateTime", "sender_MarketParticipant.mRID"]:
        el = root.find(f"ns:{tag}", ns_map)
        print(f"  {tag}: {el.text if el is not None else None}")
    print()

    all_ts = root.findall("ns:TimeSeries", ns_map)
    print(f"Total TimeSeries in this document: {len(all_ts)}\n")

    window_start = pd.Timestamp("2025-09-30T00:00:00Z")
    window_end = pd.Timestamp("2025-10-02T00:00:00Z")

    match_count = 0
    for i, ts in enumerate(all_ts):
        periods = ts.findall(".//ns:Period", ns_map)
        if not periods:
            continue
        period_start_text = periods[0].find("ns:timeInterval/ns:start", ns_map).text
        period_start = pd.Timestamp(period_start_text.replace("Z", "+00:00"))

        if not (window_start <= period_start <= window_end):
            continue

        match_count += 1

        def get(tag, parent=ts):
            el = parent.find(f"ns:{tag}", ns_map)
            return el.text if el is not None else None

        # Print EVERY direct child element of this TimeSeries, not just
        # the ones we assumed mattered last time -- we were wrong once
        # already about what distinguishes series.
        print(f"--- TimeSeries[{i}] (period_start={period_start}) ---")
        for child in ts:
            tag_name = child.tag.split("}")[-1]
            if tag_name == "Period":
                continue  # printed separately below
            print(f"  {tag_name}: {child.text}")

        resolution = periods[0].find("ns:resolution", ns_map).text
        n_points = len(periods[0].findall("ns:Point", ns_map))
        sample = [
            (p.find("ns:position", ns_map).text,
             (p.find("ns:price.amount", ns_map).text if p.find("ns:price.amount", ns_map) is not None else None))
            for p in periods[0].findall("ns:Point", ns_map)[:4]
        ]
        print(f"  resolution: {resolution}, n_points: {n_points}")
        print(f"  sample points (position, price): {sample}")
        print()

    print(f"Matched {match_count} TimeSeries in the 2025-09-30 -> 2025-10-02 window.")


if __name__ == "__main__":
    main()
