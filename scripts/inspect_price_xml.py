"""Run locally: python inspect_price_xml.py

Reads data/raw/ingestion_log.jsonl to find the price-fetch requests,
recomputes their cache file paths, and prints the structure of each
TimeSeries in the raw XML: businessType, curveType, contract type,
resolution, and point count. This tells us why price rows are ~3x
more numerous than expected (likely multiple TimeSeries per document
for different contract/curve types).
"""
import json
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
LOG_PATH = REPO_ROOT / "data" / "raw" / "ingestion_log.jsonl"
CACHE_DIR = REPO_ROOT / "data" / "raw" / "cache"


def cache_key(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def main():
    records = [json.loads(l) for l in open(LOG_PATH)]
    price_records = [r for r in records if r["data_type"] == "day_ahead_price"]
    print(f"Found {len(price_records)} price-fetch chunks in the ingestion log.\n")

    # Just inspect the first chunk in detail
    rec = price_records[0]
    params = rec["request_params"]
    path = CACHE_DIR / f"{cache_key(params)}.xml"
    print(f"Inspecting: {path}")
    print(f"Requested window: {rec['requested_start']} -> {rec['requested_end']}\n")

    if not path.exists():
        print("Cache file not found -- cache key mismatch, list cache dir instead:")
        for f in sorted(CACHE_DIR.glob("*.xml"))[:5]:
            print(" ", f.name)
        return

    xml_text = path.read_text()
    root = ET.fromstring(xml_text)
    ns = root.tag.split("}")[0].strip("{")
    ns_map = {"ns": ns}

    all_ts = root.findall("ns:TimeSeries", ns_map)
    print(f"Number of <TimeSeries> elements in this one document: {len(all_ts)}\n")

    for i, ts in enumerate(all_ts):
        def get(tag):
            el = ts.find(f"ns:{tag}", ns_map)
            return el.text if el is not None else None

        business_type = get("businessType")
        curve_type = get("curveType")
        mkt_agreement_type = None
        cma = ts.find("ns:contract_MarketAgreement.type", ns_map)
        if cma is not None:
            mkt_agreement_type = cma.text

        periods = ts.findall(".//ns:Period", ns_map)
        n_points = sum(len(p.findall("ns:Point", ns_map)) for p in periods)
        resolutions = {p.find("ns:resolution", ns_map).text for p in periods}

        # sample first and last point values for this series
        first_period = periods[0] if periods else None
        sample_vals = []
        if first_period is not None:
            for pt in first_period.findall("ns:Point", ns_map)[:3]:
                pos = pt.find("ns:position", ns_map).text
                price_el = pt.find("ns:price.amount", ns_map)
                sample_vals.append((pos, price_el.text if price_el is not None else None))

        print(f"TimeSeries[{i}]  businessType={business_type}  curveType={curve_type}  "
              f"contract_MarketAgreement.type={mkt_agreement_type}")
        print(f"    resolutions={resolutions}  n_points={n_points}  "
              f"first_period_start={first_period.find('ns:timeInterval/ns:start', ns_map).text if first_period is not None else None}")
        print(f"    sample (position, price): {sample_vals}")
        print()


if __name__ == "__main__":
    main()
