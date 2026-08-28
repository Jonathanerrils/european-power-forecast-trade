"""ENTSO-E Transparency Platform client.

Pulls exactly three data types for the DE-LU bidding zone, per section 3
of the spec:

    - Day-ahead prices                          (documentType A44)
    - Day-ahead total load forecast             (documentType A65)
    - Day-ahead generation forecast, wind/solar  (documentType A69)

Design principles enforced here (see spec sections 8 and 22):
    - Never hardcode the token; read from ENTSOE_TOKEN env var.
    - Cache every raw XML response to disk, keyed by request params, so
      re-runs don't re-hit the API and so raw data is auditable.
    - Retry with backoff on transient failures.
    - Log ingestion metadata for every request: source, retrieval
      timestamp, requested window, bidding zone, data type, unit,
      timezone, raw row count.
    - Return data in long-format, timezone-aware UTC DataFrames. Nothing
      here decides what is point-in-time available. That is enforced by
      the feature-construction / leakage-control layer (src/features.py
      + tests/test_no_leakage.py, not yet built).

ENTSO-E's API returns UTC timestamps in its XML (<start>/<end> under
<Period>, with a <resolution> and per-interval <Point> values), so all
parsing here treats the platform's timestamps as UTC.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from .utils import get_entsoe_token, load_config, REPO_ROOT

logger = logging.getLogger("power_forecast.entsoe_client")

ENTSOE_NS = "{urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3}"

# psrType codes ENTSO-E uses inside A69 (wind/solar generation forecast) documents
PSR_TYPE_SOLAR = "B16"
PSR_TYPE_WIND_ONSHORE = "B19"
PSR_TYPE_WIND_OFFSHORE = "B18"


@dataclass
class IngestionRecord:
    """One row of the ingestion metadata log (spec section 22)."""

    source: str
    retrieval_timestamp_utc: str
    requested_start: str
    requested_end: str
    bidding_zone: str
    data_type: str
    unit: str
    timezone: str
    raw_row_count: int
    cache_hit: bool
    request_params: dict = field(default_factory=dict)


class EntsoeClient:
    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_config(config_path)
        self.token = get_entsoe_token(self.cfg["entsoe"]["token_env_var"])
        self.base_url = self.cfg["entsoe"]["base_url"]
        self.timeout = self.cfg["entsoe"]["request_timeout_seconds"]
        self.max_retries = self.cfg["entsoe"]["max_retries"]
        self.backoff = self.cfg["entsoe"]["retry_backoff_seconds"]
        self.eic_code = self.cfg["market"]["eic_code"]
        self.chunk_days = self.cfg["entsoe"]["chunk_days"]

        self.cache_dir = REPO_ROOT / self.cfg["data"]["cache_dir"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ingestion_log: List[IngestionRecord] = []

    # ------------------------------------------------------------------
    # Low-level request handling: caching + retries
    # ------------------------------------------------------------------
    def _cache_key(self, params: dict) -> str:
        blob = json.dumps(params, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _cache_path(self, params: dict) -> Path:
        return self.cache_dir / f"{self._cache_key(params)}.xml"

    def _request(self, params: dict) -> tuple[str, bool]:
        """Return (raw_xml_text, cache_hit)."""
        cache_path = self._cache_path(params)
        if cache_path.exists():
            logger.info("Cache hit: %s", cache_path.name)
            return cache_path.read_text(), True

        query = dict(params)
        query["securityToken"] = self.token

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(self.base_url, params=query, timeout=self.timeout)
                if resp.status_code == 200:
                    cache_path.write_text(resp.text)
                    return resp.text, False
                if resp.status_code in (400, 401, 403):
                    # Not retryable: bad token, bad params, no permission
                    raise RuntimeError(
                        f"ENTSO-E API returned {resp.status_code}: {resp.text[:500]}"
                    )
                logger.warning(
                    "ENTSO-E request failed (status %s), attempt %d/%d",
                    resp.status_code, attempt, self.max_retries,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "ENTSO-E request error: %s, attempt %d/%d",
                    exc, attempt, self.max_retries,
                )
            time.sleep(self.backoff * attempt)

        raise RuntimeError(
            f"ENTSO-E request failed after {self.max_retries} attempts. "
            f"Last error: {last_exc}"
        )

    def _chunk_windows(self, start: datetime, end: datetime):
        """Yield (chunk_start, chunk_end) pairs no longer than chunk_days."""
        cur = start
        step = timedelta(days=self.chunk_days)
        while cur < end:
            nxt = min(cur + step, end)
            yield cur, nxt
            cur = nxt

    @staticmethod
    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%d%H%M")

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------
    def _expand_period_points(self, points_by_position, curve_type, n_positions):
        """Expands a Period's (possibly sparse) explicit Points into a
        full {position: (value, is_synthesized)} dict for positions
        1..n_positions, per ENTSO-E's official curveType specification
        ("The Introduction of Different Time Series Possibilities
        (CurveType) within ENTSO-E Electronic Documents", v1.4, section 4):

        - A01 (sequential fixed size blocks, also the default when
          curveType is omitted): every position must be provided. No
          forward-fill -- if a required position is absent, FAIL CLOSED
          with ValueError rather than emit a null that downstream means
          could silently ignore.
        - A03 (variable sized blocks): ONLY positions where the value
          CHANGES are provided ("only the position where a block change
          occurs is provided... This is useful in cases where the
          quantity is stable over a long period of time" -- spec section
          4.3). An unprovided position inherits the value of the most
          recent lower provided position; the final provided position's
          value extends through the end of the Period. THIS IS NOT
          MISSING DATA -- treating a skipped A03 position as null
          silently undercounts real published prices, which is exactly
          the bug this function exists to fix. (Genuine gaps under A03
          are represented differently: as two temporally DISJOINT Period
          elements within one TimeSeries, per spec section 5 -- that
          case isn't "a missing position inside one Period" and is
          handled naturally by this function only ever seeing one
          Period's positions at a time.)
        - A02 (points): explicitly sparse by design ("no relational
          significance between each reading" -- spec section 4.2). NOT
          expanded; only the provided positions are meaningful.
        - Anything else (A04/A05 breakpoints, unrecognized, or missing
          when a non-default value was clearly intended): FAILS CLOSED.
          A04/A05 use linear interpolation between breakpoints, a
          fundamentally different reconstruction this project doesn't
          need for price data -- silently mishandling them would be
          worse than refusing to guess.
        """
        effective_curve_type = curve_type or "A01"  # spec: omitted -> default A01

        if not isinstance(n_positions, int) or n_positions <= 0:
            raise ValueError(f"n_positions must be a positive integer, got {n_positions!r}")

        invalid_positions = sorted(
            pos for pos in points_by_position
            if not isinstance(pos, int) or pos < 1 or pos > n_positions
        )
        if invalid_positions:
            raise ValueError(
                f"Point position(s) outside valid range 1..{n_positions}: {invalid_positions}"
            )

        null_value_positions = sorted(
            pos for pos, value in points_by_position.items() if value is None
        )
        if null_value_positions:
            raise ValueError(
                "Explicit Point(s) are missing a numeric value at position(s): "
                f"{null_value_positions}"
            )

        if effective_curve_type == "A01":
            missing_positions = sorted(set(range(1, n_positions + 1)) - set(points_by_position))
            if missing_positions:
                raise ValueError(
                    "curveType A01 requires an explicit Point for every interval; "
                    f"missing position(s): {missing_positions}"
                )
            return {
                pos: (points_by_position[pos], False)
                for pos in range(1, n_positions + 1)
            }

        if effective_curve_type == "A03":
            if 1 not in points_by_position:
                raise ValueError(
                    "curveType A03 must start with an explicit Point at position 1. "
                    "A genuine gap should be represented as a separate/disjoint Period, "
                    "not as leading missing positions inside one continuous Period."
                )
            expanded = {}
            last_value = None
            for pos in range(1, n_positions + 1):
                if pos in points_by_position:
                    last_value = points_by_position[pos]
                    expanded[pos] = (last_value, False)
                else:
                    expanded[pos] = (last_value, True)  # inherited block value, not a new explicit Point
            return expanded

        if effective_curve_type == "A02":
            return {pos: (val, False) for pos, val in points_by_position.items()}

        raise NotImplementedError(
            f"curveType '{effective_curve_type}' is not handled (only A01/A02/A03 are). "
            f"Refusing to guess how to reconstruct it -- A04/A05 use linear interpolation "
            f"between breakpoints, a fundamentally different scheme. Extend "
            f"_expand_period_points() deliberately if this document type is now expected."
        )

    def _parse_timeseries(self, xml_text: str, value_tag: str = "price.amount"):
        """Parse ENTSO-E GL_MarketDocument / Publication_MarketDocument XML
        into a list of (start_utc, resolution_minutes, value, psr_type,
        auction_sequence, curve_type, is_synthesized) tuples. Handles the
        namespace ENTSO-E uses across these document types.

        auction_sequence comes from
        classificationSequence_AttributeInstanceComponent.position --
        ENTSO-E's field distinguishing multiple auctions published within
        the same auction category and contract type for the same interval
        (discovered via a real production collision: two PT15M price
        series for 2025-09-30T22:00Z with different prices, position=1
        and position=2). PRESENT THROUGHOUT THE ENTIRE HISTORY, not just
        from 2025-10-01 onward -- an earlier version of this docstring
        incorrectly claimed classificationSequence was absent before the
        quarter-hour regime. It was always present; what changes at the
        cutover is sequence 1's RESOLUTION, not its existence.

        Verified against the full historical structural audit
        (scripts/audit_price_curve_types.py, 5,538 Period rows, zero
        exceptions): sequence 1 is PT60M for every pre-2025-10-01 row and
        PT15M for every row from 2025-10-01 onward -- i.e. sequence 1 is
        the continuous standard day-ahead product whose market design
        (MTU) changed from 60 to 15 minutes exactly at the SDAC cutover.
        Sequence 2 is PT15M throughout the entire history, both before
        and after -- a parallel quarter-hour product that already existed
        pre-cutover (this is the SAME PT60M/PT15M dual-product duality
        documented below in fetch_day_ahead_prices, just visible through
        a different field). This is strong INTERNAL structural evidence
        that sequence 1 is the correct choice, consistent with (not yet
        replacing) the external EPEX cross-check still in progress -- see
        README Limitations. The official schema does NOT establish that
        position=1 is "primary" on its own; that remains a project-level
        assumption, now backed by this structural consistency finding but
        still under active external audit, not a schema guarantee.

        curveType decoding (see _expand_period_points docstring) is
        REQUIRED, not optional: post-2025-10-01 DE-LU day-ahead prices
        use curveType=A03 (variable sized blocks), where ENTSO-E only
        publishes a Point when the price CHANGES from the previous
        quarter-hour. Earlier versions of this parser only emitted rows
        for explicit <Point> elements, silently dropping every
        unchanged-price quarter-hour as if it were missing data --
        undercounting real published prices, not handling genuine gaps.
        is_synthesized=True marks a forward-filled (not explicitly
        published) row, preserved for auditability rather than
        collapsed away.
        """
        root = ET.fromstring(xml_text)
        # Namespace is document-type specific; detect it from the root tag.
        ns = root.tag.split("}")[0].strip("{")
        ns_map = {"ns": ns}

        rows = []
        for ts in root.findall("ns:TimeSeries", ns_map):
            psr_type = None
            psr_el = ts.find(".//ns:MktPSRType/ns:psrType", ns_map)
            if psr_el is not None:
                psr_type = psr_el.text

            auction_sequence = None
            seq_el = ts.find(
                "ns:classificationSequence_AttributeInstanceComponent.position", ns_map
            )
            if seq_el is not None:
                auction_sequence = int(seq_el.text)

            curve_type_el = ts.find("ns:curveType", ns_map)
            curve_type = curve_type_el.text if curve_type_el is not None else None

            for period in ts.findall(".//ns:Period", ns_map):
                start_el = period.find("ns:timeInterval/ns:start", ns_map)
                end_el = period.find("ns:timeInterval/ns:end", ns_map)
                resolution_el = period.find("ns:resolution", ns_map)
                if (
                    start_el is None or not start_el.text
                    or end_el is None or not end_el.text
                    or resolution_el is None or not resolution_el.text
                ):
                    raise ValueError(
                        "ENTSO-E Period is missing timeInterval start/end or resolution; "
                        "refusing to infer period geometry."
                    )

                start_text = start_el.text
                end_text = end_el.text
                resolution_text = resolution_el.text
                period_start = datetime.strptime(
                    start_text, "%Y-%m-%dT%H:%MZ"
                ).replace(tzinfo=timezone.utc)
                period_end = datetime.strptime(
                    end_text, "%Y-%m-%dT%H:%MZ"
                ).replace(tzinfo=timezone.utc)
                res_minutes = self._resolution_to_minutes(resolution_text)

                duration_minutes = (period_end - period_start).total_seconds() / 60
                if duration_minutes <= 0:
                    raise ValueError(
                        f"ENTSO-E Period has non-positive duration: {start_text} -> {end_text}"
                    )
                if duration_minutes % res_minutes != 0:
                    raise ValueError(
                        "ENTSO-E Period duration is not exactly divisible by its resolution: "
                        f"duration={duration_minutes:g} minutes, resolution={resolution_text}, "
                        f"period={start_text}->{end_text}"
                    )
                n_positions = int(duration_minutes // res_minutes)

                points_by_position = {}
                for point in period.findall("ns:Point", ns_map):
                    position_el = point.find("ns:position", ns_map)
                    if position_el is None or not position_el.text:
                        raise ValueError("ENTSO-E Point is missing its required position.")
                    point_position = int(position_el.text)
                    if point_position in points_by_position:
                        raise ValueError(
                            f"Duplicate ENTSO-E Point position {point_position} within one Period; "
                            "refusing to silently overwrite one published value with another."
                        )

                    val_el = point.find(f"ns:{value_tag}", ns_map)
                    if val_el is None:
                        # some docs use quantity instead of price.amount
                        val_el = point.find("ns:quantity", ns_map)
                    if val_el is None or val_el.text is None or not val_el.text.strip():
                        raise ValueError(
                            f"ENTSO-E Point at position {point_position} has no "
                            f"'{value_tag}' or 'quantity' value."
                        )
                    value = float(val_el.text)
                    points_by_position[point_position] = value

                expanded = self._expand_period_points(points_by_position, curve_type, n_positions)
                for pos, (value, is_synthesized) in expanded.items():
                    ts_utc = period_start + timedelta(minutes=res_minutes * (pos - 1))
                    rows.append(
                        (ts_utc, res_minutes, value, psr_type, auction_sequence, curve_type, is_synthesized)
                    )
        return rows

    @staticmethod
    def _resolution_to_minutes(res: str) -> int:
        # ISO 8601 durations used by ENTSO-E: PT15M, PT30M, PT60M
        if res == "PT15M":
            return 15
        if res == "PT30M":
            return 30
        if res == "PT60M":
            return 60
        raise ValueError(f"Unhandled ENTSO-E resolution: {res}")

    # ------------------------------------------------------------------
    # Public fetch methods
    # ------------------------------------------------------------------
    def fetch_day_ahead_prices(self, start: datetime, end: datetime) -> pd.DataFrame:
        """DE-LU day-ahead prices have THREE layers of collision/encoding
        subtlety that all had to be discovered empirically against real
        data, not assumed from the schema. Layers 1 and 2 below describe
        what the full historical structural audit
        (scripts/audit_price_curve_types.py) shows is very likely the
        SAME real-world duality, visible through two different XML
        fields -- not two independent problems:

        1. Two parallel PRODUCTS since ~2019: a standard day-ahead
           auction and a separate quarter-hour auction running alongside
           it. Resolved by resolution + cutover date
           (select_price_resolution).
        2. TWO classificationSequence values are published for EVERY
           interval throughout the ENTIRE history (auction_sequence here)
           -- NOT just from 2025-10-01 onward; an earlier version of this
           docstring incorrectly claimed the field was absent
           pre-cutover. Confirmed via a real collision at 2025-09-30T22:00Z
           where sequence 1 and 2 had different prices with every other
           field identical.

           Verified against the full historical structural audit (5,538
           Period rows, zero exceptions): sequence 1 is PT60M for every
           row before 2025-10-01 and PT15M for every row from
           2025-10-01 onward -- i.e. sequence 1 IS the continuous standard
           day-ahead product from point 1 above, whose market design (MTU)
           changed from 60 to 15 minutes exactly at the SDAC cutover.
           Sequence 2 is PT15M throughout the entire history, both before
           and after -- the SAME parallel quarter-hour product from point
           1, just visible via classificationSequence instead of
           resolution. Point 1's dual-product distinction and point 2's
           dual-sequence distinction are very likely describing the same
           underlying market structure, not two separate collisions.

           This is strong INTERNAL structural evidence that keeping
           auction_sequence == 1 selects the correct (standard, continuing)
           product -- but it is evidence, not external confirmation. We
           still keep auction_sequence == 1 as a project-level assumption,
           NOT something the ENTSO-E schema itself establishes as
           "primary"; under active external audit against EPEX's
           published SDAC reference prices (see README Limitations,
           scripts/verify_auction_sequence.py).
        3. Post-2025-10-01 prices use curveType=A03 ("variable sized
           blocks" per ENTSO-E's official curveType specification): only
           positions where the price CHANGES are published as explicit
           XML Points. _parse_timeseries()/_expand_period_points()
           reconstruct the full quarter-hourly series by forward-filling
           unchanged prices -- an EARLIER version of this parser only
           emitted rows for explicit Points, silently undercounting real
           published prices as if they were missing data. The full
           historical audit confirms curveType=A03 is used THROUGHOUT
           the entire history, not just post-cutover -- pre-cutover PT60M
           data was also affected by the undercounting bug (~0.45% of
           historical PT60M intervals were reconstructed, per the audit).
           Pass keep_curve_metadata=True to _fetch_generic (not done here
           by default, to keep this method's output shape stable) if you
           need curve_type/is_synthesized for auditing which rows were
           forward-filled vs. explicitly published.

        See tests/test_entsoe_client.py for regression tests built
        directly from both real collisions, and tests for
        _expand_period_points built directly from ENTSO-E's own worked
        specification example.
        """
        df = self._fetch_generic(
            start, end,
            params_extra={
                "documentType": "A44",
                "in_Domain": self.eic_code,
                "out_Domain": self.eic_code,
            },
            data_type="day_ahead_price",
            unit="EUR/MWh",
            value_tag="price.amount",
            value_col="price_eur_mwh",
            keep_resolution=True,
            keep_auction_sequence=True,
        )
        df = select_primary_auction_sequence(df)
        return select_price_resolution(
            df,
            cutover=self.cfg["market_design"]["fifteen_min_mtu_start"],
        )

    def fetch_load_forecast(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self._fetch_generic(
            start, end,
            params_extra={
                "documentType": "A65",
                "processType": "A01",
                "outBiddingZone_Domain": self.eic_code,
            },
            data_type="day_ahead_load_forecast",
            unit="MW",
            value_tag="quantity",
            value_col="load_forecast_mw",
        )

    def fetch_wind_solar_forecast(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Returns one row per (timestamp, psr_type), wide-pivoted into
        wind_onshore_forecast_mw / wind_offshore_forecast_mw / solar_forecast_mw.
        """
        raw = self._fetch_generic(
            start, end,
            params_extra={
                "documentType": "A69",
                "processType": "A01",
                "in_Domain": self.eic_code,
            },
            data_type="day_ahead_wind_solar_forecast",
            unit="MW",
            value_tag="quantity",
            value_col="value_mw",
            keep_psr_type=True,
        )
        if raw.empty:
            return raw

        # Learned from the price series: duplicates within ENTSO-E data
        # can represent genuinely distinct submissions/revisions, not
        # noise. aggfunc="mean" would silently average them into a value
        # that may never have existed as an actual forecast. Verify
        # uniqueness explicitly before pivoting rather than assume it.
        dupe_key = ["timestamp_utc", "psr_type"]
        n_dupes = int(raw.duplicated(subset=dupe_key).sum())
        if n_dupes:
            raise ValueError(
                f"{n_dupes} duplicate (timestamp_utc, psr_type) row(s) in wind/solar "
                f"forecast data. Averaging these silently (aggfunc='mean') could produce "
                f"a value that never existed as an actual forecast -- investigate the "
                f"underlying XML structure (see scripts/inspect_price_xml.py for the "
                f"pattern used to diagnose the analogous price-series collision) before "
                f"deciding how to select the correct series."
            )

        pivot = raw.pivot_table(
            index="timestamp_utc", columns="psr_type", values="value_mw", aggfunc="first"
        )
        rename_map = {
            PSR_TYPE_SOLAR: "solar_forecast_mw",
            PSR_TYPE_WIND_ONSHORE: "wind_onshore_forecast_mw",
            PSR_TYPE_WIND_OFFSHORE: "wind_offshore_forecast_mw",
        }
        pivot = pivot.rename(columns=rename_map)
        for col in rename_map.values():
            if col not in pivot.columns:
                pivot[col] = pd.NA
        pivot = pivot.reset_index()
        return pivot[["timestamp_utc"] + list(rename_map.values())]

    # ------------------------------------------------------------------
    def _fetch_generic(
        self,
        start: datetime,
        end: datetime,
        params_extra: dict,
        data_type: str,
        unit: str,
        value_tag: str,
        value_col: str,
        keep_psr_type: bool = False,
        keep_resolution: bool = False,
        keep_auction_sequence: bool = False,
        keep_curve_metadata: bool = False,
    ) -> pd.DataFrame:
        all_rows = []
        any_cache_hit = False
        for chunk_start, chunk_end in self._chunk_windows(start, end):
            params = {
                "periodStart": self._fmt(chunk_start),
                "periodEnd": self._fmt(chunk_end),
                **params_extra,
            }
            xml_text, cache_hit = self._request(params)
            any_cache_hit = any_cache_hit or cache_hit
            rows = self._parse_timeseries(xml_text, value_tag=value_tag)
            all_rows.extend(rows)

            self.ingestion_log.append(
                IngestionRecord(
                    source="ENTSO-E Transparency Platform",
                    retrieval_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    requested_start=chunk_start.isoformat(),
                    requested_end=chunk_end.isoformat(),
                    bidding_zone=self.cfg["market"]["bidding_zone"],
                    data_type=data_type,
                    unit=unit,
                    timezone="UTC",
                    raw_row_count=len(rows),
                    cache_hit=cache_hit,
                    request_params=params,
                )
            )

        all_cols = ["timestamp_utc", "resolution_min", value_col, "psr_type", "auction_sequence",
                    "curve_type", "is_synthesized"]

        if not all_rows:
            cols = ["timestamp_utc", value_col]
            cols += ["psr_type"] if keep_psr_type else []
            cols += ["resolution_min"] if keep_resolution else []
            cols += ["auction_sequence"] if keep_auction_sequence else []
            cols += ["curve_type", "is_synthesized"] if keep_curve_metadata else []
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(all_rows, columns=all_cols)
        if not keep_psr_type:
            df = df.drop(columns=["psr_type"])
        if not keep_resolution:
            df = df.drop(columns=["resolution_min"])
        if not keep_auction_sequence:
            df = df.drop(columns=["auction_sequence"])
        if not keep_curve_metadata:
            df = df.drop(columns=["curve_type", "is_synthesized"])
        return df

    def save_ingestion_log(self, path: Optional[Path] = None) -> Path:
        """Writes this client instance's ingestion records to disk.

        Overwrites the file (mode "w"), not appends -- a real production
        bug found by reading an actual uploaded ingestion_log.jsonl: an
        earlier version opened in append mode ("a"), so every re-run of
        run_ingestion.py added its own records on top of whatever was
        already there. Three re-runs on 2026-08-14/15 (using an
        old, naive-UTC-boundary version of get_default_date_range,
        before the local_delivery_date_to_utc fix) each appended their
        own 8 price + 8 load + 8 wind/solar records -- all still
        cache_hit=True, meaning the underlying fetch was fine, only the
        LOG accumulated stale duplicates -- and a fourth, corrected run
        added its own 8+8+8 on top. The result: 32 price-chunk log
        entries where only 8 reflected the current, correct boundary
        logic, so anything reading this log naively (counting chunks,
        summing raw_row_count, or grabbing "the first" entry, as the
        `diagnostics/inspect_price_xml.py` family of scripts did) would
        see a count roughly 3-4x too high and could easily inspect a
        stale cache file from a superseded run instead of the current
        one.

        This is a LOGGING/METADATA bug, not a data-correctness bug: the
        actual clean_df each run_ingestion.py execution builds always
        reflects that run's own current, correctly-computed
        get_default_date_range() boundaries and the DataFrame
        fetch_day_ahead_prices() returns for them -- it never reads from
        this log file. But the log itself is meant to document "what
        this run did" (matching delu_hourly.parquet being overwritten
        wholesale each run, per build_clean_dataset's docstring), not
        "every ingestion ever attempted" -- accumulating history that
        way should be a deliberate choice (e.g. a timestamped archive
        file), not a side effect of opening in the wrong mode.
        """
        path = path or (REPO_ROOT / self.cfg["data"]["raw_dir"] / "ingestion_log.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for rec in self.ingestion_log:
                f.write(json.dumps(rec.__dict__) + "\n")
        logger.info("Wrote %d ingestion log records to %s (overwritten, not appended)", len(self.ingestion_log), path)
        return path


def select_primary_auction_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only auction_sequence == 1 (or rows with no sequence info at
    all -- e.g. non-price document types that never carry this field).
    classificationSequence is present throughout the ENTIRE price
    history, not just from 2025-10-01 -- an earlier version of this
    docstring incorrectly claimed it was absent pre-cutover. What
    changes at the cutover is sequence 1's RESOLUTION (PT60M -> PT15M),
    not its existence; see fetch_day_ahead_prices's docstring for the
    full structural-audit-verified pattern (sequence 1 = the continuous
    standard day-ahead product, sequence 2 = a parallel PT15M product
    present throughout).

    Discovered via a real production collision: from 2025-10-01 onward,
    both sequences are PT15M for the first time, so the previously
    resolution-distinguishable products collide -- ENTSO-E publishes a
    second PT15M price series per interval
    (classificationSequence_AttributeInstanceComponent.position == 2)
    with a genuinely different price and every other classification
    field (businessType, auction.type, contract_MarketAgreement.type,
    curveType, domains, currency, unit) identical to sequence 1.

    IMPORTANT: keeping sequence 1 is a PROJECT-LEVEL ASSUMPTION, not a
    fact established by the ENTSO-E schema. The official specification
    defines classificationSequence as distinguishing multiple auctions
    published within the same auction category and contract type; it
    does not state that position 1 is "the" primary/reference result.
    The full historical structural audit provides strong INTERNAL
    evidence for this choice (sequence 1's resolution transitions
    exactly at the SDAC MTU cutover, consistent with it being the
    continuous standard product) -- but this is not yet external
    confirmation. Still under active external audit against EPEX's
    published SDAC reference prices (see README Limitations,
    scripts/verify_auction_sequence.py, scripts/inspect_sequence_xml_attributes.py,
    scripts/audit_price_curve_types.py) and should not be treated as
    settled until that audit concludes.
    """
    if df.empty or "auction_sequence" not in df.columns:
        return df

    df = df.copy()
    keep = df["auction_sequence"].isna() | (df["auction_sequence"] == 1)
    out = df[keep].drop(columns=["auction_sequence"]).reset_index(drop=True)
    return out


def select_price_resolution(df: pd.DataFrame, cutover: str) -> pd.DataFrame:
    """Given raw price rows tagged with resolution_min (60 or 15), select
    exactly one resolution per period so the two parallel DE-LU price
    products (standard hourly auction vs. quarter-hour auction) don't
    collide on timestamp and get arbitrarily deduped. See
    fetch_day_ahead_prices docstring for the reasoning.

    Cutover is resolved via clean.local_delivery_date_to_utc, not naive
    UTC midnight -- see that function's docstring for why (the 2025-10-01
    delivery day starts at 2025-09-30T22:00:00Z for DE-LU, not
    2025-10-01T00:00:00Z). Kept as a single shared implementation so this
    module and clean.py can't drift into two different interpretations
    of the same cutover.
    """
    from .clean import local_delivery_date_to_utc  # local import: avoid import cycle at module load

    if df.empty:
        return df.drop(columns=["resolution_min"], errors="ignore")

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    cutover_ts = local_delivery_date_to_utc(cutover)

    pre = df[(df["timestamp_utc"] < cutover_ts) & (df["resolution_min"] == 60)]
    post = df[(df["timestamp_utc"] >= cutover_ts) & (df["resolution_min"] == 15)]

    out = pd.concat([pre, post], ignore_index=True)
    out = out.drop(columns=["resolution_min"]).sort_values("timestamp_utc").reset_index(drop=True)
    return out


def get_default_date_range(cfg: dict) -> tuple[datetime, datetime]:
    """Resolve config's start_date/end_date as LOCAL DELIVERY-DAY
    boundaries converted to UTC, defaulting end to the start of the
    latest complete month (never a partial current month), per spec
    section 1.

    Previously this parsed start_date/end_date as naive UTC midnight,
    which doesn't match how ENTSO-E actually organizes data (by local
    delivery day) or how the rest of this codebase treats date
    boundaries (local_delivery_date_to_utc is already used for the
    2025-10-01 market-design cutover). A real ingestion run confirmed
    the mismatch: the frozen dataset's first row was 2018-12-31T23:00Z
    -- one hour before the naive-UTC start boundary of
    2019-01-01T00:00:00Z -- because ENTSO-E returns the full local
    delivery day (2019-01-01 in Europe/Berlin starts at 2018-12-31 23:00
    UTC in winter). Using local_delivery_date_to_utc for both boundaries
    makes the declared config dates match what's actually returned, and
    build_clean_dataset() then clips to this exact range so nothing
    outside it silently leaks in regardless.
    """
    from .clean import local_delivery_date_to_utc  # local import: avoid import cycle at module load

    local_tz = cfg["market"]["timezone_local"]
    start = local_delivery_date_to_utc(cfg["data"]["start_date"], local_tz=local_tz)
    end_cfg = cfg["data"].get("end_date")
    if end_cfg:
        end = local_delivery_date_to_utc(end_cfg, local_tz=local_tz)
    else:
        now = datetime.now(timezone.utc)
        first_of_month = date(now.year, now.month, 1)
        end = local_delivery_date_to_utc(first_of_month.isoformat(), local_tz=local_tz)
    return start, end
