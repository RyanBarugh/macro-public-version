from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

# SSB PxWebApi base URL — same protocol as SCB (POST JSON query, JSON-stat2 response)
# SSB confirmed: "We basically use the same API code as Statistics Sweden (SCB)"
# Table URL pattern: https://data.ssb.no/api/v0/en/table/{table_id}
BASE_URL = "https://data.ssb.no/api/v0/en/table"


class SsbProvider(BaseProvider):
    """
    Statistics Norway (SSB / Statistisk sentralbyrå) PxWebApi provider.

    Uses the identical PxWebApi protocol as SCB (Sweden) and FSO (Switzerland):
        POST {BASE_URL}/{table_id}
        Body: {"query": [...variable filters...], "response": {"format": "json-stat2"}}

    Time format in SSB responses: "YYYYMnn" (e.g. "2024M01").

    Each series definition must supply in meta:
        table_id    : str  — 5-digit SSB table number (e.g. "07129")
        query       : list — PXWeb variable filter dicts (excluding time)

    The `start` parameter is not used as a fetch-time filter (PxWeb "from" filter
    is unreliable); all periods are fetched and NaN rows are dropped in clean().

    Retail Trade Index table (Detaljomsetningsindeksen):
        table_id: "07129"
        Title: "The Index of wholesale and retail trade, by industry (SIC2007) (2021=100)"
        Available from: 2000M01
        Time variable code: "Tid"
        Industry variable code: "Næring"
        Contents variable code: "ContentsCode"

    Key ContentsCode values for table 07129:
        "VolumeIndexSeasonAdj"     — Volume index, seasonally adjusted (2021=100)  ← USE THIS
        "VolumeIndexCalAdj"        — Volume index, calendar adjusted
        "VolumeIndex"              — Volume index, unadjusted
        "ValueIndexSeasonAdj"      — Value index, seasonally adjusted
        "ValueIndex"               — Value index, unadjusted
        "Weights"                  — Weights (per mille)

    Key Næring (industry) values for table 07129:
        "47"   — Retail trade excl. motor vehicles (NACE 47) — total retail  ← USE THIS
        "471"  — Retail in non-specialised stores
        "472"  — Retail of food, beverages, tobacco
        "45"   — Wholesale/retail trade of motor vehicles

    NOTE: Only NACE 47 (retail excl. motor vehicles) has a seasonally adjusted series.
    SSB documentation: "There is only for sector 47 Retail trade (except of motor vehicles
    and motorcycles) there is seasonally adjusted series."

    NOTE: Verify ContentsCode and Næring values via metadata GET request:
        GET https://data.ssb.no/api/v0/en/table/07129

    Rate limits: 800,000 cells per request (SSB); retail history well within limit.
    No authentication required.
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — SSB is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        table_id = meta.get("table_id")
        query_filters = list(meta.get("query", []))  # copy — we'll add time filter

        if not table_id:
            raise RuntimeError(
                f"SSB series_id={series_def.series_id} missing 'table_id' in meta"
            )

        # Request all time periods — PxWeb "from" filter behaviour is inconsistent
        # across SSB/SCB tables; start-date filtering can be done in clean() if needed.
        time_filter = {
            "code": "Tid",
            "selection": {
                "filter": "all",
                "values": ["*"],
            },
        }
        query_filters.append(time_filter)

        url = f"{BASE_URL}/{table_id}"
        body = {
            "query": query_filters,
            "response": {"format": "json-stat2"},
        }

        logger.info(
            "Fetching provider=ssb series_id=%s table=%s start=%s",
            series_def.series_id, table_id, start,
        )
        logger.debug("SSB URL: %s", url)

        start_ts = time.time()
        try:
            response = session.post(url, json=body, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=ssb series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()
            data = response.json()

            if "value" not in data:
                raise RuntimeError(
                    f"SSB response missing 'value' for series_id={series_def.series_id}"
                )
            if "dimension" not in data:
                raise RuntimeError(
                    f"SSB response missing 'dimension' for series_id={series_def.series_id}"
                )

            rows = len(data.get("value", []))
            logger.info(
                "Fetched provider=ssb series_id=%s values=%d",
                series_def.series_id, rows,
            )
            return data

        except Exception:
            logger.exception("Failed provider=ssb series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        JSON-stat2 parsing.

        SSB time labels: "YYYYMnn" (e.g. "2024M01").
        We normalise these to "YYYY-MM" before handing to pandas.
        """
        logger.info("Cleaning provider=ssb series_id=%s strict=%s", series_id, strict)

        if not raw_payload or "value" not in raw_payload:
            raise ValueError(f"SSB payload empty or malformed for series_id={series_id}")

        dimension_ids = raw_payload.get("id", [])
        dimensions = raw_payload.get("dimension", {})
        values = raw_payload.get("value", [])

        # Identify time dimension — SSB uses "Tid"
        time_dim_id = _find_time_dimension(dimension_ids)
        if time_dim_id is None:
            raise ValueError(
                f"SSB payload has no identifiable time dimension for series_id={series_id}. "
                f"Dimensions: {dimension_ids}"
            )

        time_category = dimensions[time_dim_id]["category"]
        time_codes = list(time_category.get("label", {}).keys())
        if "index" in time_category:
            idx_map = time_category["index"]
            if isinstance(idx_map, dict):
                time_codes = sorted(time_codes, key=lambda c: idx_map.get(c, 0))
            elif isinstance(idx_map, list):
                time_codes = idx_map

        time_labels_map = time_category.get("label", {})
        n_times = len(time_codes)

        # Calculate stride for multi-dimensional payloads
        sizes = raw_payload.get("size", [])
        time_dim_pos = dimension_ids.index(time_dim_id) if time_dim_id in dimension_ids else -1
        stride = 1
        for i in range(time_dim_pos + 1, len(sizes)):
            stride *= sizes[i]

        if stride != 1:
            logger.warning(
                "SSB series_id=%s has stride=%d; taking first value per time period.",
                series_id, stride,
            )

        logger.info(
            "Raw values series_id=%s n_times=%d values_total=%d",
            series_id, n_times, len(values),
        )

        # Track API-null values separately from parse failures.
        # PxWeb returns None for periods where the ContentsCode has no data;
        # these are expected and should not count toward the strict threshold.
        records = []
        raw_null_count = 0
        for i, code in enumerate(time_codes):
            label = time_labels_map.get(code, code)
            normalised = _normalise_ssb_time(label)
            flat_idx = i * stride
            val = values[flat_idx] if flat_idx < len(values) else None
            if val is None:
                raw_null_count += 1
            records.append({"time": normalised, "value": val})

        if raw_null_count:
            logger.info(
                "API returned %d/%d null values for series_id=%s "
                "(expected for partial-coverage tables)",
                raw_null_count, len(records), series_id,
            )

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        out = df[["time", "value"]].copy()
        out.insert(0, "series_id", series_id)

        before = len(out)
        out = out.dropna(subset=["time", "value"]).sort_values("time")
        dropped = before - len(out)

        # Strict threshold applies only to rows that had a non-null raw value
        # but still failed parsing. API-null rows are expected in PxWeb tables
        # where a ContentsCode doesn't cover the full time range.
        parse_failures = max(dropped - raw_null_count, 0)

        if dropped:
            drop_pct = (dropped / before) * 100
            bad = df[df["time"].isna() | df["value"].isna()].head(5)
            logger.warning(
                "Dropped %d/%d rows (%.1f%%) for series_id=%s "
                "(api_nulls=%d, parse_failures=%d). Sample:\n%s",
                dropped, before, drop_pct, series_id,
                raw_null_count, parse_failures, bad.to_string(),
            )
            if strict and parse_failures > 0 and (before - raw_null_count) > 0:
                non_null_rows = before - raw_null_count
                fail_pct = (parse_failures / non_null_rows) * 100
                if fail_pct > MAX_DROP_PCT:
                    raise ValueError(
                        f"series_id={series_id}: {parse_failures}/{non_null_rows} "
                        f"non-null rows failed parsing ({fail_pct:.1f}%, "
                        f"threshold={MAX_DROP_PCT}%). Possible format change."
                    )

        if not len(out):
            raise ValueError(
                f"series_id={series_id}: no valid rows after cleaning "
                f"(total={before}, api_nulls={raw_null_count}, "
                f"parse_failures={parse_failures})"
            )

        dupes = out[out["time"].duplicated(keep=False)]
        if not dupes.empty:
            dupe_times = dupes["time"].unique().tolist()
            raise ValueError(
                f"series_id={series_id}: duplicate time values detected: {dupe_times}"
            )

        logger.info("Cleaned rows series_id=%s rows_out=%d", series_id, len(out))
        out["time"] = out["time"].dt.strftime("%Y-%m-%d")
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUARTER_TO_MONTH = {"1": "01", "2": "04", "3": "07", "4": "10"}


def _to_ssb_time(yyyymm: str) -> str:
    """
    Convert pipeline start string to SSB time code.
    Monthly:   "YYYY-MM" -> "YYYYMnn"
    Quarterly: "YYYY-Qn" -> "YYYYKn"
    """
    parts = yyyymm.split("-")
    if len(parts) == 2:
        if parts[1].startswith("Q"):
            return f"{parts[0]}K{parts[1][1:]}"
        return f"{parts[0]}M{parts[1]}"
    return yyyymm


def _normalise_ssb_time(label: str) -> str:
    """
    Normalise SSB time labels to "YYYY-MM-DD" for pandas.
    Monthly:   "YYYYMnn" -> "YYYY-MM-01"
    Quarterly: "YYYYKn"  -> "YYYY-MM-01" (first month of quarter)
    """
    # Quarterly: "1978K1"
    if "K" in label and len(label) == 6:
        year = label[:4]
        q = label[5]
        month = _QUARTER_TO_MONTH.get(q, "01")
        return f"{year}-{month}-01"
    # Monthly: "2024M01"
    if "M" in label and len(label) == 7:
        return f"{label[:4]}-{label[5:]}-01"
    return label


def _find_time_dimension(dimension_ids: list[str]) -> str | None:
    """Find the time dimension ID."""
    for dim_id in dimension_ids:
        if dim_id.lower() in ("tid", "time", "month", "period"):
            return dim_id
    return dimension_ids[-1] if dimension_ids else None