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

# BLS Public Data API v2
BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# v2 limits with registration key:
#   - 500 requests/day
#   - 50 series per request
#   - 20 years per request
# Without key: 25 requests/day, 25 series, 10 years
_MAX_YEARS_PER_REQUEST = 20
_MAX_SERIES_PER_REQUEST = 50


class BlsProvider(BaseProvider):
    """
    U.S. Bureau of Labor Statistics (BLS) Public Data API v2 provider.

    Supports two modes:
      1. Single-series fetch (original) — called by the orchestrator per series.
      2. Batch fetch — fetches up to 50 BLS series in a single API call.
         The orchestrator calls fetch_batch() before the main loop, then
         fetch() returns cached results for individual series.

    Each series definition must supply in meta:
        bls_series_id  : str — BLS series identifier

    API key (registrationkey) is recommended for higher rate limits.
    """

    def __init__(self):
        super().__init__()
        # Cache for batch-fetched results: {bls_series_id: [data_points]}
        self._batch_cache: dict[str, list[dict]] = {}

    def fetch_batch(
        self,
        series_defs: list[SeriesDef],
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> dict[str, Any]:
        """
        Fetch multiple BLS series in a single API call (up to 50).

        Returns dict: {series_id: {"bls_series_id": ..., "data": [...]}}
        Also populates self._batch_cache so subsequent fetch() calls
        can return cached results without hitting the API.
        """
        if not series_defs:
            return {}

        # Build mapping: bls_series_id -> series_id
        bls_to_series: dict[str, str] = {}
        for sd in series_defs:
            meta = sd.meta or {}
            bls_id = meta.get("bls_series_id")
            if bls_id:
                bls_to_series[bls_id] = sd.series_id

        if not bls_to_series:
            return {}

        bls_ids = list(bls_to_series.keys())

        # Parse start year
        start_year = int(start.split("-")[0])
        import datetime
        end_year = datetime.date.today().year

        # Collect all data across year chunks
        all_series_data: dict[str, list[dict]] = {bid: [] for bid in bls_ids}

        chunk_start = start_year
        while chunk_start <= end_year:
            chunk_end = min(chunk_start + _MAX_YEARS_PER_REQUEST - 1, end_year)

            # Split into batches of 50 if we ever have more
            for i in range(0, len(bls_ids), _MAX_SERIES_PER_REQUEST):
                batch = bls_ids[i:i + _MAX_SERIES_PER_REQUEST]

                payload = {
                    "seriesid": batch,
                    "startyear": str(chunk_start),
                    "endyear": str(chunk_end),
                }
                if api_key:
                    payload["registrationkey"] = api_key

                logger.info(
                    "BLS batch fetch: %d series, years=%d-%d",
                    len(batch), chunk_start, chunk_end,
                )

                start_ts = time.time()
                try:
                    response = session.post(
                        BASE_URL,
                        json=payload,
                        headers={"Content-type": "application/json"},
                        timeout=timeout,
                    )
                    elapsed_ms = int((time.time() - start_ts) * 1000)
                    logger.info(
                        "BLS batch HTTP status=%s elapsed_ms=%d series=%d years=%d-%d",
                        response.status_code, elapsed_ms, len(batch),
                        chunk_start, chunk_end,
                    )
                    response.raise_for_status()
                    data = response.json()

                    status = data.get("status")
                    if status != "REQUEST_SUCCEEDED":
                        messages = data.get("message", [])
                        raise RuntimeError(
                            f"BLS batch API returned status={status}: {messages}"
                        )

                    results = data.get("Results", {})
                    series_list = results.get("series", [])

                    for series_entry in series_list:
                        sid = series_entry.get("seriesID", "")
                        points = series_entry.get("data", [])
                        if sid in all_series_data:
                            all_series_data[sid].extend(points)

                    logger.info(
                        "BLS batch fetched: %d series returned, years=%d-%d",
                        len(series_list), chunk_start, chunk_end,
                    )

                except Exception:
                    logger.exception(
                        "BLS batch fetch failed: years=%d-%d", chunk_start, chunk_end,
                    )
                    raise

            chunk_start = chunk_end + 1

        # Populate cache and build return dict
        result: dict[str, Any] = {}
        for bls_id, points in all_series_data.items():
            series_id = bls_to_series[bls_id]
            payload = {"bls_series_id": bls_id, "data": points}
            self._batch_cache[bls_id] = points
            result[series_id] = payload
            logger.info(
                "BLS batch: %s (%s) → %d points",
                series_id, bls_id, len(points),
            )

        logger.info(
            "BLS batch complete: %d series, %d total points",
            len(result), sum(len(v["data"]) for v in result.values()),
        )
        return result

    def clear_batch_cache(self) -> None:
        """Clear the batch cache between runs."""
        self._batch_cache.clear()

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        bls_series_id = meta.get("bls_series_id")

        if not bls_series_id:
            raise RuntimeError(
                f"BLS series_id={series_def.series_id} missing 'bls_series_id' in meta"
            )

        # If batch cache has this series, return from cache (no API call)
        if bls_series_id in self._batch_cache:
            points = self._batch_cache[bls_series_id]
            logger.info(
                "BLS cache hit: series_id=%s bls_id=%s points=%d",
                series_def.series_id, bls_series_id, len(points),
            )
            return {"bls_series_id": bls_series_id, "data": points}

        # Fallback: single-series fetch (original behaviour)
        start_year = int(start.split("-")[0])

        import datetime
        end_year = datetime.date.today().year

        all_data = []
        chunk_start = start_year

        while chunk_start <= end_year:
            chunk_end = min(chunk_start + _MAX_YEARS_PER_REQUEST - 1, end_year)

            payload = {
                "seriesid": [bls_series_id],
                "startyear": str(chunk_start),
                "endyear": str(chunk_end),
            }
            if api_key:
                payload["registrationkey"] = api_key

            logger.info(
                "Fetching provider=bls series_id=%s bls_id=%s years=%d-%d",
                series_def.series_id, bls_series_id, chunk_start, chunk_end,
            )

            start_ts = time.time()
            try:
                response = session.post(
                    BASE_URL,
                    json=payload,
                    headers={"Content-type": "application/json"},
                    timeout=timeout,
                )
                elapsed_ms = int((time.time() - start_ts) * 1000)
                logger.info(
                    "HTTP provider=bls series_id=%s status=%s elapsed_ms=%d years=%d-%d",
                    series_def.series_id, response.status_code, elapsed_ms,
                    chunk_start, chunk_end,
                )
                response.raise_for_status()
                data = response.json()

                status = data.get("status")
                if status != "REQUEST_SUCCEEDED":
                    messages = data.get("message", [])
                    raise RuntimeError(
                        f"BLS API returned status={status} for "
                        f"series_id={series_def.series_id}: {messages}"
                    )

                results = data.get("Results", {})
                series_list = results.get("series", [])

                if not series_list:
                    raise RuntimeError(
                        f"BLS returned no series data for "
                        f"series_id={series_def.series_id} ({chunk_start}-{chunk_end})"
                    )

                series_data = series_list[0]
                points = series_data.get("data", [])
                all_data.extend(points)

                logger.info(
                    "Fetched provider=bls series_id=%s chunk=%d-%d points=%d",
                    series_def.series_id, chunk_start, chunk_end, len(points),
                )

            except Exception:
                logger.exception(
                    "Failed provider=bls series_id=%s chunk=%d-%d",
                    series_def.series_id, chunk_start, chunk_end,
                )
                raise

            chunk_start = chunk_end + 1

        logger.info(
            "Fetched provider=bls series_id=%s total_points=%d",
            series_def.series_id, len(all_data),
        )

        return {
            "bls_series_id": bls_series_id,
            "data": all_data,
        }

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Parse BLS API response.

        Each data point is a dict:
            {"year": "2024", "period": "M01", "periodName": "January",
             "value": "308.417", "footnotes": [{}]}

        Period codes: M01-M12 = monthly, M13 = annual average (skip),
                      S01/S02 = semi-annual (skip).
        """
        logger.info("Cleaning provider=bls series_id=%s strict=%s", series_id, strict)

        if not raw_payload or "data" not in raw_payload:
            raise ValueError(f"BLS payload empty for series_id={series_id}")

        points = raw_payload["data"]
        logger.info(
            "Raw datapoints series_id=%s rows_in=%d", series_id, len(points)
        )

        # Q01-Q04 -> quarter-start month
        QUARTER_TO_MONTH = {"Q01": 1, "Q02": 4, "Q03": 7, "Q04": 10}

        records = []
        for pt in points:
            year = pt.get("year", "")
            period = pt.get("period", "")
            value = pt.get("value", "")

            if period in QUARTER_TO_MONTH:
                month = QUARTER_TO_MONTH[period]
            elif period.startswith("M"):
                month_str = period[1:]
                try:
                    month = int(month_str)
                except ValueError:
                    continue
                if month < 1 or month > 12:
                    continue
            else:
                continue  # skip annual averages, semi-annual, etc.

            time_str = f"{year}-{month:02d}-01"
            records.append({"time": time_str, "value": value})

        df = pd.DataFrame(records) if records else pd.DataFrame(columns=["time", "value"])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        out = df[["time", "value"]].copy()
        out.insert(0, "series_id", series_id)

        before = len(out)
        out = out.dropna(subset=["time", "value"]).sort_values("time")
        dropped = before - len(out)

        if dropped:
            drop_pct = (dropped / before) * 100
            bad = df[df["time"].isna() | df["value"].isna()].head(5)
            logger.warning(
                "Dropped %d/%d rows (%.1f%%) for series_id=%s. Sample:\n%s",
                dropped, before, drop_pct, series_id, bad.to_string(),
            )
            if strict and drop_pct > MAX_DROP_PCT:
                raise ValueError(
                    f"series_id={series_id}: dropped {drop_pct:.1f}% of rows "
                    f"(threshold={MAX_DROP_PCT}%). Possible format change."
                )

        dupes = out[out["time"].duplicated(keep=False)]
        if not dupes.empty:
            # BLS can return overlapping data from chunked requests — deduplicate
            logger.info(
                "Deduplicating %d duplicate time entries for series_id=%s",
                len(dupes), series_id,
            )
            out = out.drop_duplicates(subset=["time"], keep="last").sort_values("time")

        logger.info("Cleaned rows series_id=%s rows_out=%d", series_id, len(out))
        out["time"] = out["time"].dt.strftime("%Y-%m-%d")
        return out