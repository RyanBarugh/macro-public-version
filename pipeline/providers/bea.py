from __future__ import annotations

import time
from typing import Any, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

# BEA NIPA API — free, requires API key (register at apps.bea.gov/api/signup/)
# Set via env var API_SECRET__BEA
BASE_URL = "https://apps.bea.gov/api/data/"

# Map quarter number to first month of quarter
_QUARTER_TO_MONTH = {"1": "01", "2": "04", "3": "07", "4": "10"}


def _parse_bea_period(period: str) -> str:
    """
    Convert BEA TimePeriod string to ISO date string.

    Examples:
        "1947Q1" -> "1947-01-01" (quarterly)
        "1959M01" -> "1959-01-01" (monthly)
    """
    try:
        if "Q" in period:
            # Quarterly format: "1947Q1"
            year, q = period.split("Q")
            month = _QUARTER_TO_MONTH[q]
            return f"{year}-{month}-01"
        elif "M" in period:
            # Monthly format: "1959M01"
            year, month = period.split("M")
            return f"{year}-{month}-01"
        else:
            raise ValueError(f"Unknown period format: {period}")
    except (ValueError, KeyError) as e:
        raise ValueError(f"Cannot parse BEA TimePeriod '{period}': {e}") from e


class BeaProvider(BaseProvider):
    """
    Bureau of Economic Analysis (BEA) provider.
    U.S. Department of Commerce national accounts data.

    Free access; requires a BEA API key (register at apps.bea.gov/api/signup/).
    Set via env var API_SECRET__BEA.

    Each series definition must supply in meta:
        table_name  : str — NIPA table name, e.g. "T10106"
        line_number : str — line number within the table, e.g. "1"
        series_code : str — BEA series code for validation, e.g. "A191RX"

    Response JSON shape:
        {
          "BEAAPI": {
            "Results": {
              "Data": [
                {
                  "TableName":       "T10106",
                  "SeriesCode":      "A191RX",
                  "LineNumber":      "1",
                  "LineDescription": "Gross domestic product",
                  "TimePeriod":      "1947Q1",
                  "METRIC_NAME":     "Chained Dollars",
                  "CL_UNIT":         "Level",
                  "UNIT_MULT":       "6",
                  "DataValue":       "2,182,681",
                  "NoteRef":         "T10106"
                },
                ...
              ]
            }
          }
        }

    Notes:
        - TimePeriod format is "YYYYQn" — parsed to first day of quarter
        - DataValue contains comma-formatted numbers — commas stripped before parsing
        - UNIT_MULT=6 means values are in millions (10^6); stored as-is since
          the derived layer works on percentage changes, not absolute levels
        - Year=X returns all available years (BEA's special parameter)

    Known series:
        T10106 / Line 1 / A191RX — Real GDP, chained 2017 dollars, SA,
                                    quarterly from 1947Q1
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        if not api_key:
            raise RuntimeError(
                "BeaProvider requires an API key. "
                "Set env var API_SECRET__BEA."
            )

        meta = series_def.meta or {}
        table_name = meta.get("table_name")
        line_number = str(meta.get("line_number", ""))

        if not table_name:
            raise RuntimeError(
                f"BEA series_id={series_def.series_id} missing "
                f"'table_name' in meta"
            )
        if not line_number:
            raise RuntimeError(
                f"BEA series_id={series_def.series_id} missing "
                f"'line_number' in meta"
            )

        # BEA requires Year=X for full history; start filtering happens in clean()
        params = {
            "UserID":      api_key,
            "method":      "GetData",
            "DataSetName": "NIPA",
            "TableName":   table_name,
            "Frequency":   meta.get("freq", "Q").upper(),
            "Year":        "X",
            "ResultFormat": "JSON",
        }

        url = f"{BASE_URL}?{urlencode(params)}"

        logger.info(
            "Fetching provider=bea series_id=%s table=%s line=%s",
            series_def.series_id, table_name, line_number,
        )

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=bea series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            payload = response.json()

            # Check for BEA API-level errors
            results = payload.get("BEAAPI", {}).get("Results", {})
            if "Error" in results:
                err = results["Error"]
                raise RuntimeError(
                    f"BEA API error for series_id={series_def.series_id}: "
                    f"[{err.get('APIErrorCode')}] {err.get('APIErrorDescription')}"
                )

            data = results.get("Data", [])
            if not data:
                raise RuntimeError(
                    f"BEA response contains no Data for "
                    f"series_id={series_def.series_id} table={table_name}"
                )

            # Attach meta for clean() to filter by line_number and start date
            payload["_line_number"] = line_number
            payload["_start"] = start

            row_count = len(data)
            logger.info(
                "Fetched provider=bea series_id=%s total_rows=%d (pre-filter)",
                series_def.series_id, row_count,
            )
            return payload

        except Exception:
            logger.exception("Failed provider=bea series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=bea series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"BEA payload empty for series_id={series_id}")

        results = raw_payload.get("BEAAPI", {}).get("Results", {})
        data = results.get("Data", [])
        if not data:
            raise ValueError(f"BEA Data list empty for series_id={series_id}")

        line_number = raw_payload.get("_line_number", "1")
        start = raw_payload.get("_start", "")

        df = pd.DataFrame(data)

        # Filter to the requested line number
        df = df[df["LineNumber"] == line_number].copy()
        if df.empty:
            raise ValueError(
                f"BEA series_id={series_id}: no rows for LineNumber={line_number}"
            )

        # Parse TimePeriod "YYYYQn" → ISO date string
        df["time"] = df["TimePeriod"].apply(_parse_bea_period)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # Strip commas from DataValue and convert to float
        df["value"] = (
            df["DataValue"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

        out = df[["time", "value"]].copy()
        out.insert(0, "series_id", series_id)

        # Apply start date filter
        if start:
            start_dt = pd.to_datetime(f"{start}-01", errors="coerce")
            if pd.notna(start_dt):
                out = out[out["time"] >= start_dt]

        before = len(out)
        out = out.dropna(subset=["time", "value"]).sort_values("time")
        dropped = before - len(out)

        if dropped:
            drop_pct = (dropped / before) * 100 if before > 0 else 0.0
            logger.warning(
                "Dropped %d/%d rows (%.1f%%) for series_id=%s",
                dropped, before, drop_pct, series_id,
            )
            if strict and drop_pct > MAX_DROP_PCT:
                raise ValueError(
                    f"series_id={series_id}: dropped {drop_pct:.1f}% of rows "
                    f"(threshold={MAX_DROP_PCT}%). Possible format change."
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