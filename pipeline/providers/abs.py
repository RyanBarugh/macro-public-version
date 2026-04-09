from __future__ import annotations

import time
from typing import Any, Optional, Tuple
from urllib.parse import urlencode
import io

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

# ABS Data API base (new URL as of Nov 2024)
BASE_URL = "https://data.api.abs.gov.au/rest/data"


def _to_abs_start_period(yyyymm: str, freq: str) -> str:
    """
    Convert pipeline "YYYY-MM" start to ABS API startPeriod format.
    Monthly (M): keep as "YYYY-MM"
    Quarterly (Q): convert to "YYYY-Qn"
    """
    parts = yyyymm.split("-")
    if len(parts) != 2:
        return yyyymm
    year, month = parts[0], int(parts[1])
    if freq.upper() == "Q":
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    return yyyymm


def _period_to_date(period_str: str) -> str:
    """
    Convert ABS period strings to "YYYY-MM-DD".

    "2024-Q1" → "2024-03-01"  (last month of quarter)
    "2024-Q2" → "2024-06-01"
    "2024-Q3" → "2024-09-01"
    "2024-Q4" → "2024-12-01"
    "2024-01" → "2024-01-01"  (monthly, pass through)
    """
    s = str(period_str).strip()
    if "-Q" in s:
        try:
            year, q_part = s.split("-Q")
            last_month = int(q_part) * 3
            return f"{year}-{last_month:02d}-01"
        except (ValueError, IndexError):
            return s
    if len(s) == 7 and s[4] == "-":
        return f"{s}-01"
    return s


class AbsProvider(BaseProvider):
    """
    Australian Bureau of Statistics Data API provider (SDMX REST, CSV output).

    Supports both monthly (HSI_M retail) and quarterly (CPI) ABS dataflows.
    The FREQ dimension in data_key determines how startPeriod and time periods
    are formatted — quarterly series use "YYYY-Qn" format.

    Each series definition must supply:
        meta["dataflow"]  : str  — e.g. "ABS,HSI_M" or "ABS,CPI"
        meta["data_key"]  : str  — SDMX dimension key
        meta["freq"]      : str  — "M" (monthly) or "Q" (quarterly). Default "M".

    Dimension order for HSI_M (retail, monthly):
        MEASURE.TSEST.DATA_ITEM.REGION.FREQ
        e.g. 7.30.CUR.20.AUS.M

    Dimension order for CPI (quarterly):
        MEASURE.INDEX.TSEST.REGION.FREQ
        e.g. 1.10001.10.50.Q
            1     = Index numbers
            10001 = All groups CPI
            10    = Original (NSA)
            50    = Australia
            Q     = Quarterly

    No API key required.
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],   # unused
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        dataflow = meta.get("dataflow")
        data_key = meta.get("data_key")
        freq = meta.get("freq", "M")

        if not dataflow or not data_key:
            raise RuntimeError(
                f"ABS series_id={series_def.series_id} missing 'dataflow' or "
                f"'data_key' in meta"
            )

        start_period = _to_abs_start_period(start, freq)
        params = {"startPeriod": start_period, "format": "csv"}
        url = f"{BASE_URL}/{dataflow}/{data_key}?{urlencode(params)}"

        logger.info(
            "Fetching provider=abs series_id=%s dataflow=%s freq=%s",
            series_def.series_id, dataflow, freq,
        )
        logger.debug("ABS URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=abs series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            text = response.text
            if not text or len(text.strip()) == 0:
                raise RuntimeError(
                    f"ABS returned empty response for series_id={series_def.series_id}"
                )

            first_line = text.split("\n")[0]
            if "TIME_PERIOD" not in first_line and "OBS_VALUE" not in first_line:
                raise RuntimeError(
                    f"ABS response missing expected CSV columns for "
                    f"series_id={series_def.series_id}. Header: {first_line!r}"
                )

            row_count = max(text.count("\n") - 1, 0)
            logger.info(
                "Fetched provider=abs series_id=%s rows=%d",
                series_def.series_id, row_count,
            )
            return text

        except Exception:
            logger.exception("Failed provider=abs series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=abs series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"ABS payload empty for series_id={series_id}")

        df = pd.read_csv(io.StringIO(raw_payload))

        # Normalise column names — ABS CSV may have labels appended after codes
        # e.g. "TIME_PERIOD: Time Period" → normalise to "TIME_PERIOD"
        df.columns = [c.split(":")[0].strip() for c in df.columns]

        col_map = {c.upper(): c for c in df.columns}
        time_col = col_map.get("TIME_PERIOD")
        val_col = col_map.get("OBS_VALUE")

        if not time_col or not val_col:
            raise ValueError(
                f"ABS CSV missing TIME_PERIOD or OBS_VALUE for series_id={series_id}. "
                f"Columns: {list(df.columns)}"
            )

        logger.info(
            "Raw rows (excluding header) series_id=%s rows_in=%d", series_id, len(df)
        )

        # Convert quarterly "YYYY-Qn" or monthly "YYYY-MM" to datetime
        df["time"] = df[time_col].apply(_period_to_date)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["value"] = pd.to_numeric(df[val_col], errors="coerce")

        out = df[["time", "value"]].copy()
        out.insert(0, "series_id", series_id)

        before = len(out)
        out = out.dropna(subset=["time", "value"]).sort_values("time")
        dropped = before - len(out)

        if dropped:
            drop_pct = (dropped / before) * 100
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
