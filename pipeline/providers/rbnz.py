from __future__ import annotations

import io
import time
from typing import Any, Optional, Tuple

import pandas as pd
import requests

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    cffi_requests = None

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

# RBNZ B2 wholesale interest rates — secondary market government bond closing yields
# Sheet "Data", rows 0-4 = metadata/headers, row 5+ = data
# Columns: 0=date, 9=2Y, 10=5Y, 11=10Y
# Series IDs: INM.DG102.NZZCF, INM.DG105.NZZCF, INM.DG110.NZZCF
# Daily frequency — resample to monthly in derived layer

# Two files required to cover full history:
#   Historical: 1985-2017 closing yields
#   Current:    2018-present closing yields
RBNZ_B2_HISTORICAL_URL = (
    "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics"
    "/series/b/b2/hb2-daily-close-1985-2017.xlsx"
)
RBNZ_B2_CURRENT_URL = (
    "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics"
    "/series/b/b2/hb2-daily-close.xlsx"
)

# Column indices differ between the two files — historical has 19 cols, current has 47
# Historical (1985-2017): 2Y=col7, 5Y=col8, 10Y=col9
# Current    (2018-now):  2Y=col9, 5Y=col10, 10Y=col11
SERIES_COL_MAP_HISTORICAL = {
    "INM.DG102.NZZCF": 7,   # 2-year government bond
    "INM.DG105.NZZCF": 8,   # 5-year government bond
    "INM.DG110.NZZCF": 9,   # 10-year government bond
}
SERIES_COL_MAP_CURRENT = {
    "INM.DG102.NZZCF": 9,   # 2-year government bond
    "INM.DG105.NZZCF": 10,  # 5-year government bond
    "INM.DG110.NZZCF": 11,  # 10-year government bond
}

# Module-level cache — download once per process, reuse for all RBNZ series
_XLSX_CACHE: dict[str, bytes] = {}


def _fetch_xlsx(url: str, session: requests.Session, label: str) -> bytes:
    """Download an XLSX file, using module-level cache to avoid re-downloading.

    Uses curl_cffi to impersonate Chrome's TLS fingerprint, bypassing
    Cloudflare's bot detection on rbnz.govt.nz. Falls back to plain
    requests if curl_cffi is not installed.
    """
    if url in _XLSX_CACHE:
        logger.info("provider=rbnz using cached xlsx label=%s bytes=%d", label, len(_XLSX_CACHE[url]))
        return _XLSX_CACHE[url]

    headers = {
        "Referer": "https://www.rbnz.govt.nz/statistics/series/exchange-and-interest-rates/wholesale-interest-rates",
        "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
        "Accept-Language": "en-GB,en;q=0.9",
    }

    start_ts = time.time()

    if cffi_requests is not None:
        logger.info("provider=rbnz using curl_cffi (chrome TLS) for label=%s", label)
        response = cffi_requests.get(
            url,
            headers=headers,
            impersonate="chrome",
            timeout=(30, 120),
        )
    else:
        logger.warning("provider=rbnz curl_cffi not installed, falling back to plain requests for label=%s", label)
        headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
        response = session.get(url, headers=headers, timeout=(30, 120))

    elapsed_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "HTTP provider=rbnz label=%s status=%s elapsed_ms=%d",
        label, response.status_code, elapsed_ms,
    )
    response.raise_for_status()
    _XLSX_CACHE[url] = response.content
    logger.info("Downloaded provider=rbnz label=%s bytes=%d", label, len(response.content))
    return _XLSX_CACHE[url]


def _parse_xlsx(xlsx_bytes: bytes, col_idx: int) -> list[dict]:
    """Extract date/value records from a B2 XLSX file for a given column index."""
    df_raw = pd.read_excel(
        io.BytesIO(xlsx_bytes),
        sheet_name="Data",
        engine="openpyxl",
        header=None,
    )

    # Rows 0-4 are metadata; data starts at row 5
    data_rows = df_raw.iloc[5:].copy()
    data_rows.columns = range(len(data_rows.columns))

    records = []
    for _, row in data_rows.iterrows():
        date_val = row[0]
        val = row[col_idx]
        if pd.isna(date_val) or val == "" or pd.isna(val):
            continue
        try:
            date_dt = pd.to_datetime(date_val)
            value = float(val)
            records.append({"date": date_dt.strftime("%Y-%m-%d"), "value": value})
        except (TypeError, ValueError):
            continue

    return records


class RbnzProvider(BaseProvider):
    """
    Reserve Bank of New Zealand wholesale interest rates provider.

    Downloads the B2 daily closing yield Excel files from rbnz.govt.nz and
    extracts the requested government bond maturity series. Concatenates
    the historical (2010-2017) and current (2018-present) files to provide
    full history.

    Required meta fields in series.json:
        rbnz_series_id : str — RBNZ series ID, e.g. "INM.DG102.NZZCF"
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        rbnz_series_id = meta.get("rbnz_series_id")

        if not rbnz_series_id:
            raise RuntimeError(
                f"RBNZ series_id={series_def.series_id} missing 'rbnz_series_id' in meta"
            )

        if rbnz_series_id not in SERIES_COL_MAP_CURRENT:
            raise RuntimeError(
                f"RBNZ series_id={series_def.series_id} unknown rbnz_series_id={rbnz_series_id}. "
                f"Known: {list(SERIES_COL_MAP_CURRENT)}"
            )

        logger.info(
            "Fetching provider=rbnz series_id=%s rbnz_series_id=%s",
            series_def.series_id, rbnz_series_id,
        )

        # Fetch historical file (1985-2017) — 404 is non-fatal, log and skip
        records_historical: list[dict] = []
        try:
            hist_bytes = _fetch_xlsx(RBNZ_B2_HISTORICAL_URL, session, "historical-1985-2017")
            records_historical = _parse_xlsx(hist_bytes, SERIES_COL_MAP_HISTORICAL[rbnz_series_id])
            logger.info(
                "Parsed provider=rbnz historical rows=%d series_id=%s",
                len(records_historical), series_def.series_id,
            )
        except requests.HTTPError as exc:
            logger.warning(
                "provider=rbnz historical file unavailable status=%s series_id=%s — skipping",
                exc.response.status_code if exc.response is not None else "?",
                series_def.series_id,
            )

        # Fetch current file (2018-present) — required
        current_bytes = _fetch_xlsx(RBNZ_B2_CURRENT_URL, session, "current-2018-present")
        records_current = _parse_xlsx(current_bytes, SERIES_COL_MAP_CURRENT[rbnz_series_id])
        logger.info(
            "Parsed provider=rbnz current rows=%d series_id=%s",
            len(records_current), series_def.series_id,
        )

        all_records = records_historical + records_current

        logger.info(
            "Fetched provider=rbnz series_id=%s total_rows=%d",
            series_def.series_id, len(all_records),
        )

        return {"rbnz_series_id": rbnz_series_id, "start": start, "records": all_records}

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=rbnz series_id=%s", series_id)

        records = raw_payload.get("records", [])
        start = raw_payload.get("start", "1985-01")

        if not records:
            raise ValueError(f"RBNZ: no records extracted for series_id={series_id}")

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["time", "value"])

        # Apply start filter
        try:
            start_dt = pd.to_datetime(start + "-01")
            df = df[df["time"] >= start_dt]
        except Exception:
            pass

        df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")

        before = len(df)
        dropped = before - len(df)
        if dropped:
            drop_pct = (dropped / max(before, 1)) * 100
            if strict and drop_pct > MAX_DROP_PCT:
                raise ValueError(
                    f"series_id={series_id}: dropped {drop_pct:.1f}% of rows"
                )

        out = df[["time", "value"]].copy()
        out.insert(0, "series_id", series_id)

        logger.info(
            "Cleaned rows series_id=%s rows_out=%d range=%s → %s",
            series_id, len(out),
            out["time"].iloc[0].strftime("%Y-%m-%d") if len(out) else "n/a",
            out["time"].iloc[-1].strftime("%Y-%m-%d") if len(out) else "n/a",
        )

        out["time"] = out["time"].dt.strftime("%Y-%m-%d")
        return out