from __future__ import annotations

import io
import time
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

# RBA F2 daily capital market yields — government bonds
# Sheet "Data", row 11 = Series IDs, row 12+ = data
# Columns: 0=date, 1=2Y, 2=3Y, 3=5Y, 4=10Y, 5=indexed
# Series IDs: FCMYGBAG2D, FCMYGBAG5D, FCMYGBAG10D
# Daily frequency — resample to monthly in derived layer

RBA_F2_URL = "https://www.rba.gov.au/statistics/tables/xls/f02d.xlsx"

# Map series_id suffix to column index in the Data sheet
SERIES_COL_MAP = {
    "FCMYGBAG2D":  1,
    "FCMYGBAG5D":  3,
    "FCMYGBAG10D": 4,
}

# Module-level cache — download once per process, reuse for all RBA series
_XLSX_CACHE: bytes | None = None


class RbaProvider(BaseProvider):
    """
    Reserve Bank of Australia capital market yields provider.

    Downloads the F2 daily Excel file directly from rba.gov.au and
    extracts the requested bond maturity series.

    Required meta fields in series.json:
        rba_series_id : str — RBA series ID, e.g. "FCMYGBAG2D"
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
        rba_series_id = meta.get("rba_series_id")

        if not rba_series_id:
            raise RuntimeError(
                f"RBA series_id={series_def.series_id} missing 'rba_series_id' in meta"
            )

        logger.info(
            "Fetching provider=rba series_id=%s rba_series_id=%s",
            series_def.series_id, rba_series_id,
        )

        start_ts = time.time()
        global _XLSX_CACHE
        if _XLSX_CACHE is None:
            response = session.get(RBA_F2_URL, timeout=(30, 120))
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=rba series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()
            _XLSX_CACHE = response.content
            logger.info("Downloaded provider=rba bytes=%d", len(_XLSX_CACHE))
        else:
            logger.info("provider=rba using cached xlsx bytes=%d", len(_XLSX_CACHE))

        xlsx_bytes = _XLSX_CACHE

        # Parse the Data sheet — dynamically find the series ID header row
        # instead of hardcoding row 11, which breaks when RBA adds/removes header rows
        df_raw = pd.read_excel(
            io.BytesIO(xlsx_bytes),
            sheet_name="Data",
            engine="openpyxl",
            header=None,
        )

        # Search the first 20 rows for the one containing the RBA series ID string
        header_row_idx = None
        col_idx = None
        for i in range(min(20, len(df_raw))):
            row_vals = df_raw.iloc[i].astype(str).tolist()
            for j, val in enumerate(row_vals):
                if rba_series_id in val:
                    header_row_idx = i
                    col_idx = j
                    break
            if header_row_idx is not None:
                break

        if header_row_idx is None or col_idx is None:
            # Dump first rows for debugging
            for i in range(min(15, len(df_raw))):
                logger.warning("  RBA row %d: %s", i, df_raw.iloc[i].tolist()[:6])
            raise RuntimeError(
                f"RBA: Could not find '{rba_series_id}' in first 20 rows of Data sheet. "
                f"The RBA may have restructured the spreadsheet."
            )

        logger.info(
            "  RBA series '%s' found at row=%d col=%d",
            rba_series_id, header_row_idx, col_idx,
        )

        # Data starts on the row after the series ID header
        data_rows = df_raw.iloc[header_row_idx + 1:].copy()
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

        logger.info(
            "Fetched provider=rba series_id=%s rows=%d",
            series_def.series_id, len(records),
        )

        return {"rba_series_id": rba_series_id, "start": start, "records": records}

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=rba series_id=%s", series_id)

        records = raw_payload.get("records", [])
        start = raw_payload.get("start", "2013-01")

        if not records:
            raise ValueError(f"RBA: no records extracted for series_id={series_id}")

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