from __future__ import annotations

import io
import time
import zipfile
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

# BoE daily nominal gilt spot curve — full archive ZIP
# Contains 8 Excel files covering 1979 to present.
# Sheet "4. nominal spot curve" has yearly maturities (0.5, 1, 2, 5, 10, etc.)
# Sheet "3. nominal spot, short end" has monthly maturities — NOT what we want.
#
# Required meta fields in series.json (top-level):
#   boe_maturity : int — maturity in whole years (2, 5, 10, etc.)

NOMINAL_ZIP_URL = (
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/"
    "yield-curves/glcnominalddata.zip"
)

TARGET_SHEET_CANDIDATES = [
    "4. nominal spot curve",  # older files (pre-2005)
    "4. spot curve",          # newer files (2005+)
    "4.spot curve",           # variant without space after dot
    "4. Nominal spot curve",  # capitalisation variant
    "4. Spot curve",          # capitalisation variant
    "4. Spot Curve",          # title case variant
]


def _find_spot_curve_sheet(xl: pd.ExcelFile, xlsx_name: str) -> str:
    """Find the nominal spot curve sheet, with fuzzy fallback."""
    # Exact match first
    for candidate in TARGET_SHEET_CANDIDATES:
        if candidate in xl.sheet_names:
            return candidate

    # Fuzzy: case-insensitive, look for "spot curve" but not "short end"
    for s in xl.sheet_names:
        lower = s.lower()
        if "spot curve" in lower and "short" not in lower:
            return s

    # Fuzzy: any sheet with "spot" and "nominal"
    for s in xl.sheet_names:
        lower = s.lower()
        if "spot" in lower and "nominal" in lower:
            return s

    raise ValueError(
        f"No spot curve sheet found in {xlsx_name}. "
        f"Available sheets: {xl.sheet_names}"
    )


def _is_numeric(v) -> bool:
    """Return True if v is a real finite number (not NaN, not string label)."""
    try:
        f = float(v)
        return not (f != f)  # NaN check: NaN != NaN
    except (TypeError, ValueError):
        return False


class BoeProvider(BaseProvider):
    """
    Bank of England daily nominal gilt spot curve provider.

    Downloads the full archive ZIP, reads ALL period Excel files,
    extracts the requested maturity from the yearly spot curve sheet,
    and returns pre-parsed records (JSON-serializable) for S3 storage.
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
        maturity = meta.get("boe_maturity") or (meta.get("meta") or {}).get("boe_maturity")

        if maturity is None:
            raise RuntimeError(
                f"BoE series_id={series_def.series_id} missing 'boe_maturity' in meta"
            )

        maturity = int(maturity)

        logger.info(
            "Fetching provider=boe series_id=%s maturity=%sY",
            series_def.series_id, maturity,
        )

        start_ts = time.time()
        response = session.get(NOMINAL_ZIP_URL, timeout=(30, 180), stream=True)
        elapsed_ms = int((time.time() - start_ts) * 1000)
        logger.info(
            "HTTP provider=boe series_id=%s status=%s elapsed_ms=%d",
            series_def.series_id, response.status_code, elapsed_ms,
        )
        response.raise_for_status()

        zip_bytes = response.content
        logger.info("Downloaded provider=boe bytes=%d", len(zip_bytes))

        # Open ZIP and read ALL Excel files — they cover different date ranges
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        xlsx_files = sorted(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        logger.info("BoE ZIP files: %s", xlsx_files)

        all_records = []

        for xlsx_name in xlsx_files:
            try:
                records = self._extract_file(zf, xlsx_name, maturity, series_def.series_id)
                all_records.extend(records)
                logger.info("  %s → %d rows", xlsx_name, len(records))
            except Exception as e:
                logger.warning("  %s → SKIPPED: %s", xlsx_name, e)
                # Show sheet names for debugging
                try:
                    xl_debug = pd.ExcelFile(io.BytesIO(zf.read(xlsx_name)), engine="openpyxl")
                    logger.warning("    Available sheets: %s", xl_debug.sheet_names)
                except Exception:
                    pass

        logger.info(
            "Fetched provider=boe series_id=%s total_rows=%d",
            series_def.series_id, len(all_records),
        )

        # Return JSON-serializable list of {date, value} dicts
        return {"maturity": maturity, "start": start, "records": all_records}

    def _extract_file(self, zf, xlsx_name: str, maturity: int, series_id: str) -> list:
        """Extract the target maturity from one Excel file."""
        xlsx_bytes = zf.read(xlsx_name)
        xl = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")

        # Find correct sheet name — exact match then fuzzy fallback
        sheet_name = _find_spot_curve_sheet(xl, xlsx_name)

        # Read raw — spot curve sheet structure:
        # Row 0: title (e.g. "UK nominal spot curve")
        # Row 1: "Maturity" label
        # Row 2: maturity values in years (0.5, 1, 2, 5, 10 ...)
        # Row 3+: data (col 0 = date, col 1+ = yields)
        df_raw = pd.read_excel(
            io.BytesIO(xlsx_bytes),
            sheet_name=sheet_name,
            engine="openpyxl",
            header=None,
        )

        # Find the maturity header row — the row where most values are real finite numbers
        # (maturity years like 0.5, 1, 2, 5, 10). NaN rows and string label rows are skipped.
        header_row = None
        for i in range(min(8, len(df_raw))):
            row = df_raw.iloc[i, 1:]  # skip col 0 (date column)
            numeric_count = sum(1 for v in row if _is_numeric(v))
            if numeric_count >= 5:
                header_row = i
                break

        if header_row is None:
            raise ValueError(f"Could not find maturity header row in {xlsx_name}")

        # Re-read with correct header
        df = pd.read_excel(
            io.BytesIO(xlsx_bytes),
            sheet_name=sheet_name,
            engine="openpyxl",
            header=header_row,
        )

        # First column is date
        date_col = df.columns[0]
        df["_time"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_time"])

        # Find exact maturity column (float match)
        target_f = float(maturity)
        maturity_col = None
        for col in df.columns[1:]:
            try:
                if abs(float(str(col).strip()) - target_f) < 0.01:
                    maturity_col = col
                    break
            except (TypeError, ValueError):
                continue

        if maturity_col is None:
            available = [c for c in df.columns[1:] if pd.notna(c)][:15]
            raise ValueError(
                f"Maturity {maturity}Y not found. Available: {available}"
            )

        df["_value"] = pd.to_numeric(df[maturity_col], errors="coerce")
        df = df.dropna(subset=["_value"])

        return [
            {"date": row["_time"].strftime("%Y-%m-%d"), "value": float(row["_value"])}
            for _, row in df.iterrows()
        ]

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=boe series_id=%s", series_id)

        records = raw_payload.get("records", [])
        start   = raw_payload.get("start", "1979-01")

        if not records:
            raise ValueError(f"BoE: no records extracted for series_id={series_id}")

        df = pd.DataFrame(records)
        df["time"]  = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["time", "value"])

        # Apply start filter
        try:
            start_dt = pd.to_datetime(start + "-01")
            df = df[df["time"] >= start_dt]
        except Exception:
            pass

        # Sort and deduplicate
        df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")

        before = len(df)
        df = df.dropna(subset=["time", "value"])
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