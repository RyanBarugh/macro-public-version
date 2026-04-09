from __future__ import annotations

import time
from datetime import date
from typing import Any, Optional, Tuple

import xlrd
import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

# METI Commercial Dynamics Survey (商業動態統計調査)
# Preliminary report published monthly, ~4 weeks after reference month.
# URL: https://www.meti.go.jp/statistics/tyo/syoudou/result/excel/DB_YYYYMMS.xls
# Each file contains a rolling window of ~15 months. Historical files not retained.
#
# Target sheet : DB_Table2  (業種別商業販売額指数)
# Target column : 小売業計  (located by name scan, not position)
# Target rows   : 季節調整済指数  (located by name scan across label columns)

SHEET_NAME = "DB_Table2"

# Japanese labels — stable semantic anchors, not positional
RETAIL_LABEL_JP = "小売業計"        # column header we search for
SA_INDEX_LABEL  = "季節調整済指数"  # row type label we search for

# Safety bounds for scanning — generous to handle layout drift
HEADER_SCAN_ROWS = 10   # scan first N rows to find the retail column
LABEL_SCAN_COLS  = 8    # scan first N cols of each data row for the row-type label

MAX_DROP_PCT = 5.0

BASE_URL = (
    "https://www.meti.go.jp/statistics/tyo/syoudou/result/excel/DB_{year}{month:02d}S.xls"
)


class MetiProvider(BaseProvider):
    """
    METI Commercial Dynamics Survey — SA retail sales index (2020=100).

    Fetches the monthly preliminary XLS and extracts 小売業計 季節調整済指数
    from DB_Table2. Column and row-type positions are resolved by scanning
    for the Japanese label text, so layout shifts between releases are
    handled automatically without code changes.

    series.json fields:
        backfill.start  : "YYYY-MM"  earliest month to accept from the XLS
        required        : false recommended (preliminary, subject to revision)

    No API key required.
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        series_id = series_def.series_id
        target_months = _months_to_attempt()

        all_records: list[dict] = []
        fetched: set[str] = set()

        for year, month in target_months:
            key = f"{year}{month:02d}"
            if key in fetched:
                continue
            fetched.add(key)

            url = BASE_URL.format(year=year, month=month)
            logger.info("Fetching provider=meti series_id=%s url=%s", series_id, url)

            t0 = time.time()
            try:
                resp = session.get(url, timeout=timeout)
                elapsed_ms = int((time.time() - t0) * 1000)
                logger.info(
                    "HTTP provider=meti series_id=%s status=%s elapsed_ms=%d",
                    series_id, resp.status_code, elapsed_ms,
                )
                if resp.status_code == 404:
                    logger.warning(
                        "provider=meti url=%s 404 — not yet published, skipping", url
                    )
                    continue
                resp.raise_for_status()
            except requests.HTTPError:
                raise
            except Exception:
                logger.exception(
                    "provider=meti failed url=%s series_id=%s", url, series_id
                )
                raise

            records = _parse_table2(resp.content, url, series_id)
            all_records.extend(records)
            logger.info(
                "Parsed provider=meti file=%s rows=%d series_id=%s",
                key, len(records), series_id,
            )

        if not all_records:
            raise RuntimeError(
                f"provider=meti series_id={series_id}: no data retrieved"
            )

        logger.info(
            "provider=meti series_id=%s total_raw_rows=%d files_attempted=%d",
            series_id, len(all_records), len(fetched),
        )
        return {"records": all_records, "start_filter": start}

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=meti series_id=%s strict=%s", series_id, strict)

        records = raw_payload.get("records", [])
        start_filter = raw_payload.get("start_filter")

        if not records:
            raise ValueError(f"provider=meti series_id={series_id}: empty records")

        df = pd.DataFrame(records)
        df["time"]  = pd.to_datetime(df["time"],  errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        before = len(df)
        df = df.dropna(subset=["time", "value"])
        dropped = before - len(df)
        if dropped:
            drop_pct = (dropped / before) * 100
            logger.warning(
                "provider=meti series_id=%s dropped %d/%d rows (%.1f%%)",
                series_id, dropped, before, drop_pct,
            )
            if strict and drop_pct > MAX_DROP_PCT:
                raise ValueError(
                    f"provider=meti series_id={series_id}: dropped {drop_pct:.1f}% "
                    f"of rows (threshold={MAX_DROP_PCT}%). Possible format change."
                )

        if start_filter:
            cutoff = pd.to_datetime(f"{start_filter}-01")
            df = df[df["time"] >= cutoff]

        # Deduplicate — if same month in multiple files, last wins
        df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")

        dupes = df[df["time"].duplicated(keep=False)]
        if not dupes.empty:
            raise ValueError(
                f"provider=meti series_id={series_id}: duplicate times after dedup: "
                f"{dupes['time'].unique().tolist()}"
            )

        out = df[["time", "value"]].copy()
        out.insert(0, "series_id", series_id)
        out["time"] = out["time"].dt.strftime("%Y-%m-%d")

        logger.info("provider=meti series_id=%s rows_out=%d", series_id, len(out))
        return out


# ---------------------------------------------------------------------------
# Parsing — all position resolution is done by label scanning
# ---------------------------------------------------------------------------

def _parse_table2(content: bytes, url: str, series_id: str) -> list[dict]:
    """
    Parse DB_Table2 from raw XLS bytes.

    1. Load sheet into memory.
    2. Scan first HEADER_SCAN_ROWS rows to find column index of 小売業計.
    3. For each data row, scan first LABEL_SCAN_COLS cells for 季節調整済指数.
    4. Parse time code from col 0; extract value from the located retail column.

    Raises descriptively if the retail label is not found — this is the
    canary for a layout change and should not be silently swallowed.
    """
    try:
        wb = xlrd.open_workbook(file_contents=content)
    except Exception as e:
        raise RuntimeError(
            f"provider=meti series_id={series_id}: failed to open workbook "
            f"from {url}: {e}"
        ) from e

    sheet_names = wb.sheet_names()
    if SHEET_NAME not in sheet_names:
        raise RuntimeError(
            f"provider=meti series_id={series_id}: sheet '{SHEET_NAME}' not found "
            f"in {url}. Available sheets: {sheet_names}"
        )

    ws = wb.sheet_by_name(SHEET_NAME)
    rows = [tuple(ws.row_values(i)) for i in range(ws.nrows)]

    # Locate retail column by name
    retail_col = _find_column_by_label(rows, RETAIL_LABEL_JP, series_id, url)
    logger.info(
        "provider=meti series_id=%s resolved '%s' at column index %d",
        series_id, RETAIL_LABEL_JP, retail_col,
    )

    # Extract SA monthly rows
    records = []
    for row in rows[HEADER_SCAN_ROWS:]:
        if not row or len(row) <= retail_col:
            continue

        if not _row_has_label(row, SA_INDEX_LABEL):
            continue

        time_code = row[0]
        if not time_code:
            continue

        time_str = _parse_time_code(str(time_code))
        if time_str is None:
            continue  # quarterly or annual

        raw_val = row[retail_col]
        if raw_val is None:
            continue

        try:
            value = float(raw_val)
        except (TypeError, ValueError):
            logger.warning(
                "provider=meti series_id=%s unparseable value=%r at time=%s",
                series_id, raw_val, time_str,
            )
            continue

        records.append({"time": time_str, "value": value})

    return records


def _find_column_by_label(
    rows: list[tuple],
    label: str,
    series_id: str,
    url: str,
) -> int:
    """
    Scan the first HEADER_SCAN_ROWS rows for a cell exactly matching `label`.
    Returns column index. Raises RuntimeError if not found.
    """
    for row in rows[:HEADER_SCAN_ROWS]:
        for col_idx, cell in enumerate(row):
            if cell == label:
                return col_idx
    raise RuntimeError(
        f"provider=meti series_id={series_id}: column label '{label}' not found "
        f"in first {HEADER_SCAN_ROWS} rows of {url}. "
        f"Sheet layout may have changed — manual inspection required."
    )


def _row_has_label(row: tuple, label: str) -> bool:
    """
    Return True if any of the first LABEL_SCAN_COLS cells exactly matches `label`.
    """
    for cell in row[:LABEL_SCAN_COLS]:
        if cell == label:
            return True
    return False


# ---------------------------------------------------------------------------
# Time code parsing
# ---------------------------------------------------------------------------

def _parse_time_code(code: str) -> str | None:
    """
    Parse METI 10-digit time codes to "YYYY-MM-01".

    YYYYAABBCC where:
      Annual    : AAABBBCC == 000000       → None
      Quarterly : BB != CC                 → None
      Monthly   : BB == CC, both non-zero  → "YYYY-BB-01"
    """
    code = code.strip()
    if len(code) != 10 or not code.isdigit():
        return None

    if code[4:] == "000000":
        return None

    year  = code[:4]
    mid2  = code[6:8]
    last2 = code[8:10]

    if mid2 == last2 and mid2 != "00":
        try:
            m = int(mid2)
            if 1 <= m <= 12:
                return f"{year}-{mid2}-01"
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Fetch targeting
# ---------------------------------------------------------------------------

def _months_to_attempt() -> list[tuple[int, int]]:
    """
    Return (year, month) tuples to attempt, newest first.
    Try current month + 2 prior. Each file has a ~15 month rolling window
    so the latest successfully fetched file contains all the data we need.
    404 on current month = not yet published; prior month file is the fallback.
    """
    today = date.today()
    results = []
    y, m = today.year, today.month
    for _ in range(3):
        results.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return results