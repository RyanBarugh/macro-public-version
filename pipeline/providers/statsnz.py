from __future__ import annotations

import io
import time
import zipfile
from datetime import date
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

# Stats NZ publishes full-history CSV dumps with each quarterly release.
# No auth required — plain GET to a predictable URL.
#
# Two release families used:
#
#   CPI SA:  .../Consumers-price-index/Consumers-price-index-{Month}-{year}-quarter/
#            Download-data/consumers-price-index-{month}-{year}-quarter-seasonally-adjusted.csv
#
#   BPI:     .../Business-price-indexes/Business-price-indexes-{Month}-{year}-quarter/
#            Download-data/Business-price-indexes-{month}-{year}-quarter-csv.zip  (contains .csv)
#
# Each series definition must supply in meta:
#     url_template : str  — URL with {Month}, {month}, {year} placeholders
#     series_ref   : str  — Series_reference to filter on, e.g. "CPIQ.SE9SA"
#     zip_filename : str  — (optional) substring to match CSV filename in
#                           multi-file ZIPs, e.g. "hlfs" or "qes"
#
# The provider resolves the latest completed quarter, constructs the URL,
# downloads the CSV (or ZIP containing CSV), and filters to the target series.

QUARTER_MONTHS = {1: ("March", "march"), 2: ("June", "june"),
                  3: ("September", "september"), 4: ("December", "december")}


def _latest_quarter(today: Optional[date] = None) -> Tuple[str, str, str]:
    """Return (Month, month, year) for the latest completed quarter.

    Stats NZ CPI releases ~3 weeks after quarter-end, BPI ~5 weeks.
    We use a 6-week buffer to avoid requesting a file before it exists.
    """
    d = today or date.today()
    # Walk back until we find a quarter whose release is likely out
    # Quarter ends: Mar 31, Jun 30, Sep 30, Dec 31
    # With 6-week buffer: May 12, Aug 11, Nov 11, Feb 11
    quarter_ends = [
        (d.year, 12, 31, 4),
        (d.year, 9, 30, 3),
        (d.year, 6, 30, 2),
        (d.year, 3, 31, 1),
        (d.year - 1, 12, 31, 4),
        (d.year - 1, 9, 30, 3),
    ]
    for y, m, day, q in quarter_ends:
        quarter_end = date(y, m, day)
        release_ready = date(y + (1 if m == 12 else 0),
                            (m + 2 - 1) % 12 + 1 if m != 12 else 2,
                            15)
        if d >= release_ready:
            month_title, month_lower = QUARTER_MONTHS[q]
            return month_title, month_lower, str(y)

    # Fallback: two quarters ago
    fallback_q = ((d.month - 1) // 3) - 1
    if fallback_q <= 0:
        fallback_q += 4
        y = d.year - 1
    else:
        y = d.year
    month_title, month_lower = QUARTER_MONTHS[fallback_q]
    return month_title, month_lower, str(y)


def _previous_quarter(month_title: str, year: str) -> Tuple[str, str, str]:
    """Given a quarter (Month, year), return the previous quarter."""
    _order = ["March", "June", "September", "December"]
    idx = _order.index(month_title)
    if idx == 0:
        prev_q = 4
        prev_year = str(int(year) - 1)
    else:
        prev_q = idx  # 1-indexed: June=2 -> prev is March=1
        prev_year = year
    mt, ml = QUARTER_MONTHS[prev_q]
    return mt, ml, prev_year


class StatsNzCsvProvider(BaseProvider):
    """
    Statistics New Zealand — bulk CSV downloads.

    Downloads full-history CSV files from Stats NZ quarterly releases.
    No API key required. Files are static assets on stats.govt.nz.

    Supports both plain CSV and ZIP-wrapped CSV (auto-detected).

    series.json meta fields:
        url_template : str — URL with {Month}, {month}, {year} placeholders
        series_ref   : str — exact Series_reference to filter rows on
        zip_filename : str (optional) — substring to match CSV filename within
                       multi-file ZIPs (e.g. "hlfs", "qes"). If omitted,
                       first CSV in the ZIP is used.
    """

    def __init__(self):
        # In-memory cache for parsed DataFrames within a single pipeline run.
        # Key: (resolved_url, zip_filename or None)
        # Value: normalised pd.DataFrame with Series_reference, Period, Data_value
        self._df_cache: dict[tuple, pd.DataFrame] = {}

    def _get_dataframe(
        self,
        url: str,
        zip_filename: Optional[str],
        series_id: str,
        session: requests.Session,
        timeout: Tuple[float, float],
    ) -> pd.DataFrame:
        """Download, extract, parse and cache a Stats NZ CSV.

        Returns the full normalised DataFrame (all series in the file).
        Subsequent calls with the same (url, zip_filename) return the cache.
        """
        cache_key = (url, zip_filename)
        if cache_key in self._df_cache:
            logger.info(
                "Cache hit provider=statsnz_csv series_id=%s "
                "cache_key=(%s, %s)",
                series_id, url.split("/")[-1], zip_filename,
            )
            return self._df_cache[cache_key]

        start_ts = time.time()
        response = session.get(url, timeout=timeout)
        elapsed_ms = int((time.time() - start_ts) * 1000)
        logger.info(
            "HTTP provider=statsnz_csv series_id=%s status=%s elapsed_ms=%d",
            series_id, response.status_code, elapsed_ms,
        )
        response.raise_for_status()

        raw_bytes = response.content

        # Auto-detect ZIP: check magic bytes (PK\x03\x04)
        if raw_bytes[:4] == b"PK\x03\x04":
            logger.info("Detected ZIP archive, extracting CSV")
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                csv_names = [n for n in zf.namelist()
                             if n.lower().endswith(".csv")]
                if not csv_names:
                    raise RuntimeError(
                        f"ZIP contains no CSV files: {zf.namelist()}"
                    )
                # Optional: filter to specific file in multi-file ZIPs
                if zip_filename:
                    matched = [n for n in csv_names
                               if zip_filename in n.lower()]
                    if not matched:
                        raise RuntimeError(
                            f"No CSV matching zip_filename='{zip_filename}' "
                            f"in ZIP. Found: {csv_names}"
                        )
                    csv_names = matched
                    logger.info(
                        "zip_filename='%s' matched: %s",
                        zip_filename, csv_names[0],
                    )
                csv_text = zf.read(csv_names[0]).decode("utf-8-sig")
        else:
            csv_text = raw_bytes.decode("utf-8-sig")

        # Parse CSV — only load the 3 columns we need to avoid OOM
        # on large files like the Stats NZ labour ZIP (500k+ rows × 13 cols)
        needed = {"series_reference", "period", "data_value"}
        df = pd.read_csv(
            io.StringIO(csv_text),
            usecols=lambda c: c.lower().replace(" ", "_") in needed,
            low_memory=True,
        )
        del csv_text  # free raw text before normalisation

        # Normalise column names
        col_map = {}
        for col in df.columns:
            low = col.lower().replace(" ", "_")
            if "series_ref" in low:
                col_map[col] = "Series_reference"
            elif low == "period":
                col_map[col] = "Period"
            elif "data_val" in low:
                col_map[col] = "Data_value"
        df = df.rename(columns=col_map)

        logger.info(
            "Parsed provider=statsnz_csv total_rows=%d cache_key=(%s, %s)",
            len(df), url.split("/")[-1], zip_filename,
        )

        self._df_cache[cache_key] = df
        return df

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        url_template = meta.get("url_template")
        series_ref = meta.get("series_ref")
        zip_filename = meta.get("zip_filename")

        if not url_template:
            raise RuntimeError(
                f"StatsNzCsv series_id={series_def.series_id} missing "
                f"'url_template' in meta"
            )
        if not series_ref:
            raise RuntimeError(
                f"StatsNzCsv series_id={series_def.series_id} missing "
                f"'series_ref' in meta"
            )

        month_title, month_lower, year = _latest_quarter()
        url = url_template.format(
            Month=month_title, month=month_lower, year=year
        )

        logger.info(
            "Fetching provider=statsnz_csv series_id=%s url=%s",
            series_def.series_id, url,
        )

        try:
            df = self._get_dataframe(
                url, zip_filename, series_def.series_id, session, timeout,
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # Publication lag — try previous quarter
                prev_title, prev_lower, prev_year = _previous_quarter(
                    month_title, year
                )
                url = url_template.format(
                    Month=prev_title, month=prev_lower, year=prev_year
                )
                logger.warning(
                    "404 on latest quarter — retrying previous quarter: %s",
                    url,
                )
                df = self._get_dataframe(
                    url, zip_filename, series_def.series_id, session, timeout,
                )
            else:
                raise

        try:
            total_rows = len(df)
            filtered = df[df["Series_reference"] == series_ref].copy()

            logger.info(
                "Fetched provider=statsnz_csv series_id=%s "
                "total_rows=%d filtered_rows=%d series_ref=%s",
                series_def.series_id, total_rows, len(filtered), series_ref,
            )

            if filtered.empty:
                raise RuntimeError(
                    f"No rows matching series_ref='{series_ref}' in CSV. "
                    f"Check series_ref in series.json."
                )

            return {
                "series_ref": series_ref,
                "rows": filtered.to_dict("records"),
            }

        except Exception:
            logger.exception(
                "Failed provider=statsnz_csv series_id=%s", series_def.series_id
            )
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info(
            "Cleaning provider=statsnz_csv series_id=%s strict=%s",
            series_id, strict,
        )

        if not raw_payload or "rows" not in raw_payload:
            raise ValueError(
                f"StatsNzCsv payload empty for series_id={series_id}"
            )

        df = pd.DataFrame(raw_payload["rows"])

        # Parse Period: "2025.12" → "2025-12-01", "2025.03" → "2025-03-01"
        def _period_to_date(p: str) -> str:
            try:
                parts = str(p).split(".")
                year = parts[0]
                month = parts[1].zfill(2)
                return f"{year}-{month}-01"
            except Exception:
                return p

        df["time"] = df["Period"].apply(_period_to_date)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["value"] = pd.to_numeric(df["Data_value"], errors="coerce")

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
                f"series_id={series_id}: duplicate time values detected: "
                f"{dupe_times}"
            )

        logger.info(
            "Cleaned rows series_id=%s rows_out=%d", series_id, len(out)
        )
        out["time"] = out["time"].dt.strftime("%Y-%m-%d")
        return out