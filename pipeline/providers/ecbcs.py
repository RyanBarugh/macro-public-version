from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0
MAX_FALLBACK_MONTHS = 4

# EC BCS ZIP base URL — versioned monthly by YYMM slug
# e.g. nace2_ecfin_2602 = February 2026
BASE_URL = (
    "https://ec.europa.eu/economy_finance/db_indicators/surveys/"
    "documents/series/nace2_ecfin_{yymm}/{zip_type}.zip"
)

PAGE_URL = (
    "https://economy-finance.ec.europa.eu/economic-forecast-and-surveys/"
    "business-and-consumer-surveys/download-business-and-consumer-survey-data/"
    "time-series_en"
)

# EC blocks non-browser requests — mimic a real browser
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": PAGE_URL,
    "Accept": "application/zip, application/octet-stream, */*",
}

# Default sheet name — most BCS XLSXs use MONTHLY
# Labour hoarding uses AGGREGATE — override via bcs_sheet in series.json
DEFAULT_SHEET = "MONTHLY"


def _yymm(year: int, month: int) -> str:
    return f"{str(year)[2:]}{month:02d}"


def _prior_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _fetch_zip(
    zip_type: str,
    session: requests.Session,
    timeout: Tuple[float, float],
    series_id: str,
) -> tuple[str, requests.Response]:
    """
    Try current month's ZIP URL, falling back month-by-month.
    Validates response is actually a ZIP (not an HTML error page)
    by checking Content-Type before accepting the result.
    Returns (resolved_url, response) so the caller doesn't re-download.
    """
    now = datetime.utcnow()
    year, month = now.year, now.month

    for attempt in range(MAX_FALLBACK_MONTHS):
        slug = _yymm(year, month)
        url = BASE_URL.format(yymm=slug, zip_type=zip_type)
        logger.debug(
            "EC BCS attempt=%d series_id=%s slug=%s url=%s",
            attempt + 1, series_id, slug, url,
        )
        try:
            resp = session.get(
                url,
                timeout=(timeout[0], 60.0),
                headers=_HEADERS,
                allow_redirects=True,
            )
            content_type = resp.headers.get("Content-Type", "")
            logger.debug(
                "EC BCS slug=%s status=%d content_type=%s",
                slug, resp.status_code, content_type,
            )
            if resp.status_code == 200 and "html" not in content_type.lower():
                logger.info(
                    "EC BCS resolved ZIP series_id=%s slug=%s",
                    series_id, slug,
                )
                return url, resp

            logger.debug(
                "EC BCS slug=%s rejected (status=%d content_type=%s) — trying prior month",
                slug, resp.status_code, content_type,
            )
        except requests.RequestException as e:
            logger.debug("EC BCS GET failed slug=%s: %s", slug, e)

        year, month = _prior_month(year, month)

    raise RuntimeError(
        f"EC BCS could not resolve a valid ZIP for series_id={series_id} "
        f"zip_type={zip_type} after {MAX_FALLBACK_MONTHS} attempts. "
        f"Check that the EC BCS time series page is reachable: {PAGE_URL}"
    )


class EcBcsProvider(BaseProvider):
    """
    European Commission Business and Consumer Surveys (BCS) provider.

    Data is distributed as monthly ZIP files containing XLSXs. The ZIP URL
    is versioned by a YYMM slug (e.g. nace2_ecfin_2602 for Feb 2026).
    This provider resolves the current slug by probing from the current
    month backwards, validating via Content-Type that a real ZIP was returned
    (EC returns HTML 200s for missing slugs, not 404s).

    Most BCS XLSXs store time series on the 'MONTHLY' sheet. The labour
    hoarding XLSX uses 'AGGREGATE' — override via bcs_sheet in series.json.
    Date column is always first (stored as end-of-month timestamps, normalised
    to month start by this provider). Series columns use short keys e.g.
    'EA.ESI', 'EA.EEI', 'EA.CONS', 'EA.LH'.

    No API key required — data is freely available from DG ECFIN.

    Required meta fields in series.json:
        bcs_zip        : str — ZIP filename without extension, e.g.
                               "main_indicators_sa_nace2"
        bcs_series_key : str — Column name in the sheet, e.g. "EA.ESI"

    Optional meta fields:
        bcs_sheet      : str — Sheet name to read (default: "MONTHLY")
                               Use "AGGREGATE" for labour hoarding.

    fetch() raw payload:
        {
            "zip_url":    str,           # resolved ZIP URL
            "series_key": str,           # column extracted
            "data": [[date_str, value]]  # list of [YYYY-MM-DD, float] pairs
        }
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — EC BCS is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        series_id = series_def.series_id

        bcs_zip = meta.get("bcs_zip")
        bcs_series_key = meta.get("bcs_series_key")
        bcs_sheet = meta.get("bcs_sheet", DEFAULT_SHEET)

        if not bcs_zip:
            raise RuntimeError(
                f"EC BCS series_id={series_id} missing 'bcs_zip' in meta"
            )
        if not bcs_series_key:
            raise RuntimeError(
                f"EC BCS series_id={series_id} missing 'bcs_series_key' in meta"
            )

        logger.info("Fetching provider=ecbcs series_id=%s", series_id)

        # Step 1: resolve current ZIP URL and download in one shot
        start_ts = time.time()
        zip_url, resp = _fetch_zip(bcs_zip, session, timeout, series_id)
        elapsed_ms = int((time.time() - start_ts) * 1000)

        logger.info(
            "HTTP provider=ecbcs series_id=%s status=%s elapsed_ms=%d",
            series_id, resp.status_code, elapsed_ms,
        )

        # Step 2: unzip in memory
        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile as e:
            raise RuntimeError(
                f"EC BCS response is not a valid ZIP for series_id={series_id}: {e}. "
                f"Content-Type was: {resp.headers.get('Content-Type')}. "
                f"Response size: {len(resp.content)} bytes."
            )

        # Step 3: find the XLSX
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise RuntimeError(
                f"EC BCS ZIP contains no XLSX files for series_id={series_id}. "
                f"Contents: {zf.namelist()}"
            )

        xlsx_name = xlsx_names[0]
        logger.debug(
            "EC BCS reading XLSX=%s sheet=%s for series_id=%s",
            xlsx_name, bcs_sheet, series_id,
        )

        with zf.open(xlsx_name) as f:
            df_raw = pd.read_excel(
                f, engine="openpyxl", sheet_name=bcs_sheet, header=0
            )

        # Step 4: locate the series column
        if bcs_series_key not in df_raw.columns:
            available = [c for c in df_raw.columns if not str(c).startswith("Unnamed")][:10]
            raise RuntimeError(
                f"EC BCS series_key='{bcs_series_key}' not found in sheet='{bcs_sheet}' "
                f"for series_id={series_id}. "
                f"Sample available keys: {available}"
            )

        # Date column is always first — stored as end-of-month timestamps
        date_col = df_raw.columns[0]
        df = df_raw[[date_col, bcs_series_key]].copy()
        df.columns = ["date", "value"]

        # Normalise to month start for pipeline consistency
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"])

        # Filter to start date
        df = df[df["date"] >= f"{start}-01"]

        records = [[row["date"], row["value"]] for _, row in df.iterrows()]

        logger.info(
            "Fetched provider=ecbcs series_id=%s rows=%d",
            series_id, len(records),
        )

        return {
            "zip_url": zip_url,
            "series_key": bcs_series_key,
            "data": records,
        }

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=ecbcs series_id=%s strict=%s", series_id, strict)

        if not raw_payload or "data" not in raw_payload:
            raise ValueError(
                f"EC BCS payload empty or malformed for series_id={series_id}"
            )

        records = raw_payload["data"]
        if not records:
            raise ValueError(
                f"EC BCS payload contains no data rows for series_id={series_id}"
            )

        df = pd.DataFrame(records, columns=["time", "value"])

        # Dates already normalised to YYYY-MM-DD in fetch()
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
                "Dropped %d/%d rows (%.1f%%) for series_id=%s. Sample bad rows:\n%s",
                dropped, before, drop_pct, series_id, bad.to_string(),
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