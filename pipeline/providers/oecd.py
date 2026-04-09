from __future__ import annotations

import io
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

# OECD SDMX 2.1 REST API — free, no auth required
BASE_URL = "https://sdmx.oecd.org/public/rest/data"


class OecdProvider(BaseProvider):
    """
    OECD Data Explorer SDMX 2.1 REST API provider.

    Free access, no API key required. Returns CSV with labels.
    Rate limit: ~20 requests/hour from a single IP (undocumented soft limit).

    Each series definition must supply in meta:
        dataflow    : str — full dataflow reference
                            e.g. "OECD.SDD.STES,DSD_STES@DF_INDSERV,4.3"
        data_key    : str — SDMX dimension key (dot-separated)
                            e.g. "NZL.Q.TOVM.IX.G47.Y._Z._Z.N"

    The `start` parameter ("YYYY-MM") is converted to the appropriate
    SDMX startPeriod format. For quarterly series this becomes "YYYY-Qn".

    CSV response columns include TIME_PERIOD and OBS_VALUE among others.
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — OECD is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        dataflow = meta.get("dataflow")
        data_key = meta.get("data_key")

        if not dataflow or not data_key:
            raise RuntimeError(
                f"OECD series_id={series_def.series_id} missing 'dataflow' or "
                f"'data_key' in meta"
            )

        # Convert "YYYY-MM" to SDMX startPeriod.
        # For quarterly data: "YYYY-MM" → "YYYY-Qn"
        # For monthly data: keep as "YYYY-MM"
        freq = data_key.split(".")[1] if "." in data_key else "Q"
        start_period = _to_sdmx_period(start, freq)

        params = {
            "startPeriod": start_period,
            "dimensionAtObservation": "AllDimensions",
            "format": "csvfilewithlabels",
        }

        url = f"{BASE_URL}/{dataflow}/{data_key}?{urlencode(params)}"

        logger.info(
            "Fetching provider=oecd series_id=%s dataflow=%s",
            series_def.series_id, dataflow,
        )
        logger.debug("OECD URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=oecd series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            text = response.text
            if not text or len(text.strip()) == 0:
                raise RuntimeError(
                    f"OECD returned empty response for series_id={series_def.series_id}"
                )

            # Quick shape check — first line should be CSV header
            first_line = text.split("\n")[0]
            if "TIME_PERIOD" not in first_line or "OBS_VALUE" not in first_line:
                raise RuntimeError(
                    f"OECD response missing expected CSV columns for "
                    f"series_id={series_def.series_id}. Header: {first_line!r}"
                )

            row_count = max(text.count("\n") - 1, 0)
            logger.info(
                "Fetched provider=oecd series_id=%s rows=%d",
                series_def.series_id, row_count,
            )
            # Wrap CSV text with meta needed by clean() — transformation filter
            # allows clean() to select the correct TRANSFORMATION value without
            # changing the BaseProvider.clean() signature.
            transformation = (series_def.meta or {}).get("transformation", "_Z")
            filters = (series_def.meta or {}).get("filters", {})
            return {"_csv": text, "_transformation": transformation, "_filters": filters}

        except Exception:
            logger.exception("Failed provider=oecd series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=oecd series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"OECD payload empty for series_id={series_id}")

        # Unwrap payload — fetch() returns a dict with "_csv" and "_transformation"
        if isinstance(raw_payload, dict):
            csv_text = raw_payload.get("_csv", "")
        else:
            # Backwards compatibility if raw payload is plain text
            csv_text = raw_payload

        df = pd.read_csv(io.StringIO(csv_text))

        # Normalise column names — OECD CSV may have labels appended
        # e.g. "TIME_PERIOD" and "Time period" as separate columns
        col_map = {c.upper().split(":")[0].strip(): c for c in df.columns}
        time_col = col_map.get("TIME_PERIOD")
        val_col = col_map.get("OBS_VALUE")

        if not time_col or not val_col:
            raise ValueError(
                f"OECD CSV missing TIME_PERIOD or OBS_VALUE for series_id={series_id}. "
                f"Columns: {list(df.columns)}"
            )

        logger.info(
            "Raw rows (excluding header) series_id=%s rows_in=%d", series_id, len(df)
        )

        # Filter to a single TRANSFORMATION value if the column is present.
        # OECD APIs often return multiple transformations (index level, YoY growth,
        # MoM growth, etc.) for a broad data_key. The fetch() method stashes the
        # desired transformation code in the payload wrapper (default "_Z" = raw
        # index level, no transformation). Override per-series via meta field
        # "transformation".
        trans_col = col_map.get("TRANSFORMATION")
        if trans_col is not None:
            available = df[trans_col].unique().tolist()
            target = raw_payload if isinstance(raw_payload, dict) else {}
            transformation = target.get("_transformation", "_Z")
            if transformation in available:
                before_filter = len(df)
                df = df[df[trans_col] == transformation].copy()
                logger.info(
                    "Filtered TRANSFORMATION=%s for series_id=%s: %d→%d rows "
                    "(available: %s)",
                    transformation, series_id, before_filter, len(df), available,
                )
            else:
                logger.warning(
                    "TRANSFORMATION=%s not found for series_id=%s; "
                    "available: %s. No filter applied.",
                    transformation, series_id, available,
                )

        # Apply any additional column filters stashed by fetch() in the payload.
        # e.g. {"ACTIVITY": "BTE"} to disambiguate when a broad data_key returns
        # multiple activities. Filters are matched case-insensitively against
        # the normalised col_map keys.
        extra_filters = (raw_payload if isinstance(raw_payload, dict) else {}).get("_filters", {})
        for dim, val in extra_filters.items():
            col = col_map.get(dim.upper())
            if col is not None and col in df.columns:
                available = df[col].unique().tolist()
                if val in available:
                    before_filter = len(df)
                    df = df[df[col] == val].copy()
                    logger.info(
                        "Filtered %s=%s for series_id=%s: %d→%d rows",
                        dim, val, series_id, before_filter, len(df),
                    )
                else:
                    logger.warning(
                        "Filter %s=%s not found for series_id=%s; available: %s",
                        dim, val, series_id, available,
                    )

        # Convert quarterly periods "YYYY-Qn" → datetime
        # pd.to_datetime handles "2024-Q1" natively with period parsing
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_sdmx_period(yyyymm: str, freq: str = "Q") -> str:
    """
    Convert pipeline "YYYY-MM" start to SDMX startPeriod format.

    Quarterly: "2020-01" → "2020-Q1"
    Monthly:   "2020-01" → "2020-01"
    Annual:    "2020-01" → "2020"
    """
    parts = yyyymm.split("-")
    if len(parts) != 2:
        return yyyymm

    year, month = parts[0], int(parts[1])

    if freq.upper() == "Q":
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    elif freq.upper() == "A":
        return year
    else:
        return yyyymm


def _period_to_date(period_str: str) -> str:
    """
    Convert SDMX period strings to "YYYY-MM-DD" for pandas.

    "2024-Q1" → "2024-03-01"  (last month of quarter)
    "2024-Q2" → "2024-06-01"
    "2024-Q3" → "2024-09-01"
    "2024-Q4" → "2024-12-01"
    "2024-01" → "2024-01-01"  (monthly, pass through)
    "2024"    → "2024-01-01"  (annual)
    """
    s = str(period_str).strip()

    # Quarterly: "YYYY-Qn"
    if "-Q" in s:
        try:
            year, q_part = s.split("-Q")
            q = int(q_part)
            last_month = q * 3
            return f"{year}-{last_month:02d}-01"
        except (ValueError, IndexError):
            return s

    # Monthly: "YYYY-MM" → "YYYY-MM-01"
    if len(s) == 7 and s[4] == "-":
        return f"{s}-01"

    # Annual: "YYYY" → "YYYY-01-01"
    if len(s) == 4 and s.isdigit():
        return f"{s}-01-01"

    return s