from __future__ import annotations

import base64
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

# Ministry of Finance Japan — JGB benchmark yield curve
# Daily closing yields, published each business day
# https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm
#
# Always fetches BOTH endpoints and merges, so no gaps at month boundaries:
#   Historical : jgbcme_all.csv  — 1974 to end of previous month
#   Current    : jgbcme.csv      — current month to date
#
# CSV structure (identical for both files):
#   Row 0  : Title e.g. "Interest Rate (March 2026)"
#   Row 1  : Headers — Date, 1Y, 2Y, 3Y, 4Y, 5Y, 6Y, 7Y, 8Y, 9Y, 10Y, 15Y, 20Y, 25Y, 30Y, 40Y
#   Row 2+ : Daily data, date format DD/MM/YYYY
#   Footer : Note row starting with "¦" — dropped automatically via numeric coerce
#   Units  : Percent
#
# Required meta fields in series.json:
#   mof_column : str — maturity column e.g. "2Y", "5Y", "10Y"
#
# Cache: both CSVs are cached at module level — only 2 HTTP requests per
# process regardless of how many MOF series are fetched.

MOF_HISTORICAL_URL = (
    "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
    "historical/jgbcme_all.csv"
)
MOF_CURRENT_URL = (
    "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
    "jgbcme.csv"
)

VALID_COLUMNS = {"1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
                 "10Y", "15Y", "20Y", "25Y", "30Y", "40Y"}

# Module-level cache keyed by URL — populated on first fetch, reused thereafter
_CSV_CACHE: dict[str, bytes] = {}


def _fetch_url(
    url: str,
    session: requests.Session,
    timeout: Tuple[float, float],
    series_id: str,
) -> bytes:
    """Fetch a URL with caching. Returns raw bytes."""
    global _CSV_CACHE
    if url in _CSV_CACHE:
        logger.info(
            "provider=mof using cached csv url=%s bytes=%d",
            url, len(_CSV_CACHE[url]),
        )
        return _CSV_CACHE[url]

    start_ts = time.time()
    response = session.get(url, timeout=timeout)
    elapsed_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "HTTP provider=mof series_id=%s url=%s status=%s elapsed_ms=%d",
        series_id, url, response.status_code, elapsed_ms,
    )
    response.raise_for_status()
    _CSV_CACHE[url] = response.content
    logger.info("Downloaded provider=mof url=%s bytes=%d", url, len(_CSV_CACHE[url]))
    return _CSV_CACHE[url]


class MofProvider(BaseProvider):
    """
    Ministry of Finance Japan JGB benchmark yield curve provider.

    Always fetches both the historical CSV (1974–end of last month) and the
    current-month CSV, merging them so there are no gaps at month boundaries.

    Required meta fields in series.json:
        mof_column : str — maturity column e.g. "2Y", "5Y", "10Y"
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — MOF is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        mof_column = meta.get("mof_column")

        if not mof_column:
            raise RuntimeError(
                f"MOF series_id={series_def.series_id} missing 'mof_column' in meta"
            )

        if mof_column not in VALID_COLUMNS:
            raise RuntimeError(
                f"MOF series_id={series_def.series_id} unknown mof_column={mof_column!r}. "
                f"Valid: {sorted(VALID_COLUMNS)}"
            )

        logger.info(
            "Fetching provider=mof series_id=%s column=%s",
            series_def.series_id, mof_column,
        )

        historical_bytes = _fetch_url(
            MOF_HISTORICAL_URL, session, timeout, series_def.series_id
        )
        current_bytes = _fetch_url(
            MOF_CURRENT_URL, session, timeout, series_def.series_id
        )

        # Base64-encode raw bytes so the payload is JSON-serialisable for S3
        return {
            "mof_column":     mof_column,
            "start":          start,
            "historical_b64": base64.b64encode(historical_bytes).decode("ascii"),
            "current_b64":    base64.b64encode(current_bytes).decode("ascii"),
        }

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info(
            "Cleaning provider=mof series_id=%s strict=%s", series_id, strict
        )

        mof_column       = raw_payload["mof_column"]
        start            = raw_payload.get("start", "1974-01")
        historical_bytes = base64.b64decode(raw_payload["historical_b64"])
        current_bytes    = base64.b64decode(raw_payload["current_b64"])

        def _parse_csv(csv_bytes: bytes, label: str) -> pd.DataFrame:
            if not csv_bytes:
                logger.warning("MOF: empty %s csv for series_id=%s", label, series_id)
                return pd.DataFrame(columns=["time", "value"])

            # Historical CSV is UTF-8; current month CSV is Shift-JIS
            for enc in ("utf-8-sig", "shift_jis", "cp932"):
                try:
                    df = pd.read_csv(
                        io.BytesIO(csv_bytes),
                        skiprows=1,   # skip title row; row 1 becomes header
                        encoding=enc,
                    )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(
                    f"MOF ({label}): could not decode CSV for series_id={series_id}"
                )

            if "Date" not in df.columns or mof_column not in df.columns:
                raise ValueError(
                    f"MOF ({label}): expected columns 'Date' and '{mof_column}', "
                    f"got: {list(df.columns)}"
                )

            df = df[["Date", mof_column]].copy()
            df.columns = ["time", "value"]
            # Both CSVs use YYYY/M/D with no zero-padding (e.g. 2026/3/2)
            df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna(subset=["time", "value"])

        df_hist    = _parse_csv(historical_bytes, "historical")
        df_current = _parse_csv(current_bytes, "current")

        before = len(df_hist) + len(df_current)

        # Merge — current rows take precedence on any overlapping dates
        df = (
            pd.concat([df_hist, df_current], ignore_index=True)
            .sort_values("time")
            .drop_duplicates(subset=["time"], keep="last")
        )

        dropped = before - len(df)
        if dropped:
            drop_pct = (dropped / max(before, 1)) * 100
            logger.info(
                "Deduped %d overlapping rows (%.1f%%) for series_id=%s",
                dropped, drop_pct, series_id,
            )
            # Overlap between historical and current is expected — only warn if
            # the drop rate is anomalously high (indicates a format problem)
            if strict and drop_pct > MAX_DROP_PCT:
                raise ValueError(
                    f"series_id={series_id}: dropped {drop_pct:.1f}% of rows "
                    f"(threshold={MAX_DROP_PCT}%). Possible format change."
                )

        # Apply start filter
        try:
            start_dt = pd.to_datetime(start + "-01")
            df = df[df["time"] >= start_dt]
        except Exception:
            pass

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