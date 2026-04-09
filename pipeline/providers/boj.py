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

# Bank of Japan Time-Series Data Search API — free, no auth required
# Launched February 2026
# https://www.stat-search.boj.or.jp/api/v1/getDataCode
#
# Required meta fields in series.json:
#   boj_db     : str — DB name e.g. "PR01" (CGPI), "FM08" (FX rates)
#   boj_code   : str — Series code WITHOUT DB prefix e.g. "PRCG20_2200000000"
#
# Time format in response: YYYYMM (monthly), YYYYQQ (quarterly), YYYY (annual)
# Response columns: SERIES_CODE, NAME_OF_TIME_SERIES, UNIT, FREQUENCY,
#                   CATEGORY, LAST_UPDATE, SURVEY_DATES, VALUES

BASE_URL = "https://www.stat-search.boj.or.jp/api/v1/getDataCode"


class BojProvider(BaseProvider):
    """
    Bank of Japan Time-Series Data Search API provider.

    Free access, no API key required. Returns CSV.
    Rate limit: avoid excessive requests (undocumented soft limit).

    Each series definition must supply in meta:
        boj_db   : str — DB name (e.g. "PR01" for CGPI)
        boj_code : str — Series code without DB prefix
                         (e.g. "PRCG20_2200000000")
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — BoJ is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        boj_db   = meta.get("boj_db")   or (meta.get("meta") or {}).get("boj_db")
        boj_code = meta.get("boj_code") or (meta.get("meta") or {}).get("boj_code")

        if not boj_db or not boj_code:
            raise RuntimeError(
                f"BoJ series_id={series_def.series_id} missing 'boj_db' or "
                f"'boj_code' in meta"
            )

        # Convert "YYYY-MM" start to "YYYYMM"
        try:
            start_date = start.replace("-", "")[:6]  # e.g. "196001"
        except Exception:
            start_date = "196001"

        params = {
            "format": "csv",
            "lang": "en",
            "db": boj_db,
            "code": boj_code,
            "startDate": start_date,
        }

        logger.info(
            "Fetching provider=boj series_id=%s db=%s code=%s",
            series_def.series_id, boj_db, boj_code,
        )

        start_ts = time.time()
        try:
            response = session.get(BASE_URL, params=params, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=boj series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            text = response.text
            if not text or "STATUS,200" not in text:
                raise RuntimeError(
                    f"BoJ API error for series_id={series_def.series_id}: "
                    f"{text[:200]}"
                )

            # Count data rows (exclude header lines before SERIES_CODE row)
            row_count = sum(1 for line in text.splitlines()
                           if line.startswith(boj_code))
            logger.info(
                "Fetched provider=boj series_id=%s rows=%d",
                series_def.series_id, row_count,
            )

            return {"boj_code": boj_code, "csv_text": text}

        except Exception:
            logger.exception("Failed provider=boj series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=boj series_id=%s strict=%s", series_id, strict)

        boj_code = raw_payload.get("boj_code")
        csv_text = raw_payload.get("csv_text", "")

        if not csv_text:
            raise ValueError(f"BoJ payload empty for series_id={series_id}")

        # Find the header row (SERIES_CODE,...) and parse from there
        lines = csv_text.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("SERIES_CODE"):
                header_idx = i
                break

        if header_idx is None:
            raise ValueError(
                f"BoJ: could not find SERIES_CODE header for series_id={series_id}"
            )

        data_text = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(data_text))

        # Filter to our series code
        df = df[df["SERIES_CODE"] == boj_code].copy()

        if df.empty:
            raise ValueError(
                f"BoJ: no data rows for series_id={series_id} code={boj_code}"
            )

        logger.info(
            "Raw datapoints series_id=%s rows_in=%d", series_id, len(df)
        )

        # Parse SURVEY_DATES: YYYYMM format → first day of month
        def parse_boj_date(d):
            d = str(int(d)).zfill(6)  # ensure 6 digits
            try:
                return pd.Timestamp(f"{d[:4]}-{d[4:6]}-01")
            except Exception:
                return pd.NaT

        df["time"] = df["SURVEY_DATES"].apply(parse_boj_date)
        df["value"] = pd.to_numeric(df["VALUES"], errors="coerce")

        before = len(df)
        df = df.dropna(subset=["time", "value"]).sort_values("time")
        df = df.drop_duplicates(subset=["time"], keep="last")
        dropped = before - len(df)

        if dropped:
            drop_pct = (dropped / max(before, 1)) * 100
            logger.warning(
                "Dropped %d/%d rows (%.1f%%) for series_id=%s",
                dropped, before, drop_pct, series_id,
            )
            if strict and drop_pct > MAX_DROP_PCT:
                raise ValueError(
                    f"series_id={series_id}: dropped {drop_pct:.1f}% of rows "
                    f"(threshold={MAX_DROP_PCT}%). Possible format change."
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