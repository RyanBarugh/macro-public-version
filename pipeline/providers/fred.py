from __future__ import annotations

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

# FRED REST API — free, requires API key
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredProvider(BaseProvider):
    """
    Federal Reserve Economic Data (FRED) provider.
    Hosted by the Federal Reserve Bank of St. Louis.

    Free access; requires a FRED API key (register at
    https://fredaccount.stlouisfed.org/apikeys).
    Set via env var API_SECRET__FRED.

    Each series definition must supply in meta:
        series_id : str — FRED series identifier, e.g. "INDPRO" or "IPMAN"

    Response JSON shape:
        {
          "observations": [
            {"date": "YYYY-MM-01", "value": "102.34"},
            ...
          ]
        }

    Dates are always the first of the month. Values are strings; "." denotes
    a missing observation (FRED convention) and is dropped.

    Known series (G.17 Industrial Production & Capacity Utilization):
        INDPRO  — Total IP index, SA, 2017=100, monthly from 1919-01-01
        IPMAN   — Manufacturing (NAICS), SA, 2017=100, monthly from 1972-01-01
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        if not api_key:
            raise RuntimeError(
                "FredProvider requires an API key. "
                "Set env var API_SECRET__FRED."
            )

        meta = series_def.meta or {}
        fred_series_id = meta.get("fred_series_id")
        if not fred_series_id:
            raise RuntimeError(
                f"FRED series_id={series_def.series_id} missing "
                f"'fred_series_id' in meta"
            )

        # Convert "YYYY-MM" → "YYYY-MM-01" for FRED observation_start
        observation_start = f"{start}-01" if len(start) == 7 else start

        params = {
            "series_id": fred_series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "sort_order": "asc",
        }

        url = f"{BASE_URL}?{urlencode(params)}"

        logger.info(
            "Fetching provider=fred series_id=%s fred_series_id=%s",
            series_def.series_id, fred_series_id,
        )

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=fred series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            payload = response.json()

            if "observations" not in payload:
                raise RuntimeError(
                    f"FRED response missing 'observations' key for "
                    f"series_id={series_def.series_id}. "
                    f"Keys: {list(payload.keys())}"
                )

            row_count = len(payload["observations"])
            logger.info(
                "Fetched provider=fred series_id=%s rows=%d",
                series_def.series_id, row_count,
            )
            return payload

        except Exception:
            logger.exception("Failed provider=fred series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=fred series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"FRED payload empty for series_id={series_id}")

        observations = raw_payload.get("observations", [])
        if not observations:
            raise ValueError(f"FRED observations list empty for series_id={series_id}")

        df = pd.DataFrame(observations)

        # FRED uses "." to denote missing values — drop before numeric conversion
        df = df[df["value"] != "."].copy()

        df["time"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

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
