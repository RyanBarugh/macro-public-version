from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

BASE_URL = "https://api.beta.ons.gov.uk/v1/data"


class ONSProvider(BaseProvider):
    """
    Provider for the UK Office for National Statistics (ONS) Beta API.

    Endpoint: https://api.beta.ons.gov.uk/v1/data?uri={uri}
    No API key required.

    series.json fields used:
        cdid    — ONS CDID / time series ID (e.g. "J467")
        dataset — ONS dataset code (e.g. "drsi")
        uri     — (optional) full URI override; derived from cdid+dataset if absent

    The ONS Beta API always returns the full history for a series.
    Date filtering by `start` is applied during clean() rather than at fetch time.
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — ONS is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}

        uri = meta.get("uri") or self._build_uri(meta, series_def.series_id)
        url = f"{BASE_URL}?uri={uri}"

        logger.info("Fetching provider=ons series_id=%s", series_def.series_id)
        logger.debug("ONS URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)

            logger.info(
                "HTTP provider=ons series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id,
                response.status_code,
                elapsed_ms,
            )

            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                raise RuntimeError(
                    f"ONS returned unexpected type for series_id={series_def.series_id}"
                )
            if "months" not in data and "quarters" not in data:
                raise RuntimeError(
                    f"ONS response missing 'months' or 'quarters' array for series_id={series_def.series_id}. "
                    f"Keys present: {list(data.keys())}"
                )

            rows = len(data.get("months") or data.get("quarters", []))
            logger.info(
                "Fetched provider=ons series_id=%s rows=%d",
                series_def.series_id,
                rows,
            )
            return data

        except Exception:
            logger.exception("Failed provider=ons series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=ons series_id=%s strict=%s", series_id, strict)

        if not raw_payload or ("months" not in raw_payload and "quarters" not in raw_payload):
            raise ValueError(f"ONS payload empty or malformed for series_id={series_id}")

        # Prefer monthly data if present (ONS API sometimes returns both for monthly series)
        # Only use quarterly data if no monthly data exists
        if "months" in raw_payload and raw_payload["months"]:
            records_raw = raw_payload["months"]
            freq = "M"
        elif "quarters" in raw_payload and raw_payload["quarters"]:
            records_raw = raw_payload["quarters"]
            freq = "Q"
        else:
            raise ValueError(f"ONS payload contains empty data arrays for series_id={series_id}")

        logger.info("Raw rows series_id=%s freq=%s rows_in=%d", series_id, freq, len(records_raw))

        # Each monthly record:   {"date": "2024 JAN", ...}
        # Each quarterly record: {"date": "1955 Q1",  ...}
        records = [
            {"time": m.get("date", ""), "value": m.get("value")}
            for m in records_raw
        ]

        df = pd.DataFrame(records)

        if freq == "Q":
            # ONS quarterly date format: "1955 Q1" -> parse to first month of quarter
            def _parse_ons_quarter(s: str) -> str:
                try:
                    year, q = s.strip().split(" Q")
                    month = {"1": "01", "2": "04", "3": "07", "4": "10"}[q]
                    return f"{year}-{month}-01"
                except (ValueError, KeyError):
                    return ""
            df["time"] = pd.to_datetime(df["time"].apply(_parse_ons_quarter), errors="coerce")
        else:
            # ONS monthly date format: "2024 JAN"
            df["time"] = pd.to_datetime(df["time"], format="%Y %b", errors="coerce")
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

    @staticmethod
    def _build_uri(meta: dict, series_id: str) -> str:
        cdid = meta.get("cdid")
        dataset = meta.get("dataset")

        if not cdid:
            raise RuntimeError(
                f"ONS series_id={series_id} missing 'cdid' in series.json"
            )
        if not dataset:
            raise RuntimeError(
                f"ONS series_id={series_id} missing 'dataset' in series.json"
            )

        topic = meta.get("topic", "businessindustryandtrade/retailindustry")
        return f"/{topic}/timeseries/{cdid.lower()}/{dataset.lower()}"