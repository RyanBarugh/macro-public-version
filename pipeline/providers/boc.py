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

# Bank of Canada Valet API — free, no auth required
# https://www.bankofcanada.ca/valet/
# URL pattern:
#   GET {BASE_URL}/observations/{boc_series}/json?start_date=YYYY-MM-DD
#
# Required meta field in series.json:
#   boc_series : str — BoC Valet series code e.g. "BD.CDN.2YR.DQ.YLD"
#
# Response structure:
#   observations: [ { "d": "YYYY-MM-DD", "{boc_series}": { "v": "1.66" } }, ... ]

BASE_URL = "https://www.bankofcanada.ca/valet"


class BocProvider(BaseProvider):
    """
    Bank of Canada Valet API provider.

    Free access, no API key required. Returns JSON.
    Weekdays only — BoC does not publish on weekends or holidays.

    Each series definition must supply in meta:
        boc_series : str — Valet series code (e.g. "BD.CDN.2YR.DQ.YLD")
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — BoC is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        boc_series = meta.get("boc_series") or (meta.get("meta") or {}).get("boc_series")

        if not boc_series:
            raise RuntimeError(
                f"BoC series_id={series_def.series_id} missing 'boc_series' in meta"
            )

        # Convert start "YYYY-MM" to "YYYY-MM-DD"
        try:
            start_date = pd.to_datetime(start + "-01").strftime("%Y-%m-%d")
        except Exception:
            start_date = start

        url = f"{BASE_URL}/observations/{boc_series}/json?start_date={start_date}"

        logger.info(
            "Fetching provider=boc series_id=%s boc_series=%s",
            series_def.series_id, boc_series,
        )
        logger.debug("BoC URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=boc series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            payload = response.json()

            observations = payload.get("observations", [])
            if not observations:
                raise RuntimeError(
                    f"BoC returned empty observations for series_id={series_def.series_id}"
                )

            logger.info(
                "Fetched provider=boc series_id=%s rows=%d",
                series_def.series_id, len(observations),
            )

            return {"boc_series": boc_series, "observations": observations}

        except Exception:
            logger.exception("Failed provider=boc series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=boc series_id=%s strict=%s", series_id, strict)

        boc_series = raw_payload.get("boc_series")
        observations = raw_payload.get("observations", [])

        if not observations:
            raise ValueError(f"BoC payload empty for series_id={series_id}")

        records = []
        raw_null_count = 0

        for obs in observations:
            date_str = obs.get("d")
            val_block = obs.get(boc_series, {})
            val_str = val_block.get("v") if isinstance(val_block, dict) else None

            if val_str is None or val_str == "":
                raw_null_count += 1
                continue

            try:
                records.append({"time": date_str, "value": float(val_str)})
            except (TypeError, ValueError):
                raw_null_count += 1

        if raw_null_count:
            logger.warning(
                "provider=boc series_id=%s raw_null_count=%d",
                series_id, raw_null_count,
            )

        if not records:
            raise ValueError(f"BoC: no valid records for series_id={series_id}")

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

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