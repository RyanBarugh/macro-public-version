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

# StatCan WDS base URL
BASE_URL = "https://www150.statcan.gc.ca/t1/wds/rest"

# Scalar factor multipliers from WDS codeset
# code 3 = thousands, code 6 = millions, etc.
_SCALAR_MULTIPLIERS = {
    0: 1,
    1: 10,
    2: 100,
    3: 1_000,
    4: 10_000,
    5: 100_000,
    6: 1_000_000,
    7: 10_000_000,
    8: 100_000_000,
    9: 1_000_000_000,
}


class StatCanProvider(BaseProvider):
    """
    Statistics Canada Web Data Service (WDS) provider.

    Uses getDataFromVectorByReferencePeriodRange (GET, by reference period).
    Each series definition must supply:
        meta["vector_id"]   : int  — the WDS vector number (e.g. 41690973)

    The WDS scalar factor is applied automatically so returned values are
    in the native unit (usually thousands of CAD for retail trade).

    Rate limits: 25 requests/s per IP, 50/s server-wide.
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],   # unused — WDS is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        vector_id = meta.get("vector_id")

        if not vector_id:
            raise RuntimeError(
                f"StatCan series_id={series_def.series_id} missing 'vector_id' in meta"
            )

        # start is "YYYY-MM"; API wants "YYYY-MM-DD"
        start_date = f"{start}-01"
        # end date: today
        import datetime
        end_date = datetime.date.today().strftime("%Y-%m-%d")

        url = (
            f"{BASE_URL}/getDataFromVectorByReferencePeriodRange"
            f'?vectorIds={vector_id}'
            f"&startRefPeriod={start_date}"
            f"&endReferencePeriod={end_date}"
        )

        logger.info(
            "Fetching provider=statcan series_id=%s vector_id=%s",
            series_def.series_id, vector_id,
        )
        logger.debug("StatCan URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=statcan series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()
            data = response.json()

            # WDS returns a list with one object per requested vector
            if not isinstance(data, list) or len(data) == 0:
                raise RuntimeError(
                    f"StatCan unexpected response shape for series_id={series_def.series_id}"
                )

            obj = data[0]
            if obj.get("status") != "SUCCESS":
                raise RuntimeError(
                    f"StatCan returned status={obj.get('status')} "
                    f"for series_id={series_def.series_id}: {obj}"
                )

            points = obj.get("object", {}).get("vectorDataPoint", [])
            logger.info(
                "Fetched provider=statcan series_id=%s datapoints=%d",
                series_def.series_id, len(points),
            )
            return obj["object"]  # pass the full object so clean() can read scalarFactorCode

        except Exception:
            logger.exception("Failed provider=statcan series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=statcan series_id=%s strict=%s", series_id, strict)

        if not raw_payload or "vectorDataPoint" not in raw_payload:
            raise ValueError(
                f"StatCan payload empty or malformed for series_id={series_id}"
            )

        points = raw_payload["vectorDataPoint"]
        # scalar factor is consistent across the vector; take it from the first point
        scalar_code = points[0].get("scalarFactorCode", 0) if points else 0
        multiplier = _SCALAR_MULTIPLIERS.get(scalar_code, 1)

        if multiplier != 1:
            logger.info(
                "Applying scalar multiplier=%d (code=%d) for series_id=%s",
                multiplier, scalar_code, series_id,
            )

        records = []
        for pt in points:
            # refPerRaw is the actual reference period in YYYY-MM-DD format
            ref = pt.get("refPerRaw") or pt.get("refPer")
            val = pt.get("value")
            # Skip suppressed / not-available points (statusCode != 0 with no value)
            if val is None or val == "":
                continue
            records.append({"time": ref, "value": val})

        logger.info(
            "Raw datapoints series_id=%s rows_in=%d", series_id, len(records)
        )

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce") * multiplier

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
