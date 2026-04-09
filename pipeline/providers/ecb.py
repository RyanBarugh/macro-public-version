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

# ECB Data Portal SDMX 2.1 REST API — free, no auth required
# Migrated from sdw-wsrest.ecb.europa.eu to data-api.ecb.europa.eu (2024)
BASE_URL = "https://data-api.ecb.europa.eu/service/data"


class EcbProvider(BaseProvider):
    """
    European Central Bank Data Portal SDMX 2.1 REST API provider.

    Free access, no API key required. Returns CSV with labels.
    No documented rate limit (institutional API).

    As of 4 February 2026, the ECB replaced the old ICP dataset with a new
    HICP dataset (DSD: ECB_ICP3, base period 2025=100). The dimension
    structure remains the same:

        FREQ.REF_AREA.ADJUSTMENT.ICP_ITEM.STS_INSTITUTION.ICP_SUFFIX

    For the SA HICP overall index (euro area):
        Old: ICP / M.U2.Y.000000.3.INX
        New: HICP / M.U2.Y.000000.3.INX

    Dimension values:
        FREQ              M       Monthly
        REF_AREA          U2      Euro area (changing composition)
        ADJUSTMENT        Y       Working day and seasonally adjusted
                          N       Neither seasonally nor working day adjusted
        ICP_ITEM          000000  HICP - Overall index
                          XEF000  HICP - All-items excl. energy and food
                          FOOD00  HICP - Food incl. alcohol and tobacco
                          SERV00  HICP - Services
                          NRG000  HICP - Energy
        STS_INSTITUTION   3       European Central Bank (SA series)
                          4       Eurostat (NSA series)
        ICP_SUFFIX        INX     Index level
                          ANR     Annual rate of change
                          MOR     Monthly rate of change

    Each series definition must supply in meta:
        ecb_dataset   : str  — "HICP" (new) or "ICP" (legacy, redirects)
        data_key      : str  — SDMX dimension key (dot-separated)
                               e.g. "M.U2.Y.000000.3.INX"

    URL pattern:
        GET {BASE_URL}/{ecb_dataset}/{data_key}?startPeriod=YYYY-MM&format=csvdata
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — ECB is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        ecb_dataset = meta.get("ecb_dataset", "HICP")
        data_key = meta.get("data_key")

        if not data_key:
            raise RuntimeError(
                f"ECB series_id={series_def.series_id} missing 'data_key' in meta"
            )

        params = {
            "startPeriod": start,
            "format": "csvdata",
        }

        url = f"{BASE_URL}/{ecb_dataset}/{data_key}?{urlencode(params)}"

        logger.info(
            "Fetching provider=ecb series_id=%s dataset=%s",
            series_def.series_id, ecb_dataset,
        )
        logger.debug("ECB URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=ecb series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            text = response.text
            if not text or len(text.strip()) == 0:
                raise RuntimeError(
                    f"ECB returned empty response for series_id={series_def.series_id}"
                )

            # Quick shape check — first line should be CSV header
            first_line = text.split("\n")[0]
            if "TIME_PERIOD" not in first_line or "OBS_VALUE" not in first_line:
                raise RuntimeError(
                    f"ECB response missing expected CSV columns for "
                    f"series_id={series_def.series_id}. Header: {first_line!r}"
                )

            row_count = max(text.count("\n") - 1, 0)
            logger.info(
                "Fetched provider=ecb series_id=%s rows=%d",
                series_def.series_id, row_count,
            )
            return text

        except Exception:
            logger.exception("Failed provider=ecb series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=ecb series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"ECB payload empty for series_id={series_id}")

        df = pd.read_csv(io.StringIO(raw_payload))

        # Normalise column names — ECB CSV may include label suffixes
        col_map = {c.upper().split(":")[0].strip(): c for c in df.columns}
        time_col = col_map.get("TIME_PERIOD")
        val_col = col_map.get("OBS_VALUE")

        if not time_col or not val_col:
            raise ValueError(
                f"ECB CSV missing TIME_PERIOD or OBS_VALUE for series_id={series_id}. "
                f"Columns: {list(df.columns)}"
            )

        logger.info(
            "Raw rows (excluding header) series_id=%s rows_in=%d", series_id, len(df)
        )

        # ECB monthly data is "YYYY-MM" format — straightforward parse
        df["time"] = pd.to_datetime(df[time_col], errors="coerce")
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
