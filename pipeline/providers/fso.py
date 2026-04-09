from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0

# Swiss Federal Statistical Office — new SDMX REST API (replaced PXWeb)
# Confirmed working as of March 2026.
BASE_URL = "https://disseminate.stats.swiss/rest/data"

# SDMX XML namespaces used by the FSO endpoint
NS = {
    "message": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "generic": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
    "common": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}


class FsoProvider(BaseProvider):
    """
    Swiss Federal Statistical Office (FSO / BFS) SDMX REST API provider.

    The FSO migrated from PXWeb to an SDMX 2.1 REST API hosted at
    disseminate.stats.swiss (Swiss Stats Explorer backend).

    Data query URL pattern:
        GET {BASE_URL}/{agency},{dataflow},{version}/{key}?startPeriod={YYYY-MM}

    The response is SDMX Generic Data XML.

    For the retail trade turnover monthly series (DF_KEU_M1):
        Agency:   CH1.KEU
        Dataflow: DF_KEU_M1
        Version:  1.0.0

        Dimension order: NOGA.ADJUSTMENT.INDICATOR_KE.UNIT_MEASURE.FREQ
            NOGA          = NACE industry code (47 = retail trade)
            ADJUSTMENT    = W (seasonally adjusted), Y (calendar adjusted), N (unadjusted)
            INDICATOR_KE  = UTOT (total turnover)
            UNIT_MEASURE  = VARM-1 (MoM change %), VARM-12 (YoY change %)
            FREQ          = M (monthly)

    Each series definition must supply in meta:
        agency    : str — e.g. "CH1.KEU"
        dataflow  : str — e.g. "DF_KEU_M1"
        version   : str — e.g. "1.0.0" (default)
        data_key  : str — SDMX dimension key, e.g. "47.W.UTOT.VARM-12.M"

    No API key required. No authentication.
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — FSO is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        agency = meta.get("agency", "CH1.KEU")
        dataflow = meta.get("dataflow", "DF_KEU_M1")
        version = meta.get("version", "1.0.0")
        data_key = meta.get("data_key")

        if not data_key:
            raise RuntimeError(
                f"FSO series_id={series_def.series_id} missing 'data_key' in meta"
            )

        # Convert YYYY-MM to YYYY-Qn for quarterly dataflows
        if data_key.endswith(".Q"):
            year, month = start.split("-")
            quarter = (int(month) - 1) // 3 + 1
            start_period = f"{year}-Q{quarter}"
        else:
            start_period = start

        url = (
            f"{BASE_URL}/{agency},{dataflow},{version}/{data_key}"
            f"?startPeriod={start_period}"
        )

        logger.info(
            "Fetching provider=fso series_id=%s dataflow=%s",
            series_def.series_id, dataflow,
        )
        logger.debug("FSO URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=fso series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            text = response.text
            if not text or len(text.strip()) == 0:
                raise RuntimeError(
                    f"FSO returned empty response for series_id={series_def.series_id}"
                )

            # Quick sanity check — should be XML with observations
            if "<generic:Obs>" not in text and "<Obs>" not in text:
                raise RuntimeError(
                    f"FSO response contains no observations for "
                    f"series_id={series_def.series_id}"
                )

            # Count observations for logging
            obs_count = text.count("<generic:Obs>") or text.count("<Obs>")
            logger.info(
                "Fetched provider=fso series_id=%s observations=%d",
                series_def.series_id, obs_count,
            )
            return text

        except Exception:
            logger.exception("Failed provider=fso series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Parse SDMX Generic Data XML response.

        Structure:
            <generic:Series>
                <generic:SeriesKey>...</generic:SeriesKey>
                <generic:Obs>
                    <generic:ObsDimension id="TIME_PERIOD" value="2020-01"/>
                    <generic:ObsValue value="0.34"/>
                </generic:Obs>
                ...
            </generic:Series>
        """
        logger.info("Cleaning provider=fso series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"FSO payload empty for series_id={series_id}")

        root = ET.fromstring(raw_payload)

        records = []

        # Find all Obs elements — try namespaced first, then plain
        obs_elements = root.findall(".//generic:Obs", NS)
        if not obs_elements:
            # Fallback: try without namespace
            obs_elements = root.findall(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Obs")

        if not obs_elements:
            raise ValueError(
                f"FSO: no observations found in XML for series_id={series_id}"
            )

        for obs in obs_elements:
            # Extract TIME_PERIOD
            obs_dim = obs.find("generic:ObsDimension", NS)
            if obs_dim is None:
                obs_dim = obs.find(
                    "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}ObsDimension"
                )
            time_val = obs_dim.get("value") if obs_dim is not None else None

            # Extract observation value
            obs_value = obs.find("generic:ObsValue", NS)
            if obs_value is None:
                obs_value = obs.find(
                    "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}ObsValue"
                )
            val = obs_value.get("value") if obs_value is not None else None

            if time_val is not None and val is not None:
                records.append({"time": time_val, "value": val})

        logger.info(
            "Raw observations series_id=%s rows_in=%d", series_id, len(records)
        )

        df = pd.DataFrame(records) if records else pd.DataFrame(columns=["time", "value"])
        # Handle quarterly periods (e.g. "2004-Q1") → first day of quarter
        def parse_time(t):
            if isinstance(t, str) and "-Q" in t:
                year, q = t.split("-Q")
                month = (int(q) - 1) * 3 + 1
                return pd.Timestamp(f"{year}-{month:02d}-01")
            return pd.to_datetime(t, errors="coerce")

        df["time"] = df["time"].apply(parse_time)
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
                "Dropped %d/%d rows (%.1f%%) for series_id=%s. Sample:\n%s",
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