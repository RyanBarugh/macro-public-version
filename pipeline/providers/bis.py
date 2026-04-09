"""
BIS Provider — Central Bank Policy Rates
==========================================

Fetches daily central bank policy rates from the BIS SDMX REST API.
No authentication required. Free access.

API endpoint:
    GET https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.{REF_AREA}
        ?format=csv&startPeriod={YYYY-MM-DD}

The BIS policy rates dataset tracks the evolution of policy rates across
40+ economies, with daily and monthly frequency. The BIS collaborates
directly with member central banks to select the appropriate policy rate.

For target-band central banks (e.g. the Fed), the midpoint is shown.
For periods when monetary policy was not conducted with an interest rate
instrument, the most widely referenced money market rate is used.

Response: CSV with columns including:
    FREQ, REF_AREA, TIME_PERIOD, OBS_VALUE, OBS_STATUS, ...

We use TIME_PERIOD (date) and OBS_VALUE (rate in % per annum).
Weekends and holidays have NaN values — these are dropped.

Required meta fields in series JSON:
    bis_ref_area : str — BIS country code, e.g. "US", "GB", "XM", "AU"

Reference:
    https://data.bis.org/topics/CBPOL
    https://stats.bis.org/api-doc/v2/
"""

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

BASE_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0"


class BisProvider(BaseProvider):
    """
    Bank for International Settlements — central bank policy rates.

    Free access, no API key. Daily data going back to 1946 for some
    economies. Updated mid-week.

    BIS country codes for G10+:
        US  — Federal Reserve (FFR target midpoint)
        XM  — ECB (main refinancing rate / deposit facility rate)
        GB  — Bank of England (Bank Rate)
        JP  — Bank of Japan (call rate target)
        AU  — Reserve Bank of Australia (cash rate target)
        CA  — Bank of Canada (overnight rate target)
        CH  — Swiss National Bank (policy rate / SARON target)
        NO  — Norges Bank (sight deposit rate)
        SE  — Sveriges Riksbank (repo rate)
        NZ  — Reserve Bank of New Zealand (OCR)
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        ref_area = meta.get("bis_ref_area")

        if not ref_area:
            raise RuntimeError(
                f"BIS series_id={series_def.series_id} missing "
                f"'bis_ref_area' in meta"
            )

        # Convert "YYYY-MM" → "YYYY-MM-01" for BIS startPeriod
        if len(start) == 7:
            start_period = f"{start}-01"
        else:
            start_period = start

        # D = daily frequency, ref_area = country code
        url = f"{BASE_URL}/D.{ref_area}"

        params = {
            "format": "csv",
            "startPeriod": start_period,
        }

        logger.info(
            "Fetching provider=bis series_id=%s ref_area=%s from=%s",
            series_def.series_id, ref_area, start_period,
        )

        start_ts = time.time()
        try:
            response = session.get(url, params=params, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=bis series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            # Return raw CSV text
            text = response.text
            row_count = text.count("\n") - 1  # minus header
            logger.info(
                "Fetched provider=bis series_id=%s rows=~%d",
                series_def.series_id, max(row_count, 0),
            )
            return text

        except Exception:
            logger.exception(
                "Failed provider=bis series_id=%s", series_def.series_id
            )
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=bis series_id=%s", series_id)

        if not raw_payload:
            raise ValueError(f"BIS payload empty for series_id={series_id}")

        df = pd.read_csv(io.StringIO(raw_payload))

        # ── Validate expected columns ─────────────────────────────────
        required_cols = {"TIME_PERIOD", "OBS_VALUE"}
        missing = required_cols - set(df.columns)
        if missing:
            raise RuntimeError(
                f"BIS series_id={series_id}: missing CSV columns {missing}. "
                f"Got: {list(df.columns)}"
            )

        total_rows = len(df)
        if total_rows == 0:
            raise ValueError(
                f"BIS series_id={series_id}: CSV parsed but zero rows"
            )

        # ── Parse dates and values ────────────────────────────────────
        df["time"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
        df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

        # Drop rows where the date itself is unparseable — genuinely bad
        bad_dates = df["time"].isna().sum()
        if bad_dates:
            logger.warning(
                "BIS %s: %d rows with unparseable dates — dropping",
                series_id, bad_dates,
            )
            df = df.dropna(subset=["time"])

        df = df.sort_values("time")

        # ── Forward-fill: policy rates are step functions ─────────────
        # BIS includes rows for every calendar day. Weekends and holidays
        # have NaN values because markets are closed, but the policy rate
        # is still in effect. Forward-fill carries the last known rate
        # into non-observation days — this is semantically correct.
        obs_count = df["value"].notna().sum()
        nan_count = df["value"].isna().sum()

        if obs_count == 0:
            raise ValueError(
                f"BIS series_id={series_id}: all {total_rows} rows have "
                f"NaN values. No usable data."
            )

        df["value"] = df["value"].ffill()

        # After ffill, only leading NaNs remain (dates before first
        # ever published rate). Drop those.
        leading_nans = df["value"].isna().sum()
        if leading_nans:
            df = df.dropna(subset=["value"])

        logger.info(
            "BIS %s: %d total rows, %d observations, %d forward-filled, "
            "%d leading NaNs dropped",
            series_id, total_rows, obs_count, nan_count - leading_nans,
            leading_nans,
        )

        # ── Integrity check on observation rows ───────────────────────
        # At this point all rows should be clean. If anything is still
        # NaN after ffill, something is genuinely broken.
        still_nan = df["value"].isna().sum()
        if still_nan:
            raise ValueError(
                f"BIS series_id={series_id}: {still_nan} NaN values remain "
                f"after forward-fill. Data integrity issue."
            )

        # ── Deduplicate — keep last value for any date ────────────────
        dupes = df["time"].duplicated(keep=False).sum()
        if dupes:
            logger.info(
                "BIS %s: %d duplicate dates — keeping last", series_id, dupes
            )
        df = df.drop_duplicates(subset=["time"], keep="last")

        # ── Build output ──────────────────────────────────────────────
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