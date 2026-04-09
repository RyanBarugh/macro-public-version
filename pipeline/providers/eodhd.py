"""
EODHD Provider
==============

Fetches daily EOD data from EODHD across multiple exchanges.

API endpoint:
    GET https://eodhd.com/api/eod/{TICKER}.{EXCHANGE}?api_token={KEY}&fmt=json&from={DATE}

Supported exchanges (set via eodhd_exchange in series meta):
    GBOND  — government bond yields (default)
    INDX   — indices (GSPC, VIX3M, etc.)
    COMM   — commodity futures (HG, etc.)
    FOREX  — currency pairs and spot metals (AUDJPY, XAUUSD, etc.)

Response: JSON array of OHLCV objects:
    [{"date": "2025-03-01", "open": 3.94, "high": 3.96, "low": 3.91,
      "close": 3.95, "adjusted_close": 3.95, "volume": 0}, ...]

We use `close` as the value.

Required meta fields in series.json:
    eodhd_ticker   : str — e.g. "AU2Y", "GSPC", "AUDJPY"
    eodhd_exchange : str — optional, defaults to "GBOND"

API key: stored in AWS Secrets Manager or .env as EODHD_API_KEY
"""

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

BASE_URL = "https://eodhd.com/api/eod"


class EodhdProvider(BaseProvider):
    """
    EODHD multi-exchange provider.

    Downloads daily EOD data from any EODHD exchange (GBOND, INDX, COMM, FOREX).
    Exchange is determined by the eodhd_exchange meta field, defaulting to GBOND
    for backwards compatibility with existing bond series.

    Required meta fields in series.json:
        eodhd_ticker   : str — symbol without exchange suffix, e.g. "AU2Y", "GSPC"
        eodhd_exchange : str — optional, defaults to "GBOND"
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
                f"EODHD series_id={series_def.series_id}: missing API key. "
                "Set EODHD_API_KEY in secrets."
            )

        meta = series_def.meta or {}
        eodhd_ticker = meta.get("eodhd_ticker")

        if not eodhd_ticker:
            raise RuntimeError(
                f"EODHD series_id={series_def.series_id} missing 'eodhd_ticker' in meta"
            )

        # Convert start "YYYY-MM" to "YYYY-MM-DD"
        try:
            start_date = pd.to_datetime(start + "-01").strftime("%Y-%m-%d")
        except Exception:
            start_date = start

        exchange = meta.get("eodhd_exchange", "GBOND")
        symbol = f"{eodhd_ticker}.{exchange}"
        url = f"{BASE_URL}/{symbol}"

        params = {
            "api_token": api_key,
            "fmt": "json",
            "from": start_date,
        }

        logger.info(
            "Fetching provider=eodhd series_id=%s ticker=%s from=%s",
            series_def.series_id, symbol, start_date,
        )

        start_ts = time.time()
        try:
            response = session.get(url, params=params, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=eodhd series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            data = response.json()

            # EODHD returns a list of dicts on success,
            # or a dict with an error message on failure
            if isinstance(data, dict):
                msg = data.get("message", data.get("error", str(data)))
                raise RuntimeError(
                    f"EODHD API error for {symbol}: {msg}"
                )

            logger.info(
                "Fetched provider=eodhd series_id=%s rows=%d",
                series_def.series_id, len(data),
            )
            return data

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 402:
                raise RuntimeError(
                    f"EODHD 402 Payment Required for {symbol} — "
                    "ticker may require a higher plan"
                ) from e
            raise
        except Exception:
            logger.exception("Failed provider=eodhd series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Convert EODHD JSON response to standard DataFrame.

        Returns DataFrame with columns: series_id, time, value
        where value = close price (yield, index level, FX rate, etc.).
        """
        if not raw:
            logger.warning("EODHD clean: empty payload for %s", series_id)
            return pd.DataFrame(columns=["series_id", "time", "value"])

        df = pd.DataFrame(raw)

        if "date" not in df.columns or "close" not in df.columns:
            raise RuntimeError(
                f"EODHD series_id={series_id}: unexpected response columns: "
                f"{list(df.columns)}"
            )

        df["time"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["close"], errors="coerce")

        before = len(df)
        df = df.dropna(subset=["time", "value"])

        # Drop zero/negative yields only if they look like missing data
        # (JPY can legitimately have negative yields)
        df = df[df["value"] != 0]

        dropped = before - len(df)
        if dropped:
            drop_pct = (dropped / max(before, 1)) * 100
            logger.info(
                "EODHD clean %s: dropped %d rows (%.1f%%)",
                series_id, dropped, drop_pct,
            )
            if strict and drop_pct > MAX_DROP_PCT:
                raise ValueError(
                    f"series_id={series_id}: dropped {drop_pct:.1f}% of rows"
                )

        df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")

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