from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests

from .base import BaseProvider
from ..engine.series import SeriesDef
from ..engine.logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0


class CensusProvider(BaseProvider):

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        dataset = meta.get("dataset")
        get_vars = meta.get("get", ["cell_value"])
        params = meta.get("params", {})

        if isinstance(get_vars, str):
            get_vars = [get_vars]

        url = self._build_url(dataset, get_vars, params, start, api_key)

        logger.info("Fetching provider=census series_id=%s", series_def.series_id)
        logger.debug("Census URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)

            logger.info(
                "HTTP provider=census series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id,
                response.status_code,
                elapsed_ms,
            )

            response.raise_for_status()
            data = response.json()

            if not isinstance(data, list) or len(data) < 1:
                raise RuntimeError(
                    f"Census returned invalid JSON shape for series_id={series_def.series_id}"
                )

            header = data[0]
            if not isinstance(header, list):
                raise RuntimeError(
                    f"Census header row is not a list for series_id={series_def.series_id}"
                )

            if "time" not in header:
                raise RuntimeError(
                    f"Census response missing 'time' column for series_id={series_def.series_id}"
                )

            for var in get_vars:
                if var not in header:
                    raise RuntimeError(
                        f"Census response missing column '{var}' for series_id={series_def.series_id}"
                    )

            rows = max(len(data) - 1, 0)
            logger.info(
                "Fetched provider=census series_id=%s rows=%d",
                series_def.series_id,
                rows,
            )
            return data

        except Exception:
            logger.exception("Failed provider=census series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=census series_id=%s strict=%s", series_id, strict)

        if not raw_payload or len(raw_payload) < 2:
            raise ValueError(f"Census payload empty or malformed for series_id={series_id}")

        meta_row = raw_payload[0]
        rows = raw_payload[1:]

        value_col = "cell_value"

        logger.info(
            "Raw rows (excluding header) series_id=%s rows_in=%d",
            series_id,
            len(rows),
        )

        df = pd.DataFrame(rows, columns=meta_row)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")

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

    def _build_url(
        self,
        dataset: str,
        get_vars: List[str],
        params: Dict[str, str],
        start: str,
        api_key: Optional[str],
    ) -> str:
        query = {
            "get": ",".join(get_vars),
            **params,
            "time": f"from {start}",
        }
        if api_key:
            query["key"] = api_key
        return f"{dataset}?{urlencode(query)}"