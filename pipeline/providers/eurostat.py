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

# v1 — legacy JSON-STAT API (existing series: IP, retail, HICP, PPI)
BASE_URL_V1 = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

# v3 — SDMX 3.0 REST API (new labour series and beyond)
BASE_URL_V3 = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT"

_QUARTER_TO_MONTH = {"1": "01", "2": "04", "3": "07", "4": "10"}


def _parse_eurostat_time(label: str) -> str:
    """
    Normalise Eurostat time labels to ISO date strings.

    Quarterly:  "2025-Q1" -> "2025-01-01"
    Monthly:    "2000-01" -> "2000-01-01"  (pass-through)
    Annual:     "2000"    -> "2000-01-01"  (pass-through)
    """
    if isinstance(label, str) and "-Q" in label:
        try:
            year, q = label.split("-Q")
            month = _QUARTER_TO_MONTH[q]
            return f"{year}-{month}-01"
        except (ValueError, KeyError):
            return label
    return label


class EurostatProvider(BaseProvider):
    """
    Eurostat provider supporting two API versions:

    v1 (default) — legacy JSON-STAT API
        series.json: "dataset" + "params" dict
        URL: /statistics/1.0/data/{dataset}?format=JSON&sinceTimePeriod=...&param=val

    v3 — SDMX 3.0 REST API
        series.json: "api_version": "v3", "dataset", "sdmx_key", "filters" dict
        URL: /sdmx/3.0/data/dataflow/ESTAT/{dataset}/1.0/{sdmx_key}
             ?c[param]=val&c[TIME_PERIOD]=ge:{start}&compress=false&format=json&lang=en

        sdmx_key: wildcard dimension key matching the dataset's dimension count
                  e.g. "*.*.*.*.*.*.*" for a 7-dimension dataset.
                  Count from the URL you verified in the browser — one * per dimension.

        filters:  dict of dimension filters, e.g.:
                  {"freq": "Q", "s_adj": "SA", "sex": "T", "geo": "EA20", ...}
                  These become c[freq]=Q&c[s_adj]=SA etc. in the query string.
                  Do NOT include TIME_PERIOD here — it is injected from `start`.
    """

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — Eurostat is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        api_version = meta.get("api_version", "v1")

        if api_version == "v3":
            return self._fetch_v3(series_def, session, timeout, start, meta)
        else:
            return self._fetch_v1(series_def, session, timeout, start, meta)

    def _fetch_v1(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        start: str,
        meta: dict,
    ) -> Any:
        dataset = meta.get("dataset")
        params = meta.get("params", {})

        if not dataset:
            raise RuntimeError(
                f"Missing 'dataset' for series_id={series_def.series_id}"
            )

        query = {
            "format": "JSON",
            "sinceTimePeriod": start,
            **params,
        }

        url = f"{BASE_URL_V1}/{dataset}?{urlencode(query)}"
        return self._get(series_def.series_id, session, url, timeout)

    def _fetch_v3(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        start: str,
        meta: dict,
    ) -> Any:
        dataset = meta.get("dataset")
        sdmx_key = meta.get("sdmx_key", "*")
        filters = meta.get("filters", {})

        if not dataset:
            raise RuntimeError(
                f"Missing 'dataset' for series_id={series_def.series_id}"
            )

        # Build c[param]=value pairs — TIME_PERIOD uses ge: (greater-or-equal)
        filter_parts = [f"c[{k}]={v}" for k, v in filters.items()]
        filter_parts.append(f"c[TIME_PERIOD]=ge:{start}")
        filter_parts += ["compress=false", "format=json", "lang=en"]
        query_string = "&".join(filter_parts)

        url = f"{BASE_URL_V3}/{dataset}/1.0/{sdmx_key}?{query_string}"
        return self._get(series_def.series_id, session, url, timeout)

    def _get(
        self,
        series_id: str,
        session: requests.Session,
        url: str,
        timeout: Tuple[float, float],
    ) -> Any:
        logger.info("Fetching provider=eurostat series_id=%s", series_id)
        logger.debug("Eurostat URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)

            logger.info(
                "HTTP provider=eurostat series_id=%s status=%s elapsed_ms=%d",
                series_id,
                response.status_code,
                elapsed_ms,
            )

            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Eurostat returned unexpected type for series_id={series_id}"
                )

            rows = (
                len(data.get("value", {}))          # v1
                if "value" in data
                else self._v3_row_count(data)        # v3
            )
            logger.info(
                "Fetched provider=eurostat series_id=%s values=%d",
                series_id,
                rows,
            )
            return data

        except Exception:
            logger.exception("Failed provider=eurostat series_id=%s", series_id)
            raise

    @staticmethod
    def _v3_row_count(data: dict) -> int:
        try:
            series = data["data"]["dataSets"][0]["series"]
            return sum(
                len(s.get("observations", {}))
                for s in series.values()
            )
        except (KeyError, IndexError):
            return 0

    # ------------------------------------------------------------------
    # clean
    # ------------------------------------------------------------------

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=eurostat series_id=%s strict=%s", series_id, strict)

        # Detect version from payload shape
        if "value" in raw_payload:
            return self._clean_v1(raw_payload, series_id, strict)
        elif "data" in raw_payload and "dataSets" in raw_payload.get("data", {}):
            return self._clean_v3(raw_payload, series_id, strict)
        else:
            raise ValueError(
                f"Eurostat payload unrecognised format for series_id={series_id}"
            )

    def _clean_v1(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool,
    ) -> pd.DataFrame:
        """Parse legacy JSON-STAT response."""
        if not raw_payload or "value" not in raw_payload:
            raise ValueError(f"Eurostat v1 payload empty or malformed for series_id={series_id}")

        try:
            time_dim = raw_payload["dimension"]["time"]
            time_labels = list(time_dim["category"]["label"].values())
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Eurostat v1 payload missing time dimension for series_id={series_id}: {e}"
            )

        values_raw = raw_payload["value"]
        n = len(time_labels)
        logger.info(
            "Raw values series_id=%s time_periods=%d values_present=%d",
            series_id, n, len(values_raw),
        )

        records = []
        for i, t in enumerate(time_labels):
            v = values_raw.get(i) or values_raw.get(str(i))
            records.append({"time": t, "value": v})

        return self._finalise(pd.DataFrame(records), series_id, strict)

    def _clean_v3(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool,
    ) -> pd.DataFrame:
        """
        Parse SDMX-JSON 2.0 response (v3 API).

        Structure:
          data.structures[0].dimensions.observation[0]  — TIME_PERIOD dimension
            .values: [{"id": "2025-Q3"}, ...]           — ordered time labels
          data.dataSets[0].series                        — keyed by dim position string
            {"0:0:0:0:0:0": {"observations": {"0": [val], "1": [val], ...}}}
        """
        try:
            structure = raw_payload["data"]["structures"][0]
            obs_dims = structure["dimensions"]["observation"]
            time_dim = next(d for d in obs_dims if d["id"] == "TIME_PERIOD")
            time_labels = [v["id"] for v in time_dim["values"]]
        except (KeyError, TypeError, StopIteration) as e:
            raise ValueError(
                f"Eurostat v3 payload missing TIME_PERIOD dimension for series_id={series_id}: {e}"
            )

        try:
            dataset_series = raw_payload["data"]["dataSets"][0]["series"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Eurostat v3 payload missing dataSets/series for series_id={series_id}: {e}"
            )

        if len(dataset_series) != 1:
            raise ValueError(
                f"Eurostat v3 expected exactly 1 series key, "
                f"got {len(dataset_series)} for series_id={series_id}. "
                f"Check filters — multiple dimension values may be returning."
            )

        observations = next(iter(dataset_series.values())).get("observations", {})

        n = len(time_labels)
        logger.info(
            "Raw values series_id=%s time_periods=%d values_present=%d",
            series_id, n, len(observations),
        )

        records = []
        for i, t in enumerate(time_labels):
            obs = observations.get(i) or observations.get(str(i))
            # Each observation is a list; first element is the primary value
            v = obs[0] if obs is not None else None
            records.append({"time": t, "value": v})

        return self._finalise(pd.DataFrame(records), series_id, strict)

    def _finalise(
        self,
        df: pd.DataFrame,
        series_id: str,
        strict: bool,
    ) -> pd.DataFrame:
        """Shared post-processing: parse times, drop nulls, dedup, format."""
        df["time"] = df["time"].apply(_parse_eurostat_time)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
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