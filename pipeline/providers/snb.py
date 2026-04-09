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

# Swiss National Bank Data Portal API — free, no auth required
BASE_URL = "https://data.snb.ch/api/cube"
WAREHOUSE_URL = "https://data.snb.ch/api/warehouse/cube"


class SnbProvider(BaseProvider):
    """
    Swiss National Bank (SNB) Data Portal API provider.

    Free access, no API key required.
    Returns semicolon-delimited CSV with a 3-line header:
        Line 1: "CubeId";"<cube_id>"
        Line 2: "PublishingDate";"<date>"
        Line 3: (blank)
        Line 4+: "Date";"D0";"Value" (header + data)

    Each series definition must supply in meta:
        cube      : str — e.g. "plkopr" (headline CPI) or "plkoprex" (core CPI)
        dim_sel   : str — dimension selection, e.g. "D0(LD2010100)" or "D0(K1)"

    URL pattern:
        GET {BASE_URL}/{cube}/data/csv/en?dimSel={dim_sel}&fromDate={YYYY-MM}

    Key cubes:
        plkopr    — Consumer prices total
                    D0(LD2010100) = National index (headline, NSA)
                    D0(VVP)       = YoY % change
        plkoprex  — Consumer prices supplementary classifications
                    D0(K1)        = Core inflation 1 (ex fresh/seasonal & energy)
                    D0(K2)        = Core inflation 2 (ex administered prices)
        conconm   — Consumer confidence survey
                    D0(NIK)       = Consumer sentiment index
    """

    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],  # unused — SNB is keyless
        start: str,
    ) -> Any:
        meta = series_def.meta or {}
        cube = meta.get("cube")
        dim_sel = meta.get("dim_sel")
        warehouse = meta.get("warehouse", False)

        if not cube or not dim_sel:
            raise RuntimeError(
                f"SNB series_id={series_def.series_id} missing 'cube' or "
                f"'dim_sel' in meta"
            )

        if warehouse:
            # New warehouse JSON API (post-2025 migration)
            params = {
                "dimSel": dim_sel,
                "fromDate": start,
            }
            url = f"{WAREHOUSE_URL}/{cube}/data/json/en?{urlencode(params)}"
        else:
            # Legacy CSV API
            params = {
                "dimSel": dim_sel,
                "fromDate": start,
            }
            url = f"{BASE_URL}/{cube}/data/csv/en?{urlencode(params)}"

        logger.info(
            "Fetching provider=snb series_id=%s cube=%s warehouse=%s",
            series_def.series_id, cube, warehouse,
        )
        logger.debug("SNB URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=snb series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()

            text = response.text
            if not text or len(text.strip()) == 0:
                raise RuntimeError(
                    f"SNB returned empty response for series_id={series_def.series_id}"
                )

            if warehouse:
                row_count = text.count('"date"')
            else:
                row_count = max(text.count("\n") - 4, 0)

            logger.info(
                "Fetched provider=snb series_id=%s rows=%d",
                series_def.series_id, row_count,
            )
            return text

        except Exception:
            logger.exception("Failed provider=snb series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Parse SNB response — auto-detects warehouse JSON vs legacy CSV.
        """
        logger.info("Cleaning provider=snb series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"SNB payload empty for series_id={series_id}")

        # Auto-detect format: warehouse JSON starts with { or [
        stripped = raw_payload.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return self._clean_warehouse_json(raw_payload, series_id, strict)
        else:
            return self._clean_legacy_csv(raw_payload, series_id, strict)

    def _clean_warehouse_json(
        self,
        raw_payload: str,
        series_id: str,
        strict: bool,
    ) -> pd.DataFrame:
        """Parse new warehouse JSON API response."""
        import json as _json

        data = _json.loads(raw_payload)
        ts_list = data.get("timeseries", [])
        if not ts_list:
            raise ValueError(f"SNB warehouse: no timeseries for series_id={series_id}")

        values = ts_list[0].get("values", [])
        if not values:
            raise ValueError(f"SNB warehouse: empty values for series_id={series_id}")

        logger.info("Raw rows series_id=%s rows_in=%d", series_id, len(values))

        rows = []
        for v in values:
            rows.append({"date": v.get("date"), "value": v.get("value")})

        df = pd.DataFrame(rows)
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

    def _clean_legacy_csv(
        self,
        raw_payload: str,
        series_id: str,
        strict: bool,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=snb series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"SNB payload empty for series_id={series_id}")

        # Skip the 3-line preamble (CubeId, PublishingDate, blank)
        lines = raw_payload.splitlines()
        data_lines = []
        blank_seen = False
        for line in lines:
            if not blank_seen:
                if line.strip() == "":
                    blank_seen = True
                continue
            data_lines.append(line)

        if not data_lines:
            raise ValueError(
                f"SNB: no data lines found after header for series_id={series_id}"
            )

        csv_text = "\n".join(data_lines)
        df = pd.read_csv(
            io.StringIO(csv_text),
            sep=";",
            quotechar='"',
            header=0,
        )

        # Normalise column names
        df.columns = [c.strip().strip('"').upper() for c in df.columns]

        if "DATE" not in df.columns or "VALUE" not in df.columns:
            raise ValueError(
                f"SNB CSV missing DATE or VALUE columns for series_id={series_id}. "
                f"Columns: {list(df.columns)}"
            )

        logger.info(
            "Raw rows series_id=%s rows_in=%d", series_id, len(df)
        )

        # If multiple dimension values returned (e.g. D0 has "I6" and "VVP6"),
        # keep only the first unique value to avoid duplicate timestamps
        if "D0" in df.columns:
            unique_d0 = df["D0"].dropna().unique()
            if len(unique_d0) > 1:
                keep = unique_d0[0]
                logger.info(
                    "Multiple D0 values %s — keeping '%s' for series_id=%s",
                    unique_d0.tolist(), keep, series_id,
                )
                df = df[df["D0"] == keep]

        # DATE format auto-detect: daily (YYYY-MM-DD), quarterly (YYYY-Qn),
        # or monthly (YYYY-MM)
        sample = str(df["DATE"].dropna().iloc[0]) if not df["DATE"].dropna().empty else ""
        if "Q" in sample:
            # Convert "YYYY-Q1" → first month of quarter
            _q_map = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
            def _parse_quarter(val: str) -> pd.Timestamp:
                try:
                    year, q = val.strip().split("-")
                    month = _q_map.get(q.upper())
                    if not month:
                        return pd.NaT
                    return pd.Timestamp(f"{year}-{month}-01")
                except Exception:
                    return pd.NaT
            df["time"] = df["DATE"].apply(_parse_quarter)
            logger.info("Parsed quarterly DATE format for series_id=%s", series_id)
        elif len(sample) > 7:
            # Daily format: YYYY-MM-DD
            df["time"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce")
            logger.info("Parsed daily DATE format for series_id=%s", series_id)
        else:
            df["time"] = pd.to_datetime(df["DATE"], format="%Y-%m", errors="coerce")
        df["value"] = pd.to_numeric(df["VALUE"], errors="coerce")

        # Filter rows where VALUE is blank/non-numeric (expected for
        # weekends/holidays on daily data, or pre-data periods on quarterly).
        # These are not parse failures — remove them before the drop threshold.
        blank_mask = df["value"].isna() & df["time"].notna()
        blank_count = blank_mask.sum()
        if blank_count > 0:
            logger.info(
                "Filtered %d blank VALUE rows (non-trading/pre-data) for series_id=%s",
                blank_count, series_id,
            )
            df = df[~blank_mask]

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