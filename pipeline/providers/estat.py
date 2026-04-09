from __future__ import annotations

import re
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

BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
CATALOG_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsList"

_PAGE_SIZE = 100_000


class EStatProvider(BaseProvider):
    """
    Japan e-Stat provider with auto-discovery of latest annual correction table.

    METI publishes retail sales index data as annual snapshot tables on e-Stat.
    Each year (typically March) a new table is published with monthly observations
    for that year plus overlap. This provider discovers the latest table via the
    getStatsList catalog API at fetch time.

    Discovery logic:
        1. Search catalog for statsCode + keyword
        2. Filter to tables where STATISTICS_NAME contains both "確報" and "年間補正"
        3. Sort by survey_date descending, take the latest
        4. Use that statsDataId for the data fetch

    series.json fields used:
        stats_code       : str  - e.g. "00550030" (default)
        catalog_keyword  : str  - search keyword (default: "業種別商業販売額指数")
        cd_cat01         : str  - category 1 filter (e.g. "0102000" = retail total)
        cd_cat02         : str  - category 2 filter (e.g. "01080300" = SA index)
        stats_data_id    : str  - optional override to skip discovery

    Requires API key (appId). lang=J required for METI tables.
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
                f"e-Stat requires an API key (appId) for series_id={series_def.series_id}"
            )

        meta = series_def.meta or {}

        # Step 1: Discover latest statsDataId via catalog search
        stats_data_id = meta.get("stats_data_id")

        if not stats_data_id:
            stats_data_id = self._discover_latest_table(
                session=session,
                timeout=timeout,
                api_key=api_key,
                stats_code=meta.get("stats_code", "00550030"),
                keyword=meta.get("catalog_keyword", "\u696d\u7a2e\u5225\u5546\u696d\u8ca9\u58f2\u984d\u6307\u6570"),
                series_id=series_def.series_id,
            )

        logger.info(
            "Using statsDataId=%s for series_id=%s",
            stats_data_id, series_def.series_id,
        )

        # Step 2: Fetch data
        params: dict = {
            "appId": api_key,
            "lang": "J",
            "statsDataId": stats_data_id,
            "metaGetFlg": "Y",
            "cntGetFlg": "N",
            "startPosition": 1,
            "limit": _PAGE_SIZE,
        }

        if meta.get("cd_cat01"):
            params["cdCat01"] = meta["cd_cat01"]
        if meta.get("cd_cat02"):
            params["cdCat02"] = meta["cd_cat02"]
        if meta.get("cd_cat03"):
            params["cdCat03"] = meta["cd_cat03"]
        if meta.get("cd_cat04"):
            params["cdCat04"] = meta["cd_cat04"]
        if meta.get("cd_tab"):
            params["cdTab"] = meta["cd_tab"]
        if meta.get("cd_area"):
            params["cdArea"] = meta["cd_area"]

        url = f"{BASE_URL}?{urlencode(params)}"

        logger.info(
            "Fetching provider=estat series_id=%s stats_data_id=%s",
            series_def.series_id, stats_data_id,
        )
        logger.debug("e-Stat URL: %s", url)

        start_ts = time.time()
        try:
            response = session.get(url, timeout=timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                "HTTP provider=estat series_id=%s status=%s elapsed_ms=%d",
                series_def.series_id, response.status_code, elapsed_ms,
            )
            response.raise_for_status()
            data = response.json()

            result = data.get("GET_STATS_DATA", {}).get("RESULT", {})
            status = result.get("STATUS")
            if status != 0:
                raise RuntimeError(
                    f"e-Stat API error status={status} msg={result.get('ERROR_MSG')} "
                    f"for series_id={series_def.series_id}"
                )

            stat_data = data["GET_STATS_DATA"]["STATISTICAL_DATA"]
            total = stat_data.get("RESULT_INF", {}).get("TOTAL_NUMBER", 0)
            logger.info(
                "Fetched provider=estat series_id=%s total_records=%d",
                series_def.series_id, total,
            )

            if int(total) > _PAGE_SIZE:
                logger.warning(
                    "e-Stat series_id=%s total_records=%d exceeds page_size=%d. "
                    "Only first page retrieved.",
                    series_def.series_id, total, _PAGE_SIZE,
                )

            return stat_data

        except Exception:
            logger.exception("Failed provider=estat series_id=%s", series_def.series_id)
            raise

    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        logger.info("Cleaning provider=estat series_id=%s strict=%s", series_id, strict)

        if not raw_payload:
            raise ValueError(f"e-Stat payload empty for series_id={series_id}")

        # Build time code lookup from CLASS_INF metadata when present.
        # Some tables (e.g. METI IIP) use opaque internal @time codes that map
        # to YYYYMM strings via the "time" CLASS_OBJ. Other tables (e.g. MIC CPI)
        # use YYYYMM directly as @time codes — for those, the lookup will be empty
        # and _parse_estat_time handles them as before.
        time_lookup: dict[str, str] = {}
        class_inf = raw_payload.get("CLASS_INF", {})
        class_objs = class_inf.get("CLASS_OBJ", [])
        if isinstance(class_objs, dict):
            class_objs = [class_objs]
        for obj in class_objs:
            if obj.get("@id") == "time":
                classes = obj.get("CLASS", [])
                if isinstance(classes, dict):
                    classes = [classes]
                for c in classes:
                    code = c.get("@code", "")
                    name = c.get("@name", "")
                    # Map entries whose @name looks like YYYYMM (6 digits)
                    # or Japanese format e.g. "2024年1月" / "2024年12月"
                    if len(name) == 6 and name.isdigit():
                        time_lookup[code] = name
                    else:
                        m = re.match(r"(\d{4})年(\d{1,2})月", name)
                        if m:
                            time_lookup[code] = f"{m.group(1)}{int(m.group(2)):02d}"
                logger.info(
                    "Built time lookup series_id=%s entries=%d",
                    series_id, len(time_lookup),
                )
                break

        data_inf = raw_payload.get("DATA_INF", {})
        values = data_inf.get("VALUE", [])
        if isinstance(values, dict):
            values = [values]

        logger.info(
            "Raw datapoints series_id=%s rows_in=%d", series_id, len(values)
        )

        _skip = frozenset([
            "", "-", "\u2026", "***", "x", "X", "\u00d7",
            "\u3000-", "\u3000\uff0d", "\uff0d", " -", "   -",
            "       \u2026", "\u2025", " \u3000-", "\u3000 -", "x ",
        ])

        records = []
        for v in values:
            time_code = v.get("@time", "")
            obs_value = v.get("$")

            if obs_value is None or obs_value in _skip:
                continue

            # Resolve time: use metadata lookup first, fall back to direct parse
            if time_lookup:
                yyyymm = time_lookup.get(time_code)
                if yyyymm is None:
                    # Code not in lookup (e.g. weight rows) — skip
                    continue
                time_str = f"{yyyymm[:4]}-{yyyymm[4:6]}-01"
            else:
                time_str = _parse_estat_time(time_code)
                if time_str is None:
                    continue

            records.append({"time": time_str, "value": obs_value})

        df = pd.DataFrame(records) if records else pd.DataFrame(columns=["time", "value"])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
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

    def _discover_latest_table(
        self,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: str,
        stats_code: str,
        keyword: str,
        series_id: str,
    ) -> str:
        """
        Search e-Stat catalog for the latest annual correction table.
        Filters to STATISTICS_NAME containing both "確報" and "年間補正".
        Returns statsDataId of the most recent table by survey_date.
        """
        params = {
            "appId": api_key,
            "lang": "J",
            "statsCode": stats_code,
            "searchWord": keyword,
            "limit": 50,
        }

        url = f"{CATALOG_URL}?{urlencode(params)}"

        logger.info(
            "Discovering latest e-Stat table for series_id=%s stats_code=%s",
            series_id, stats_code,
        )

        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            result = data.get("GET_STATS_LIST", {}).get("RESULT", {})
            if result.get("STATUS") != 0:
                raise RuntimeError(
                    f"e-Stat catalog search failed: {result.get('ERROR_MSG')}"
                )

            tables = data["GET_STATS_LIST"]["DATALIST_INF"].get("TABLE_INF", [])
            if isinstance(tables, dict):
                tables = [tables]

            candidates = []
            for t in tables:
                stat_name = t.get("STATISTICS_NAME", "")
                if "\u5e74\u9593\u88dc\u6b63" in stat_name and "\u78ba\u5831" in stat_name:
                    table_id = t.get("@id", "")
                    survey_date = t.get("SURVEY_DATE", "")
                    open_date = t.get("OPEN_DATE", "")
                    candidates.append({
                        "id": table_id,
                        "name": stat_name,
                        "survey_date": survey_date,
                        "open_date": open_date,
                    })

            if not candidates:
                raise RuntimeError(
                    f"No annual correction tables found in e-Stat catalog for "
                    f"series_id={series_id}"
                )

            candidates.sort(key=lambda c: c["survey_date"], reverse=True)
            best = candidates[0]

            logger.info(
                "Discovered latest table: id=%s name=%s survey_date=%s",
                best["id"], best["name"], best["survey_date"],
            )

            return best["id"]

        except Exception:
            logger.exception(
                "Failed to discover e-Stat table for series_id=%s", series_id
            )
            raise


def _parse_estat_time(time_code: str) -> str | None:
    """
    Parse e-Stat time codes into "YYYY-MM-01" strings.
    Monthly: mid2 == last2 (e.g. "2024000101" -> "2024-01-01")
    Annual/quarterly: skip.
    """
    if not time_code or not time_code[0].isdigit():
        return None

    if len(time_code) == 10:
        year = time_code[:4]
        if time_code[4:] == "000000":
            return None
        last2 = time_code[8:10]
        mid2 = time_code[6:8]
        if mid2 != last2:
            return None
        try:
            m = int(last2)
            if 1 <= m <= 12:
                return f"{year}-{last2}-01"
        except ValueError:
            pass
        return None

    if len(time_code) == 6:
        return f"{time_code[:4]}-{time_code[4:6]}-01"

    if len(time_code) == 8:
        return f"{time_code[:4]}-{time_code[4:6]}-{time_code[6:8]}"

    return None