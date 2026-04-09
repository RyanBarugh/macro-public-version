from __future__ import annotations

import time
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlencode

import requests

from .logger import get_logger
from .secrets import get_secret
from .config import MacroConfig

logger = get_logger(__name__)


def build_census_url(
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


def fetch_census_series(
    session: requests.Session,
    timeout: Tuple[float, float],
    cfg: MacroConfig,
    api_key: Optional[str],
    series_id: str,
    dataset: str,
    get_vars: List[str],
    params: Dict[str, str],
    start: str,
) -> Any:

    url = build_census_url(dataset, get_vars, params, start, api_key)

    logger.info("Fetching Census provider=census series_id=%s", series_id)
    logger.debug("Census URL: %s", url)

    start_ts = time.time()
    try:
        response = session.get(url, timeout=timeout)
        elapsed_ms = int((time.time() - start_ts) * 1000)

        logger.info(
            "HTTP provider=census series_id=%s status=%s elapsed_ms=%d",
            series_id,
            response.status_code,
            elapsed_ms,
        )

        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list) or len(data) < 1:
            raise RuntimeError(f"Census returned invalid JSON shape for series_id={series_id}")

        header = data[0]
        if not isinstance(header, list):
            raise RuntimeError(f"Census header row is not a list for series_id={series_id}")

        if "time" not in header:
            raise RuntimeError(f"Census response missing 'time' column for series_id={series_id}")

        for var in get_vars:
            if var not in header:
                raise RuntimeError(f"Census response missing column '{var}' for series_id={series_id}")

        rows = max(len(data) - 1, 0)
        logger.info("Fetched provider=census series_id=%s rows=%d", series_id, rows)
        return data

    except Exception:
        logger.exception("Failed provider=census series_id=%s", series_id)
        raise