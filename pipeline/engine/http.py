from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class HttpConfig:

    timeout: Tuple[float, float] = (5.0, 30.0)

    total_retries: int = 6
    backoff_factor: float = 0.6 
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504)

    pool_connections: int = 20
    pool_maxsize: int = 20


def create_http_session(cfg: Optional[HttpConfig] = None) -> requests.Session:
    cfg = cfg or HttpConfig()

    retry = Retry(
        total=cfg.total_retries,
        connect=cfg.total_retries,
        read=cfg.total_retries,
        status=cfg.total_retries,
        backoff_factor=cfg.backoff_factor,
        status_forcelist=cfg.status_forcelist,
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True, 
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=cfg.pool_connections,
        pool_maxsize=cfg.pool_maxsize,
    )

    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    s.headers.update({"User-Agent": "edgeflow-macro-pipeline/1.0"})
    return s


# ── Per-provider overrides ────────────────────────────────────────────────────
# Providers that are slow or unreliable get reduced retries to avoid
# burning Lambda time on hopeless requests.

PROVIDER_HTTP_OVERRIDES: dict[str, HttpConfig] = {
    "meti":     HttpConfig(total_retries=1, backoff_factor=0.3, timeout=(5.0, 30.0)),
    "meti_iip": HttpConfig(total_retries=1, backoff_factor=0.3, timeout=(5.0, 30.0)),
}


def get_provider_session(
    provider: str,
    default_session: requests.Session,
    *,
    _cache: dict[str, requests.Session] = {},
) -> tuple[requests.Session, Tuple[float, float]]:
    """Return (session, timeout) for a provider.

    Uses the default shared session unless the provider has an override
    in PROVIDER_HTTP_OVERRIDES, in which case a dedicated session is
    created (and cached for the process lifetime).
    """
    override = PROVIDER_HTTP_OVERRIDES.get(provider)
    if override is None:
        return default_session, HttpConfig().timeout

    if provider not in _cache:
        _cache[provider] = create_http_session(override)
    return _cache[provider], override.timeout