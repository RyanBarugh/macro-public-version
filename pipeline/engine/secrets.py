from __future__ import annotations

"""
secrets.py
==========
AWS Secrets Manager client with in-process cache.

5-min TTL balances cost (fewer API calls) against freshness
(picks up rotated credentials within a few minutes on warm Lambda invocations).

Matches TFF/Oanda pipeline pattern exactly.
"""

import json
import time
import boto3
from typing import Dict, Any, Tuple
from botocore.exceptions import ClientError

from .logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL_SECONDS = 300

_SECRET_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def get_secret(secret_name: str, region_name: str) -> Dict[str, Any]:
    key = (region_name, secret_name)
    now = time.monotonic()

    cached = _SECRET_CACHE.get(key)
    if cached and (now - cached["fetched_at"]) < _CACHE_TTL_SECONDS:
        return cached["value"]

    if cached:
        logger.info("Cache expired for secret '%s' — re-fetching", secret_name)

    secret = _fetch_secret(secret_name, region_name)
    _SECRET_CACHE[key] = {"value": secret, "fetched_at": now}
    return secret


def clear_cache() -> None:
    """Clear the secret cache. Useful for testing or forced refresh."""
    _SECRET_CACHE.clear()


def _fetch_secret(secret_id: str, region: str) -> Dict[str, Any]:
    logger.info("Fetching secret '%s' from region '%s'", secret_id, region)
    client = boto3.client("secretsmanager", region_name=region)

    try:
        resp = client.get_secret_value(SecretId=secret_id)

        if "SecretString" in resp and resp["SecretString"]:
            return json.loads(resp["SecretString"])
        if "SecretBinary" in resp and resp["SecretBinary"]:
            return json.loads(resp["SecretBinary"].decode("utf-8"))

        raise RuntimeError(f"Secret '{secret_id}' returned empty value")

    except ClientError:
        logger.exception("Failed to retrieve secret '%s'", secret_id)
        raise