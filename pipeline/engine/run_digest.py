from __future__ import annotations

"""
run_digest.py
=============
Daily digest email — queries macro_run_state for the last 24h
and sends a single summary email.

Schedule via EventBridge: once per day at ~23:00 UTC
Or run manually: python -m pipeline.engine.run_digest

Can also be invoked as a Lambda handler with event {"action": "digest"}.
"""

from .logger import configure_logging, get_logger
from .config import load_macro_config
from .secrets import get_secret
from .db_config import open_db_connection
from .email_alerts import send_daily_digest


def run_digest() -> None:
    configure_logging()
    logger = get_logger(__name__)

    logger.info("Running daily digest email...")

    conn = None
    try:
        cfg = load_macro_config()
        db_secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
        db_config = cfg.build_db_config(db_secret)
        conn = open_db_connection(db_config, timeout_ms=30_000)

        send_daily_digest(conn, schema=cfg.db_schema)
        logger.info("Daily digest sent.")

    except Exception:
        logger.exception("Daily digest failed")
        raise
    finally:
        if conn and not conn.closed:
            try:
                conn.close()
            except Exception:
                pass


def lambda_handler(event: dict, context) -> dict:
    run_digest()
    return {"status": "digest_sent"}


def main() -> None:
    run_digest()


if __name__ == "__main__":
    main()