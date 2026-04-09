"""
Equity Pipeline — Core Orchestrator
=====================================

Fetches daily OHLCV for S&P 1500 constituents from EODHD, then runs
derived computations (breadth) in sequence.

Single DB connection passed to every stage — matches macro pipeline parity:
  - stage() context manager with structured timing
  - run_state audit trail (STARTED → SUCCESS / FAILED / SKIPPED)
  - Lambda timeout guard with 60s buffer
  - TCP keepalive on DB connection
  - Secret scrubbing in logs
  - Failure email alerts (via equities/email_alerts.py)

Run types:
  "incremental"  — fetch last 10 days + derived (daily run)
  "backfill"     — full history from 1998 + derived (one-time)

Usage:
    python -m pipeline.equities.run_incremental
    python -m pipeline.equities.run_backfill
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

from ..engine.logger import configure_logging, get_logger
from ..engine.config import load_macro_config
from ..engine.secrets import get_secret
from ..engine.db_config import open_db_connection
from ..engine.http import create_http_session, HttpConfig
from ..engine.run_state import (
    RunStateConfig,
    ensure_run_state_table,
    insert_started,
    mark_success,
    mark_failed,
    mark_skipped,
)
from ..engine.email_alerts import send_email_alert

from .email_alerts import format_failure_email, format_success_email
from .prices import (
    load_constituents_json,
    bulk_fetch_and_store,
    BACKFILL_START,
    UPDATE_LOOKBACK_DAYS,
)
from ..derived.equities.breadth_derived import compute_and_store_breadth
from ..derived.equities.rs_derived import compute_and_store_rs
from ..derived.equities.vcp_derived import compute_and_store_vcp


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

EQUITY_RUN_STATE = RunStateConfig(
    pipeline_name="equity",
    table_name="equity_run_state",
)

EQUITY_SCHEMA = "equity"
_TIMEOUT_BUFFER_MS = 60_000


class DbStageError(RuntimeError):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE CONTEXT MANAGER (matches macro pattern)
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def stage(logger, name: str, run_id: str, **meta):
    start = time.time()
    meta_str = " ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""
    logger.info("STAGE_START %s run_id=%s %s", name, run_id, meta_str)
    try:
        yield
        dur = round(time.time() - start, 3)
        logger.info("STAGE_OK %s run_id=%s duration_s=%s", name, run_id, dur)
    except Exception:
        dur = round(time.time() - start, 3)
        logger.exception("STAGE_FAIL %s run_id=%s duration_s=%s", name, run_id, dur)
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_derived(
    *,
    logger,
    run_id: str,
    conn,
    db_schema: str,
    run_type: str = "incremental",
    lambda_context=None,
) -> str:
    """
    Run all equity derived computations.
    Returns summary string like "1/1".
    Matches macro run_derived_db() pattern.
    """
    all_modules = [
        ("breadth_derived", "Breadth derived", compute_and_store_breadth, {}),
        ("rs_derived", "RS rankings", compute_and_store_rs, {}),
        ("vcp_derived", "VCP scanner", compute_and_store_vcp, {}),
    ]

    total = len(all_modules)
    completed = 0

    for name, label, func, extra_kwargs in all_modules:
        # ── Timeout guard per derived module ──────────────────────
        if lambda_context is not None:
            remaining_ms = lambda_context.get_remaining_time_in_millis()
            if remaining_ms < _TIMEOUT_BUFFER_MS:
                reason = f"Timeout approaching — skipped derived: {name}"
                logger.warning(reason)
                raise RuntimeError(reason)

        with stage(logger, f"DERIVED_{name}", run_id, step=f"{completed+1}/{total}"):
            try:
                func(conn=conn, schema=db_schema, **extra_kwargs)
                completed += 1
            except Exception as e:
                raise DbStageError(f"Derived stage {name} failed: {e}") from e

    logger.info("Derived complete: %d/%d modules", completed, total)
    return f"{completed}/{total}"


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    run_type: str = "incremental",
    lambda_context=None,
    *,
    skip_derived: bool = False,
) -> dict:

    configure_logging()
    logger = get_logger(__name__)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    t0 = time.time()
    logger.info("Starting Equity Pipeline run_id=%s", run_id)
    logger.info("run_type=%s", run_type)

    conn = None
    session = None
    fail_stage = "PIPELINE"
    already_marked = False
    fetch_stats = {}

    try:
        # ── CONFIG ────────────────────────────────────────────────────────
        with stage(logger, "CONFIG", run_id):
            cfg = load_macro_config()
            session = create_http_session(HttpConfig())

            eodhd_api_key = None
            eodhd_secret_id = cfg.api_secrets.get("eodhd")
            if eodhd_secret_id:
                eodhd_secret = get_secret(eodhd_secret_id, cfg.region)
                eodhd_api_key = (
                    eodhd_secret.get("api_key", "")
                    if isinstance(eodhd_secret, dict)
                    else str(eodhd_secret)
                )

            if not eodhd_api_key:
                raise RuntimeError(
                    "EODHD API key not available — set API_SECRET__EODHD in env"
                )

        # ── DB CONNECTION ─────────────────────────────────────────────────
        with stage(logger, "DB_CONNECT", run_id):
            db_secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
            db_config = cfg.build_db_config(db_secret)
            timeout_ms = 600_000 if run_type == "backfill" else 300_000
            conn = open_db_connection(db_config, timeout_ms=timeout_ms)

            ensure_run_state_table(conn, schema=EQUITY_SCHEMA, cfg=EQUITY_RUN_STATE)
            insert_started(
                conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
                run_type=run_type, cfg=EQUITY_RUN_STATE,
            )

        # ── FETCH PRICES ──────────────────────────────────────────────────
        fail_stage = "FETCH_PRICES"

        constituents = load_constituents_json()

        if run_type == "backfill":
            start_date = BACKFILL_START
        else:
            start_date = (
                datetime.now() - timedelta(days=UPDATE_LOOKBACK_DAYS)
            ).strftime("%Y-%m-%d")

        with stage(logger, "FETCH_PRICES", run_id,
                   tickers=len(constituents), start=start_date):
            fetch_stats = bulk_fetch_and_store(
                conn=conn,
                constituents=constituents,
                session=session,
                timeout=HttpConfig().timeout,
                api_key=eodhd_api_key,
                start_date=start_date,
                lambda_context=lambda_context,
            )

        logger.info(
            "Prices: %d fetched, %d failed, %d rows written",
            fetch_stats.get("fetched", 0),
            fetch_stats.get("failed", 0),
            fetch_stats.get("rows_written", 0),
        )

        # ── GATE: check failure rate ──────────────────────────────────────
        total = fetch_stats.get("fetched", 0) + fetch_stats.get("failed", 0)
        if total > 0:
            fail_rate = fetch_stats.get("failed", 0) / total
            if fail_rate > 0.50:
                reason = (
                    f"Too many fetch failures: {fetch_stats['failed']}/{total} "
                    f"({fail_rate:.0%}) — aborting before derived"
                )
                logger.error(reason)
                fail_stage = "FETCH_GATE"
                mark_failed(
                    conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
                    fail_stage=fail_stage, fail_reason=reason,
                    cfg=EQUITY_RUN_STATE,
                )
                already_marked = True
                raise RuntimeError(reason)

        # ── DERIVED COMPUTATIONS ──────────────────────────────────────────
        if skip_derived:
            logger.info("skip_derived=True — skipping derived modules")
            derived_completed = "0/0 (skipped)"
        else:
            derived_completed = run_derived(
                logger=logger,
                run_id=run_id,
                conn=conn,
                db_schema=EQUITY_SCHEMA,
                run_type=run_type,
                lambda_context=lambda_context,
            )

        # ── SUCCESS ───────────────────────────────────────────────────────
        if skip_derived:
            mark_skipped(
                conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
                reason="skip_derived=True (prices only)", cfg=EQUITY_RUN_STATE,
            )
        else:
            mark_success(conn=conn, schema=EQUITY_SCHEMA, run_id=run_id, cfg=EQUITY_RUN_STATE)

        duration_s = time.time() - t0

        logger.info("RUN_OK run_id=%s fetched=%d failed=%d derived=%s duration=%.1fs",
                     run_id,
                     fetch_stats.get("fetched", 0),
                     fetch_stats.get("failed", 0),
                     derived_completed,
                     duration_s)

        # ── Success email ─────────────────────────────────────────────────
        try:
            subject, body = format_success_email(
                run_type=run_type,
                tickers_fetched=fetch_stats.get("fetched", 0),
                tickers_failed=fetch_stats.get("failed", 0),
                tickers_skipped=fetch_stats.get("skipped", 0),
                rows_written=fetch_stats.get("rows_written", 0),
                latest_price_date=fetch_stats.get("latest_price_date"),
                derived_completed=derived_completed,
                duration_s=round(duration_s, 1),
                failed_symbols=fetch_stats.get("failed_symbols", []),
            )
            send_email_alert(subject=subject, body_html=body)
        except Exception:
            logger.warning("Success email failed", exc_info=True)

        return {
            "status": "success",
            "run_id": run_id,
            "run_type": run_type,
            "tickers_fetched": fetch_stats.get("fetched", 0),
            "tickers_failed": fetch_stats.get("failed", 0),
            "rows_written": fetch_stats.get("rows_written", 0),
            "derived_completed": derived_completed,
            "duration_s": round(duration_s, 1),
        }

    except Exception as e:
        logger.exception("Equity pipeline crashed unexpectedly")

        if conn and not conn.closed and not already_marked:
            try:
                conn.rollback()
                mark_failed(
                    conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
                    fail_stage=fail_stage, fail_reason=str(e),
                    cfg=EQUITY_RUN_STATE,
                )
            except Exception:
                pass

        try:
            subject, body = format_failure_email(
                e=e,
                run_type=run_type,
                fail_stage=fail_stage,
                tickers_fetched=fetch_stats.get("fetched", 0),
                tickers_total=len(locals().get("constituents", [])),
                tickers_failed=fetch_stats.get("failed", 0),
            )
            send_email_alert(subject=subject, body_html=body)
        except Exception:
            logger.warning("Failure email alert failed", exc_info=True)

        raise

    finally:
        if conn and not conn.closed:
            try:
                conn.close()
            except Exception:
                pass
        if session:
            try:
                session.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED-ONLY RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_derived_only(lambda_context=None) -> dict:
    """Recompute derived (breadth) without fetching prices."""
    configure_logging()
    logger = get_logger(__name__)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_derived"
    t0 = time.time()
    conn = None

    try:
        cfg = load_macro_config()
        db_secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
        db_config = cfg.build_db_config(db_secret)
        conn = open_db_connection(db_config, timeout_ms=300_000)

        ensure_run_state_table(conn, schema=EQUITY_SCHEMA, cfg=EQUITY_RUN_STATE)
        insert_started(
            conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
            run_type="derived_only", cfg=EQUITY_RUN_STATE,
        )

        result = run_derived(
            logger=logger,
            run_id=run_id,
            conn=conn,
            db_schema=EQUITY_SCHEMA,
            lambda_context=lambda_context,
        )

        mark_success(conn=conn, schema=EQUITY_SCHEMA, run_id=run_id, cfg=EQUITY_RUN_STATE)

        duration_s = time.time() - t0
        logger.info("DERIVED_ONLY complete: %s in %.1fs", result, duration_s)
        return {"status": "derived_complete", "result": result, "duration_s": round(duration_s, 1)}

    except Exception as e:
        logger.exception("DERIVED_ONLY failed")
        if conn and not conn.closed:
            try:
                conn.rollback()
                mark_failed(
                    conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
                    fail_stage="DERIVED_ONLY", fail_reason=str(e),
                    cfg=EQUITY_RUN_STATE,
                )
            except Exception:
                pass

        try:
            subject, body = format_failure_email(
                e=e, run_type="derived_only", fail_stage="DERIVED_ONLY",
            )
            send_email_alert(subject=subject, body_html=body)
        except Exception:
            logger.warning("Failure email alert failed", exc_info=True)

        raise

    finally:
        if conn and not conn.closed:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# RS-ONLY RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_rs_only(lambda_context=None) -> dict:
    """Recompute RS rankings only — no prices, no breadth."""
    configure_logging()
    logger = get_logger(__name__)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_rs"
    t0 = time.time()
    conn = None

    try:
        cfg = load_macro_config()
        db_secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
        db_config = cfg.build_db_config(db_secret)
        conn = open_db_connection(db_config, timeout_ms=300_000)

        ensure_run_state_table(conn, schema=EQUITY_SCHEMA, cfg=EQUITY_RUN_STATE)
        insert_started(
            conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
            run_type="rs_only", cfg=EQUITY_RUN_STATE,
        )

        with stage(logger, "DERIVED_rs_derived", run_id, step="1/1"):
            compute_and_store_rs(conn=conn, schema=EQUITY_SCHEMA)

        mark_success(conn=conn, schema=EQUITY_SCHEMA, run_id=run_id, cfg=EQUITY_RUN_STATE)

        duration_s = time.time() - t0
        logger.info("RS_ONLY complete in %.1fs", duration_s)
        return {"status": "rs_complete", "duration_s": round(duration_s, 1)}

    except Exception as e:
        logger.exception("RS_ONLY failed")
        if conn and not conn.closed:
            try:
                conn.rollback()
                mark_failed(
                    conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
                    fail_stage="RS_ONLY", fail_reason=str(e),
                    cfg=EQUITY_RUN_STATE,
                )
            except Exception:
                pass

        try:
            subject, body = format_failure_email(
                e=e, run_type="rs_only", fail_stage="RS_ONLY",
            )
            send_email_alert(subject=subject, body_html=body)
        except Exception:
            logger.warning("Failure email alert failed", exc_info=True)

        raise

    finally:
        if conn and not conn.closed:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VCP-ONLY RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_vcp_only(lambda_context=None) -> dict:
    """Run VCP scanner only — requires RS rankings to exist."""
    configure_logging()
    logger = get_logger(__name__)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_vcp"
    t0 = time.time()
    conn = None

    try:
        cfg = load_macro_config()
        db_secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
        db_config = cfg.build_db_config(db_secret)
        conn = open_db_connection(db_config, timeout_ms=300_000)

        ensure_run_state_table(conn, schema=EQUITY_SCHEMA, cfg=EQUITY_RUN_STATE)
        insert_started(
            conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
            run_type="vcp_only", cfg=EQUITY_RUN_STATE,
        )

        with stage(logger, "DERIVED_vcp_derived", run_id, step="1/1"):
            compute_and_store_vcp(conn=conn, schema=EQUITY_SCHEMA)

        mark_success(conn=conn, schema=EQUITY_SCHEMA, run_id=run_id, cfg=EQUITY_RUN_STATE)

        duration_s = time.time() - t0
        logger.info("VCP_ONLY complete in %.1fs", duration_s)
        return {"status": "vcp_complete", "duration_s": round(duration_s, 1)}

    except Exception as e:
        logger.exception("VCP_ONLY failed")
        if conn and not conn.closed:
            try:
                conn.rollback()
                mark_failed(
                    conn=conn, schema=EQUITY_SCHEMA, run_id=run_id,
                    fail_stage="VCP_ONLY", fail_reason=str(e),
                    cfg=EQUITY_RUN_STATE,
                )
            except Exception:
                pass

        try:
            subject, body = format_failure_email(
                e=e, run_type="vcp_only", fail_stage="VCP_ONLY",
            )
            send_email_alert(subject=subject, body_html=body)
        except Exception:
            logger.warning("Failure email alert failed", exc_info=True)

        raise

    finally:
        if conn and not conn.closed:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# LAMBDA HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point for the equity pipeline.

    Event format:
        {"run_type": "incremental"}                          — daily update + derived
        {"run_type": "backfill"}                              — full history + derived
        {"action": "derived"}                                 — derived only (breadth + RS)
        {"action": "rs"}                                      — RS rankings only
        {"run_type": "incremental", "skip_derived": true}    — prices only
    """
    action = event.get("action")

    if action == "derived":
        return run_derived_only(context)

    if action == "rs":
        return run_rs_only(context)

    if action == "vcp":
        return run_vcp_only(context)

    run_type = event.get("run_type", "incremental")
    return run_pipeline(
        run_type=run_type,
        lambda_context=context,
        skip_derived=event.get("skip_derived", False),
    )


def main() -> None:
    run_pipeline(run_type="incremental")


if __name__ == "__main__":
    main()