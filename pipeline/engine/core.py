"""
Macro Pipeline — Main Orchestrator
====================================
Fetches macro data from ~15 providers, cleans, stores to PostgreSQL,
then runs all derived computations and signal generators in dependency order.

Hardened to TFF/Oanda pipeline parity:
  - Single DB connection passed to every stage
  - stage() context manager with structured timing
  - run_state audit trail (STARTED → SUCCESS / FAILED / SKIPPED)
  - Lambda timeout guard with 60s buffer
  - TCP keepalive on DB connection
  - Secret scrubbing in logs
  - No S3 — CloudWatch handles logs natively
  - No sys.exit — raises for proper Lambda error handling
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone, date

warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

from .logger import configure_logging, get_logger
from .config import load_macro_config
from .secrets import get_secret
from .db_config import get_db_config, open_db_connection
from .series import load_series
from .insert_to_db import upsert_series_data
from .http import create_http_session, HttpConfig, get_provider_session
from .run_state import (
    ensure_run_state_table, insert_started, mark_success,
    mark_failed, mark_skipped,
)
from .email_alerts import send_email_alert, format_failure_email
from .release_alert import check_and_send_release_alert
from ..providers.registry import get_provider

# ── Shared derived (multi-currency) ──────────────────────────────────────────
from ..derived.macro.growth_derived import compute_and_store_growth_derived
from ..derived.macro.labour_derived import compute_and_store_labour_derived
from ..derived.macro.monetary_derived import compute_and_store_monetary_derived
from ..derived.macro.rates_derived import compute_and_store_rates_derived
from ..derived.macro.yields_derived import compute_and_store_yields_derived

# ── Signals + composite ──────────────────────────────────────────────────────
from ..signals.growth_signals import compute_and_store_growth_signals
from ..signals.labour_signals import compute_and_store_labour_signals
from ..signals.monetary_signals import compute_and_store_monetary_signals
from ..signals.rates_signals import compute_and_store_rates_signals
from ..signals.composite import compute_and_store_composite_signals


# Providers that don't require an API key
KEYLESS_PROVIDERS = {
    "eurostat", "ons", "abs", "statcan", "fso",
    "scb", "ssb", "estat", "oecd", "ecb", "snb", "meti",
    "meti_iip", "ecbcs", "boe", "rba", "boc", "boj", "mof", "statsnz_csv", "rbnz", "fso_csv", "bis",
}


class DbStageError(RuntimeError):
    pass


# ─── Stage context manager (matches TFF/Oanda) ───────────────────────────────

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


# ─── Timeout guard ────────────────────────────────────────────────────────────

_TIMEOUT_BUFFER_MS = 60_000


# ─── Lookback helper ─────────────────────────────────────────────────────────

def _lookback_start(lookback_months: int) -> str:
    today = date.today()
    y = today.year + (today.month - 1 - lookback_months) // 12
    m = (today.month - 1 - lookback_months) % 12 + 1
    return f"{y:04d}-{m:02d}"


# ─── Fetch / clean / insert loop ─────────────────────────────────────────────

def run_fetch_clean_loop(
    *,
    logger,
    run_id: str,
    conn,
    db_schema: str,
    start_override: str | None,
    run_type: str,
    session,
    timeout,
    cfg,
    api_keys: dict[str, str],
    series_filter: list[str] | None = None,
    currencies_filter: list[str] | None = None,
    lambda_context=None,
) -> tuple[list[str], list[tuple[str, str, bool]], dict[str, tuple[int, int]], dict[str, int]]:
    """Returns (successes, failures, provider_counts, currency_counts).

    provider_counts: {provider: (succeeded, total)}
    currency_counts: {currency: total_series_count}
    """

    all_series_defs = load_series(currencies=currencies_filter)
    if series_filter:
        unknown = [s for s in series_filter if s not in {d.series_id for d in all_series_defs}]
        if unknown:
            raise ValueError(f"Unknown series IDs: {unknown}")
        series_defs = [d for d in all_series_defs if d.series_id in series_filter]
        logger.info(
            "Series filter active — running %d of %d series: %s",
            len(series_defs), len(all_series_defs), series_filter,
        )
    else:
        series_defs = all_series_defs

    successes: list[str] = []
    failures: list[tuple[str, str, bool]] = []

    # Per-provider and per-currency tracking for email
    provider_total: dict[str, int] = {}
    provider_ok: dict[str, int] = {}
    currency_counts: dict[str, int] = {}
    for sd in series_defs:
        p = sd.provider or "unknown"
        provider_total[p] = provider_total.get(p, 0) + 1
        provider_ok.setdefault(p, 0)
        ccy = (sd.meta or {}).get("currency", "?").lower()
        currency_counts[ccy] = currency_counts.get(ccy, 0) + 1

    # ── BLS batch pre-fetch ───────────────────────────────────────
    # BLS v2 API allows 50 series per request. Pre-fetch all BLS
    # series in one call so the per-series loop hits cache only.
    bls_defs = [sd for sd in series_defs if sd.provider == "bls"]
    if bls_defs:
        bls_provider = get_provider("bls")
        bls_api_key = api_keys.get("bls")
        bls_start = start_override
        if run_type in ("backfill", "reconcile"):
            starts = [
                sd.meta.get("backfill", {}).get("start", "2000-01")
                for sd in bls_defs if sd.meta
            ]
            bls_start = min(starts) if starts else "2000-01"
        try:
            bls_provider.fetch_batch(
                series_defs=bls_defs,
                session=session,
                timeout=timeout,
                api_key=bls_api_key,
                start=bls_start,
            )
            logger.info("BLS batch pre-fetch OK: %d series cached", len(bls_defs))
        except Exception as e:
            logger.error("BLS batch pre-fetch failed: %s — falling back to per-series", e)

    with stage(logger, "FETCH_CLEAN_LOOP", run_id, series_count=len(series_defs), start=start_override):
        for series_def in series_defs:
            # ── Timeout guard per series ──────────────────────────────
            if lambda_context is not None:
                remaining_ms = lambda_context.get_remaining_time_in_millis()
                if remaining_ms < _TIMEOUT_BUFFER_MS:
                    remaining_ids = [s.series_id for s in series_defs if s.series_id not in set(successes)]
                    reason = f"Timeout approaching — {len(remaining_ids)} series remaining"
                    logger.warning(reason)
                    for sid in remaining_ids:
                        if sid not in [s for s, _, _ in failures]:
                            failures.append((sid, "skipped: timeout", False))
                    break

            series_id  = series_def.series_id
            provider   = series_def.provider
            meta       = series_def.meta or {}
            is_required = bool(meta.get("required", False))

            try:
                api_key = api_keys.get(provider)
                if not api_key and provider not in KEYLESS_PROVIDERS:
                    raise RuntimeError(f"Missing API key for provider={provider}")

                if run_type in ("backfill", "reconcile"):
                    backfill_cfg = meta.get("backfill")
                    if not backfill_cfg or "start" not in backfill_cfg:
                        raise RuntimeError(f"Missing backfill.start for {series_id}")
                    start = backfill_cfg["start"]
                else:
                    if not start_override:
                        raise RuntimeError(
                            f"{series_id}: start_override is required for run_type={run_type}"
                        )
                    start = start_override

                db_series_id = meta.get("db_series_id", series_id)
                insert_only  = bool(meta.get("insert_only", False))

                logger.info(
                    "Fetching %s start=%s db_series_id=%s insert_only=%s ...",
                    series_id, start, db_series_id, insert_only,
                )

                p           = get_provider(provider)
                p_session, p_timeout = get_provider_session(provider, session)
                raw_payload = p.fetch(
                    series_def=series_def,
                    session=p_session,
                    timeout=p_timeout,
                    api_key=api_key,
                    start=start,
                )
                df = p.clean(raw_payload, db_series_id, strict=cfg.strict_db)

                # No S3 upload — raw data is trivially re-fetchable from providers

                upsert_series_data(
                    df, conn=conn, schema=db_schema, insert_only=insert_only,
                )

                successes.append(series_id)
                provider_ok[provider] = provider_ok.get(provider, 0) + 1
                logger.info("OK %s", series_id)

            except Exception as e:
                failures.append((series_id, str(e), is_required))
                logger.error(
                    "FAIL %s required=%s error=%s", series_id, is_required, e, exc_info=True,
                )

    logger.info("Finished fetch/clean. Success=%d Fail=%d", len(successes), len(failures))
    provider_counts = {p: (provider_ok.get(p, 0), provider_total[p]) for p in provider_total}
    return successes, failures, provider_counts, currency_counts


# ─── Derived computation stage ────────────────────────────────────────────────

def run_derived_db(
    *,
    logger,
    run_id: str,
    conn,
    db_schema: str,
    run_type: str = "incremental",
    derived_filter: list[str] | None = None,
    lambda_context=None,
) -> str:
    """Returns 'completed/total' string for email reporting."""

    # Signals write last 90 days on incremental, full history on backfill
    signal_kwargs = {}
    if run_type in ("backfill", "reconcile"):
        signal_kwargs["lookback_days"] = 0

    all_modules = [
        # ── Layer 1: Derived (raw metrics) ────────────────────────
        ("labour_derived",      "Labour derived",      compute_and_store_labour_derived,   {}),
        ("growth_derived",      "Growth derived",      compute_and_store_growth_derived,   {}),
        ("monetary_derived",    "Monetary derived",    compute_and_store_monetary_derived, {}),
        ("rates_derived",       "Rates derived",       compute_and_store_rates_derived,    {}),
        ("yields_derived",      "Yields derived",      compute_and_store_yields_derived,   {}),
        # ── Layer 2: Signals (z-scored) ───────────────────────────
        ("growth_signals",      "Growth signals",      compute_and_store_growth_signals,   signal_kwargs),
        ("labour_signals",      "Labour signals",      compute_and_store_labour_signals,   signal_kwargs),
        ("monetary_signals",    "Monetary signals",    compute_and_store_monetary_signals, signal_kwargs),
        ("rates_signals",       "Rates signals",       compute_and_store_rates_signals,    signal_kwargs),
        # ── Layer 3: Composite ────────────────────────────────────
        ("composite_signals",   "Composite signals",   compute_and_store_composite_signals, signal_kwargs),
    ]

    modules_to_run = derived_filter if derived_filter else [name for name, _, _, _ in all_modules]

    if derived_filter:
        logger.info("Derived filter active: %s", ", ".join(modules_to_run))

    total = len(modules_to_run)
    completed = 0

    for name, label, func, extra_kwargs in all_modules:
        if name not in modules_to_run:
            continue

        # ── Timeout guard per derived module ──────────────────────
        if lambda_context is not None:
            remaining_ms = lambda_context.get_remaining_time_in_millis()
            if remaining_ms < _TIMEOUT_BUFFER_MS:
                skipped = [n for n, _, _, _ in all_modules if n in modules_to_run and n not in [name]]
                reason = f"Timeout approaching — skipped derived: {skipped}"
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


# ─── Pipeline entry point ────────────────────────────────────────────────────

def run_pipeline(
    run_type: str = "incremental",
    lambda_context=None,
    *,
    series_filter: list[str] | None = None,
    currencies_filter: list[str] | None = None,
    derived_filter: list[str] | None = None,
    skip_derived: bool = False,
) -> dict:

    configure_logging()
    logger = get_logger(__name__)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    t0 = time.time()
    t0_utc = datetime.now(timezone.utc)
    logger.info("Starting Macro Pipeline run_id=%s", run_id)
    logger.info("run_type=%s", run_type)
    if series_filter:
        logger.info("series_filter=%s", series_filter)
    if currencies_filter:
        logger.info("currencies_filter=%s", currencies_filter)
    if derived_filter:
        logger.info("derived_filter=%s", derived_filter)

    conn = None
    session = None
    db_schema = "macro"
    fail_stage = "PIPELINE"
    already_marked = False

    try:
        # ── CONFIG ────────────────────────────────────────────────────────
        with stage(logger, "CONFIG", run_id):
            cfg = load_macro_config()
            db_schema = cfg.db_schema
            session = create_http_session(HttpConfig())

            api_keys: dict[str, str] = {}
            for provider, secret_name in cfg.api_secrets.items():
                s = get_secret(secret_name, region_name=cfg.region)
                k = s.get("api_key")
                if not k:
                    raise RuntimeError(
                        f"Secret {secret_name} missing 'api_key' for provider={provider}"
                    )
                api_keys[provider] = k

        # ── DB CONNECTION ─────────────────────────────────────────────────
        with stage(logger, "DB_CONNECT", run_id):
            db_secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
            db_config = cfg.build_db_config(db_secret)
            timeout_ms = 600_000 if run_type == "backfill" else 300_000
            conn = open_db_connection(db_config, timeout_ms=timeout_ms)

            ensure_run_state_table(conn, schema=db_schema)
            insert_started(conn=conn, schema=db_schema, run_id=run_id, run_type=run_type)

        # ── COMPUTE START OVERRIDE ────────────────────────────────────────
        if run_type == "incremental":
            start_override = _lookback_start(cfg.lookback_months)
        else:
            start_override = None  # backfill/reconcile use per-series backfill.start

        # ── FETCH / CLEAN / INSERT ────────────────────────────────────────
        successes, failures, provider_counts, currency_counts = run_fetch_clean_loop(
            logger=logger,
            run_id=run_id,
            conn=conn,
            db_schema=db_schema,
            start_override=start_override,
            run_type=run_type,
            session=session,
            timeout=HttpConfig().timeout,
            cfg=cfg,
            api_keys=api_keys,
            series_filter=series_filter,
            currencies_filter=currencies_filter,
            lambda_context=lambda_context,
        )

        # ── GATE: check required series ───────────────────────────────────
        required_failures = [(sid, err) for (sid, err, req) in failures if req]
        optional_failures = [(sid, err) for (sid, err, req) in failures if not req]

        if required_failures:
            missing = [sid for sid, _ in required_failures]
            reason = f"required series failed: {missing}"
            logger.error("Required series failed: %s", missing)
            fail_stage = "DERIVED_GATE"
            mark_failed(
                conn=conn, schema=db_schema, run_id=run_id,
                fail_stage=fail_stage, fail_reason=reason,
            )
            already_marked = True
            raise RuntimeError(reason)

        if optional_failures:
            failed_optional = [sid for sid, _ in optional_failures]
            logger.warning(
                "Optional series failed: %s — continuing to derived stage.",
                failed_optional,
            )

        # ── DERIVED COMPUTATIONS ──────────────────────────────────────────
        if skip_derived:
            logger.info("skip_derived=True — skipping derived/signal modules")
            derived_completed = "0/0 (skipped)"
        else:
            derived_completed = run_derived_db(
                logger=logger,
                run_id=run_id,
                conn=conn,
                db_schema=db_schema,
                run_type=run_type,
                derived_filter=derived_filter,
                lambda_context=lambda_context,
            )

        # ── RELEASE ALERTS — detect new data and email if found ─────
        check_and_send_release_alert(conn, db_schema)

        # ── SUCCESS ───────────────────────────────────────────────────────
        if skip_derived:
            mark_skipped(conn=conn, schema=db_schema, run_id=run_id, reason="skip_derived=True (raw data only)")
        else:
            mark_success(conn=conn, schema=db_schema, run_id=run_id)
        duration_s = time.time() - t0

        series_total = len(successes) + len(failures)

        logger.info("RUN_OK run_id=%s success=%d optional_fail=%d duration=%.1fs",
                     run_id, len(successes), len(optional_failures), duration_s)

        # No email on success — runs every 5 mins.
        # Daily digest email sent by a separate scheduled invocation.

        return {
            "status": "success",
            "run_id": run_id,
            "run_type": run_type,
            "series_fetched": len(successes),
            "series_failed": len(failures),
            "derived_completed": derived_completed,
            "duration_s": round(duration_s, 1),
            "optional_failures": [sid for sid, _ in optional_failures],
        }

    except Exception as e:
        logger.exception("Pipeline crashed unexpectedly")
        if conn and not conn.closed and not already_marked:
            try:
                conn.rollback()
                mark_failed(
                    conn=conn, schema=db_schema, run_id=run_id,
                    fail_stage=fail_stage, fail_reason=str(e),
                )
            except Exception:
                pass

        try:
            subject, body = format_failure_email(
                e=e,
                run_type=run_type,
                fail_stage=fail_stage,
                series_fetched=len(locals().get("successes", [])),
                series_total=len(locals().get("successes", [])) + len(locals().get("failures", [])),
                required_failures=[sid for sid, _ in locals().get("required_failures", [])],
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


# ── Derived-only runner ───────────────────────────────────────────────────────

def run_derived_only(lambda_context=None) -> dict:
    configure_logging()
    logger = get_logger(__name__)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_derived"
    t0 = time.time()
    conn = None
    db_schema = "macro"

    try:
        cfg = load_macro_config()
        db_schema = cfg.db_schema
        db_secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
        db_config = cfg.build_db_config(db_secret)
        conn = open_db_connection(db_config, timeout_ms=300_000)

        ensure_run_state_table(conn, schema=db_schema)
        insert_started(conn=conn, schema=db_schema, run_id=run_id, run_type="derived_only")

        result = run_derived_db(
            logger=logger,
            run_id=run_id,
            conn=conn,
            db_schema=db_schema,
            run_type="derived_only",
            lambda_context=lambda_context,
        )

        check_and_send_release_alert(conn, db_schema)

        mark_success(conn=conn, schema=db_schema, run_id=run_id)

        duration_s = time.time() - t0
        logger.info("DERIVED_ONLY complete: %s in %.1fs", result, duration_s)
        return {"status": "derived_complete", "result": result, "duration_s": round(duration_s, 1)}

    except Exception as e:
        logger.exception("DERIVED_ONLY failed")
        if conn and not conn.closed:
            try:
                conn.rollback()
                mark_failed(
                    conn=conn, schema=db_schema, run_id=run_id,
                    fail_stage="DERIVED_ONLY", fail_reason=str(e),
                )
            except Exception:
                pass

        try:
            subject, body = format_failure_email(
                e=e,
                run_type="derived_only",
                fail_stage="DERIVED_ONLY",
            )
            send_email_alert(subject=subject, body_html=body)
        except Exception:
            logger.warning("Failure email alert failed", exc_info=True)

        raise
    finally:
        if conn and not conn.closed:
            conn.close()


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    action = event.get("action", "pipeline")

    if action == "digest":
        from .run_digest import run_digest
        run_digest()
        return {"status": "digest_sent"}

    if action == "derived":
        return run_derived_only(context)

    run_type = event.get("run_type", "incremental")
    return run_pipeline(
        run_type=run_type,
        lambda_context=context,
        series_filter=event.get("series_filter"),
        currencies_filter=event.get("currencies_filter"),
        derived_filter=event.get("derived_filter"),
        skip_derived=event.get("skip_derived", False),
    )


def main() -> None:
    run_pipeline(run_type="incremental")


if __name__ == "__main__":
    main()