"""
test_emails.py
==============
Test all 3 email types without running the full pipeline.

Usage:
    # Preview all emails as HTML files (no SMTP needed)
    python -m pipeline.engine.test_emails --preview

    # Preview + actually send via Mailjet
    python -m pipeline.engine.test_emails --send

    # Test just one type
    python -m pipeline.engine.test_emails --preview --type failure
    python -m pipeline.engine.test_emails --preview --type digest
    python -m pipeline.engine.test_emails --preview --type release

    # Test with live DB data (queries macro_run_state + series_data)
    python -m pipeline.engine.test_emails --preview --live

HTML files are written to logs/ directory for browser preview.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Bootstrap ─────────────────────────────────────────────────────────────────

if not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

from .logger import configure_logging, get_logger
from .email_alerts import (
    send_email_alert,
    format_failure_email,
    format_daily_digest,
    query_daily_stats,
    query_series_freshness,
)
from .release_alert import format_release_alert

configure_logging()
logger = get_logger(__name__)

OUTPUT_DIR = Path("logs")


# ═════════════════════════════════════════════════════════════════════════════
# MOCK DATA — used when --live is not specified
# ═════════════════════════════════════════════════════════════════════════════

def _mock_failure_email() -> tuple[str, str]:
    """Simulate a DERIVED_GATE failure with required series missing."""
    return format_failure_email(
        e=RuntimeError("required series failed: ['us_cpi_all_items_sa', 'us_nfp_total_sa', 'usd_fed_funds_rate']"),
        run_type="incremental",
        fail_stage="DERIVED_GATE",
        series_fetched=185,
        series_total=200,
        required_failures=["us_cpi_all_items_sa", "us_nfp_total_sa", "usd_fed_funds_rate"],
    )


def _mock_failure_crash_email() -> tuple[str, str]:
    """Simulate an unexpected pipeline crash."""
    return format_failure_email(
        e=ConnectionError("could not connect to server: Connection refused\n\tIs the server running on host \"edgeflow-db.abc123.eu-west-2.rds.amazonaws.com\" and accepting TCP/IP connections on port 5432?"),
        run_type="incremental",
        fail_stage="DB_CONNECT",
        series_fetched=0,
        series_total=0,
    )


def _mock_digest_all_ok() -> tuple[str, str]:
    """Simulate a clean day — all runs succeeded."""
    stats = {
        "total_runs": 288,
        "succeeded": 288,
        "failed": 0,
        "latest_run_status": "SUCCESS",
        "latest_run_ts": "23:55 UTC",
        "skipped": 0,
        "last_success_ts": "23:55 UTC",
        "last_fail_ts": None,
        "last_fail_stage": None,
        "last_fail_reason": None,
        "failures": [],
    }
    freshness = {
        "series_updated": 47,
        "latest_update": "13:35 UTC",
    }
    mock_ladder = [{"pair":"AUDCAD","value":1.783},{"pair":"NZDCAD","value":1.655},{"pair":"AUDCHF","value":1.577},{"pair":"AUDJPY","value":1.446},{"pair":"AUDUSD","value":1.202},{"pair":"USDCAD","value":0.581},{"pair":"EURGBP","value":0.092},{"pair":"EURJPY","value":-0.012},{"pair":"GBPJPY","value":-0.103},{"pair":"EURUSD","value":-0.255},{"pair":"GBPUSD","value":-0.347},{"pair":"EURAUD","value":-1.457},{"pair":"GBPAUD","value":-1.549}]
    return format_daily_digest(stats, freshness, pair_ladder=mock_ladder)


def _mock_digest_with_failures() -> tuple[str, str]:
    """Simulate a day with some failures."""
    stats = {
        "total_runs": 288,
        "succeeded": 282,
        "failed": 4,
        "skipped": 2,
        "latest_run_status": "SUCCESS",
        "latest_run_ts": "23:55 UTC",
        "last_success_ts": "23:55 UTC",
        "last_fail_ts": "14:20 UTC",
        "last_fail_stage": "DERIVED_GATE",
        "last_fail_reason": "required series failed: ['us_nfp_total_sa']",
        "failures": [
            {"run_id": "2026-04-02T14-20-00Z", "fail_stage": "DERIVED_GATE", "fail_reason": "required series failed: ['us_nfp_total_sa']", "start_ts": "14:20"},
            {"run_id": "2026-04-02T14-15-00Z", "fail_stage": "FETCH_CLEAN_LOOP", "fail_reason": "ConnectionError: FRED API timeout", "start_ts": "14:15"},
            {"run_id": "2026-04-02T08-30-00Z", "fail_stage": "DB_CONNECT", "fail_reason": "connection refused", "start_ts": "08:30"},
            {"run_id": "2026-04-02T08-25-00Z", "fail_stage": "DB_CONNECT", "fail_reason": "connection refused", "start_ts": "08:25"},
        ],
    }
    freshness = {
        "series_updated": 47,
        "latest_update": "13:35 UTC",
    }
    mock_ladder = [{"pair":"AUDCAD","value":1.783},{"pair":"NZDCAD","value":1.655},{"pair":"AUDCHF","value":1.577},{"pair":"AUDJPY","value":1.446},{"pair":"AUDUSD","value":1.202},{"pair":"USDCAD","value":0.581},{"pair":"EURGBP","value":0.092},{"pair":"EURJPY","value":-0.012},{"pair":"GBPJPY","value":-0.103},{"pair":"EURUSD","value":-0.255},{"pair":"GBPUSD","value":-0.347},{"pair":"EURAUD","value":-1.457},{"pair":"GBPAUD","value":-1.549}]
    return format_daily_digest(stats, freshness, pair_ladder=mock_ladder)


def _mock_release_alert_nfp() -> tuple[str, str] | None:
    """Simulate NFP release — Employment Situation publication."""
    releases = {
        "Employment Situation": [
            {
                "series_id": "us_nfp_total_sa",
                "currency": "USD",
                "publication_name": "Employment Situation",
                "source_agency": "BLS",
                "time": datetime(2026, 3, 1),
                "value": 228000,
                "previous_value": 175000,
                "updated_at": datetime.now(timezone.utc),
                "revised_at": datetime.now(timezone.utc),
            },
            {
                "series_id": "us_nfp_private_sa",
                "currency": "USD",
                "publication_name": "Employment Situation",
                "source_agency": "BLS",
                "time": datetime(2026, 3, 1),
                "value": 205000,
                "previous_value": 160000,
                "updated_at": datetime.now(timezone.utc),
                "revised_at": datetime.now(timezone.utc),
            },
            {
                "series_id": "us_unemployment_rate_sa",
                "currency": "USD",
                "publication_name": "Employment Situation",
                "source_agency": "BLS",
                "time": datetime(2026, 3, 1),
                "value": 4.1,
                "previous_value": 4.0,
                "updated_at": datetime.now(timezone.utc),
                "revised_at": datetime.now(timezone.utc),
            },
            {
                "series_id": "us_avg_hourly_earnings_sa",
                "currency": "USD",
                "publication_name": "Employment Situation",
                "source_agency": "BLS",
                "time": datetime(2026, 3, 1),
                "value": 0.3,
                "previous_value": 0.4,
                "updated_at": datetime.now(timezone.utc),
                "revised_at": datetime.now(timezone.utc),
            },
        ],
    }
    pair_ladder = [
        {"pair": "AUDCAD", "value": 1.783, "delta": 0.042},
        {"pair": "NZDCAD", "value": 1.655, "delta": 0.038},
        {"pair": "AUDCHF", "value": 1.577, "delta": 0.025},
        {"pair": "NZDCHF", "value": 1.448, "delta": 0.021},
        {"pair": "AUDJPY", "value": 1.446, "delta": 0.035},
        {"pair": "NZDJPY", "value": 1.317, "delta": 0.031},
        {"pair": "AUDUSD", "value": 1.202, "delta": 0.049},
        {"pair": "NZDUSD", "value": 1.073, "delta": 0.045},
        {"pair": "USDCAD", "value": 0.581, "delta": -0.007},
        {"pair": "USDCHF", "value": 0.375, "delta": -0.017},
        {"pair": "EURCAD", "value": 0.326, "delta": -0.016},
        {"pair": "USDJPY", "value": 0.244, "delta": -0.004},
        {"pair": "GBPCAD", "value": 0.234, "delta": 0.001},
        {"pair": "AUDNZD", "value": 0.128, "delta": 0.004},
        {"pair": "EURCHF", "value": 0.119, "delta": -0.008},
        {"pair": "EURGBP", "value": 0.092, "delta": -0.017},
        {"pair": "GBPCHF", "value": 0.028, "delta": 0.009},
        {"pair": "EURJPY", "value": -0.012, "delta": -0.021},
        {"pair": "GBPJPY", "value": -0.103, "delta": -0.004},
        {"pair": "CHFJPY", "value": -0.131, "delta": 0.013},
        {"pair": "CADCHF", "value": -0.207, "delta": -0.010},
        {"pair": "EURUSD", "value": -0.255, "delta": -0.065},
        {"pair": "CADJPY", "value": -0.338, "delta": 0.003},
        {"pair": "GBPUSD", "value": -0.347, "delta": -0.048},
        {"pair": "EURNZD", "value": -1.329, "delta": -0.059},
        {"pair": "GBPNZD", "value": -1.421, "delta": -0.042},
        {"pair": "EURAUD", "value": -1.457, "delta": -0.091},
        {"pair": "GBPAUD", "value": -1.549, "delta": -0.074},
    ]
    return format_release_alert(releases, pair_ladder)


def _mock_release_alert_multi() -> tuple[str, str] | None:
    """Simulate multiple releases at once — CPI + UK GDP."""
    releases = {
        "Consumer Price Index": [
            {
                "series_id": "us_cpi_all_items_sa",
                "currency": "USD",
                "publication_name": "Consumer Price Index",
                "source_agency": "BLS",
                "time": datetime(2026, 3, 1),
                "value": 313.2,
                "previous_value": 312.8,
                "updated_at": datetime.now(timezone.utc),
                "revised_at": datetime.now(timezone.utc),
            },
            {
                "series_id": "us_cpi_core_sa",
                "currency": "USD",
                "publication_name": "Consumer Price Index",
                "source_agency": "BLS",
                "time": datetime(2026, 3, 1),
                "value": 321.5,
                "previous_value": 321.0,
                "updated_at": datetime.now(timezone.utc),
                "revised_at": datetime.now(timezone.utc),
            },
        ],
        "GDP First Estimate": [
            {
                "series_id": "gbp_gdp_real",
                "currency": "GBP",
                "publication_name": "GDP First Estimate",
                "source_agency": "ONS",
                "time": datetime(2026, 1, 1),
                "value": 0.3,
                "previous_value": 0.1,
                "updated_at": datetime.now(timezone.utc),
                "revised_at": datetime.now(timezone.utc),
            },
        ],
    }
    # Same ladder — in real usage this comes from the DB
    pair_ladder = [
        {"pair": "AUDCAD", "value": 1.783, "delta": 0.042},
        {"pair": "NZDCAD", "value": 1.655, "delta": 0.038},
        {"pair": "AUDCHF", "value": 1.577, "delta": 0.025},
        {"pair": "AUDJPY", "value": 1.446, "delta": 0.035},
        {"pair": "AUDUSD", "value": 1.202, "delta": 0.049},
        {"pair": "USDCAD", "value": 0.581, "delta": -0.007},
        {"pair": "EURGBP", "value": 0.092, "delta": -0.017},
        {"pair": "EURJPY", "value": -0.012, "delta": -0.021},
        {"pair": "GBPJPY", "value": -0.103, "delta": -0.004},
        {"pair": "EURUSD", "value": -0.255, "delta": -0.065},
        {"pair": "GBPUSD", "value": -0.347, "delta": -0.048},
        {"pair": "EURAUD", "value": -1.457, "delta": -0.091},
        {"pair": "GBPAUD", "value": -1.549, "delta": -0.074},
    ]
    return format_release_alert(releases, pair_ladder)


# ═════════════════════════════════════════════════════════════════════════════
# LIVE DATA — queries real DB
# ═════════════════════════════════════════════════════════════════════════════

def _live_digest(conn, schema: str) -> tuple[str, str]:
    """Query real run_state, series_data, and pair signals for digest."""
    from .email_alerts import query_pair_ladder
    stats = query_daily_stats(conn, schema)
    freshness = query_series_freshness(conn, schema)
    pair_ladder = query_pair_ladder(conn, schema)
    return format_daily_digest(stats, freshness, pair_ladder=pair_ladder)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _save_html(name: str, subject: str, body: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"test_email_{name}.html"
    path.write_text(body, encoding="utf-8")
    logger.info("Saved: %s — Subject: %s", path, subject)
    return path


def _open_db():
    """Open a DB connection for live tests."""
    from .config import load_macro_config
    from .secrets import get_secret
    from .db_config import open_db_connection

    cfg = load_macro_config()
    db_secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
    db_config = cfg.build_db_config(db_secret)
    return open_db_connection(db_config, timeout_ms=30_000), cfg.db_schema


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test macro pipeline emails")
    parser.add_argument("--preview", action="store_true", help="Save HTML files to logs/ for browser preview")
    parser.add_argument("--send", action="store_true", help="Actually send via Mailjet SMTP")
    parser.add_argument("--live", action="store_true", help="Query real DB for digest data")
    parser.add_argument("--type", choices=["failure", "digest", "release", "all"], default="all",
                        help="Which email type to test (default: all)")
    args = parser.parse_args()

    if not args.preview and not args.send:
        print("Specify --preview and/or --send")
        print("  --preview  saves HTML to logs/ for browser viewing")
        print("  --send     actually sends via SMTP")
        print("  --live     uses real DB data for digest")
        return

    emails: list[tuple[str, str, str]] = []  # (name, subject, body)

    # ── Failure emails ────────────────────────────────────────────
    if args.type in ("failure", "all"):
        s, b = _mock_failure_email()
        emails.append(("failure_derived_gate", s, b))

        s, b = _mock_failure_crash_email()
        emails.append(("failure_db_crash", s, b))

    # ── Digest emails ─────────────────────────────────────────────
    if args.type in ("digest", "all"):
        if args.live:
            conn, schema = _open_db()
            try:
                s, b = _live_digest(conn, schema)
                emails.append(("digest_live", s, b))
            finally:
                conn.close()
        else:
            s, b = _mock_digest_all_ok()
            emails.append(("digest_all_ok", s, b))

            s, b = _mock_digest_with_failures()
            emails.append(("digest_with_failures", s, b))

    # ── Release alert emails ──────────────────────────────────────
    if args.type in ("release", "all"):
        result = _mock_release_alert_nfp()
        if result:
            s, b = result
            emails.append(("release_nfp", s, b))

        result = _mock_release_alert_multi()
        if result:
            s, b = result
            emails.append(("release_multi_cpi_gdp", s, b))

    # ── Output ────────────────────────────────────────────────────
    if not emails:
        logger.warning("No emails generated")
        return

    for name, subject, body in emails:
        if args.preview:
            path = _save_html(name, subject, body)

        if args.send:
            logger.info("Sending: %s", subject)
            send_email_alert(subject=f"[TEST] {subject}", body_html=body)

    if args.preview:
        print(f"\n  {len(emails)} HTML files saved to {OUTPUT_DIR}/")
        print(f"  Open in browser to preview.\n")

    if args.send:
        print(f"\n  {len(emails)} emails sent (prefixed with [TEST]).\n")


if __name__ == "__main__":
    main()