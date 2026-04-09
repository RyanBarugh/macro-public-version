"""
Run Signals Pipeline v2
========================

Computes all v2 signal layers in sequence:
    1. Growth derived v2   (excess/change metrics → growth_derived)
    2. ToT derived         (commodity baskets → tot_derived)
    3. COT derived         (CME futures positioning → cot_derived)
    4. Growth signals v2   (reads growth_derived → writes growth_signals_v2)
    5. Labour signals v2   (reads labour_derived → writes labour_signals_v2)
    6. Monetary signals v2 (reads monetary_derived → writes monetary_signals_v2)
   6b. Yields derived     (reads rates_derived + series_data → writes yields_derived)
    7. Rates signals v2    (reads yields_derived → writes rates_signals_v2)
    8. ToT signals v2      (reads tot_derived → writes tot_signals_v2)
    9. COT signals         (reads cot_derived → writes cot_signals)
   10. Composite v2        (reads all v2 block signals → writes composite_signals_v2)

Usage:
    python -m pipeline.engine.run_signals_v2
    python -m pipeline.engine.run_signals_v2 --skip-derived   # skip derived, just signals
    python -m pipeline.engine.run_signals_v2 --only-composite  # only composite from existing blocks
    python -m pipeline.engine.run_signals_v2 --only-cot        # COT derived + signals + composite only
    python -m pipeline.engine.run_signals_v2 --only-rates      # Rates signals + composite only
"""

from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from ..engine.logger import configure_logging
configure_logging()

from ..engine.config import load_macro_config
from ..engine.secrets import get_secret
from ..engine.db_config import open_db_connection
from ..engine.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="EdgeFlow Signals v2 Pipeline")
    parser.add_argument("--skip-derived", action="store_true",
                        help="Skip derived layer, run signals only")
    parser.add_argument("--only-composite", action="store_true",
                        help="Only run composite from existing block signals in DB")
    parser.add_argument("--only-cot", action="store_true",
                        help="Only run COT derived + COT signals + composite (skips all other blocks)")
    parser.add_argument("--only-rates", action="store_true",
                        help="Only run Rates signals + composite (skips all other blocks)")
    args = parser.parse_args()

    # Convenience: which blocks to run
    run_all = not (args.only_composite or args.only_cot or args.only_rates)
    run_macro_derived = run_all and not args.skip_derived
    run_macro_signals = run_all
    run_cot = run_all or args.only_cot
    run_rates = run_all or args.only_rates

    logger.info("═══════════════════════════════════════════")
    logger.info("  EdgeFlow Signals Pipeline v2")
    if args.only_cot:
        logger.info("  Mode: COT + composite only")
    elif args.only_rates:
        logger.info("  Mode: Rates + composite only")
    logger.info("═══════════════════════════════════════════")

    t0 = time.time()

    cfg = load_macro_config()
    secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
    db_config = cfg.build_db_config(secret)
    schema = cfg.db_schema

    # ── Open shared connection for all blocks ─────────────────────
    conn = open_db_connection(db_config, timeout_ms=600_000)

    block_panels = {}

    try:
        if not args.only_composite:

            # ── Step 1: Growth Derived v2 ─────────────────────────
            if run_macro_derived:
                logger.info("── Step 1/10: Growth derived v2 ──")
                t1 = time.time()
                from ..derived.macro.growth_derived_v2 import compute_and_store_growth_derived_v2
                compute_and_store_growth_derived_v2(db_config, schema)
                logger.info("  Done in %.1fs", time.time() - t1)
            else:
                logger.info("── Step 1/10: Growth derived v2 — SKIPPED ──")

            # ── Step 2: ToT Derived ───────────────────────────────
            if run_macro_derived:
                logger.info("── Step 2/10: ToT derived ──")
                t2 = time.time()
                from ..derived.macro.tot_derived import compute_and_store_tot_derived
                compute_and_store_tot_derived(conn, schema)
                logger.info("  Done in %.1fs", time.time() - t2)
            else:
                logger.info("── Step 2/10: ToT derived — SKIPPED ──")

            # ── Step 3: COT Derived ───────────────────────────────
            if run_cot and not args.skip_derived:
                logger.info("── Step 3/10: COT derived ──")
                t3 = time.time()
                from ..derived.cot.cot_derived import compute_and_store_cot_derived
                compute_and_store_cot_derived(conn, schema)
                logger.info("  Done in %.1fs", time.time() - t3)
            else:
                logger.info("── Step 3/10: COT derived — SKIPPED ──")

            # ── Step 4: Growth Signals v2 ─────────────────────────
            if run_macro_signals:
                logger.info("── Step 4/10: Growth signals v2 ──")
                t3 = time.time()
                from ..signals.v2.growth_signals_v2 import compute_and_store_growth_signals_v2
                growth_panels = compute_and_store_growth_signals_v2(conn, schema)
                if growth_panels and "growth_score_v2" in growth_panels:
                    block_panels["growth_score_v2"] = growth_panels["growth_score_v2"]
                logger.info("  Done in %.1fs", time.time() - t3)
            else:
                logger.info("── Step 4/10: Growth signals v2 — SKIPPED ──")

            # ── Step 5: Labour Signals v2 ─────────────────────────
            if run_macro_signals:
                logger.info("── Step 5/10: Labour signals v2 ──")
                t4 = time.time()
                from ..signals.v2.labour_signals_v2 import compute_and_store_labour_signals_v2
                labour_panels = compute_and_store_labour_signals_v2(conn, schema)
                if labour_panels and "labour_score_v2" in labour_panels:
                    block_panels["labour_score_v2"] = labour_panels["labour_score_v2"]
                logger.info("  Done in %.1fs", time.time() - t4)
            else:
                logger.info("── Step 5/10: Labour signals v2 — SKIPPED ──")

            # ── Step 6: Monetary Signals v2 ───────────────────────
            if run_macro_signals:
                logger.info("── Step 6/10: Monetary signals v2 ──")
                t5 = time.time()
                from ..signals.v2.monetary_signals_v2 import compute_and_store_monetary_signals_v2
                monetary_panels = compute_and_store_monetary_signals_v2(conn, schema)
                if monetary_panels and "monetary_score_v2" in monetary_panels:
                    block_panels["monetary_score_v2"] = monetary_panels["monetary_score_v2"]
                logger.info("  Done in %.1fs", time.time() - t5)
            else:
                logger.info("── Step 6/10: Monetary signals v2 — SKIPPED ──")

            # ── Step 6b: Yields Derived ──────────────────────────
            if run_rates and not args.skip_derived:
                logger.info("── Step 6b/11: Yields derived ──")
                t6b = time.time()
                from ..derived.macro.yields_derived import compute_and_store_yields_derived
                compute_and_store_yields_derived(conn, schema)
                logger.info("  Done in %.1fs", time.time() - t6b)
            else:
                logger.info("── Step 6b/11: Yields derived — SKIPPED ──")

            # ── Step 7: Rates Signals v2 ──────────────────────────
            if run_rates:
                logger.info("── Step 7/10: Rates signals v2 ──")
                t6 = time.time()
                from ..signals.v2.rates_signals_v2 import compute_and_store_rates_signals_v2
                rates_panels = compute_and_store_rates_signals_v2(conn, schema)
                if rates_panels and "rates_score_v2" in rates_panels:
                    block_panels["rates_score_v2"] = rates_panels["rates_score_v2"]
                logger.info("  Done in %.1fs", time.time() - t6)
            else:
                logger.info("── Step 7/10: Rates signals v2 — SKIPPED ──")

            # ── Step 8: ToT Signals v2 ────────────────────────────
            if run_macro_signals:
                logger.info("── Step 8/10: ToT signals v2 ──")
                t7 = time.time()
                from ..signals.v2.tot_signals_v2 import compute_and_store_tot_signals_v2
                tot_panels = compute_and_store_tot_signals_v2(conn, schema)
                if tot_panels and "tot_score_v2" in tot_panels:
                    block_panels["tot_score_v2"] = tot_panels["tot_score_v2"]
                logger.info("  Done in %.1fs", time.time() - t7)
            else:
                logger.info("── Step 8/10: ToT signals v2 — SKIPPED ──")

            # ── Step 9: COT Signals ───────────────────────────────
            if run_cot:
                logger.info("── Step 9/10: COT signals ──")
                t9 = time.time()
                from ..signals.v2.cot_signals import compute_and_store_cot_signals
                cot_panels = compute_and_store_cot_signals(conn, schema)
                if cot_panels and "cot_score" in cot_panels:
                    block_panels["cot_score"] = cot_panels["cot_score"]
                logger.info("  Done in %.1fs", time.time() - t9)
            else:
                logger.info("── Step 9/10: COT signals — SKIPPED ──")

        # ── Step 10: Composite v2 ──────────────────────────────────
        logger.info("── Step 10/10: Composite signals v2 ──")
        t8 = time.time()
        from pipeline.signals.v2.composite_v2 import compute_and_store_composite_signals
        # For --only-* modes, don't pass block_panels — let composite load
        # ALL blocks from DB (other blocks have data from prior full runs).
        # Passing a partial dict would make the composite only see those blocks.
        partial_mode = args.only_cot or args.only_rates
        use_panels = None if partial_mode else (block_panels if block_panels else None)
        compute_and_store_composite_signals(
            conn=conn,
            schema=schema,
            block_panels=use_panels,
        )
        logger.info("  Done in %.1fs", time.time() - t8)

        elapsed = time.time() - t0
        logger.info("═══════════════════════════════════════════")
        logger.info("  Signals v2 pipeline complete in %.1fs", elapsed)
        logger.info("═══════════════════════════════════════════")

    except Exception:
        logger.exception("Signals v2 pipeline failed")
        raise
    finally:
        if conn and not conn.closed:
            conn.close()
            logger.info("DB connection closed.")


if __name__ == "__main__":
    main()