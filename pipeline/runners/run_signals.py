"""
Run Signals Pipeline
=====================

Computes all signal layers in sequence:
    1. Monetary signals  (reads monetary_derived → writes monetary_signals)
    2. Growth signals    (reads growth_derived → writes growth_signals)
    3. Labour signals    (reads labour_derived → writes labour_signals)
    4. Rates signals     (reads rates_derived → writes rates_signals)
    5. Composite signals (reads block signals → writes composite_signals)

Usage:
    python -m pipeline.engine.run_signals
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from ..engine.logger import configure_logging
configure_logging()

import sys
import time

from ..engine.config import load_macro_config
from ..engine.secrets import get_secret
from ..engine.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("═══════════════════════════════════════════")
    logger.info("  EdgeFlow Signals Pipeline")
    logger.info("═══════════════════════════════════════════")

    t0 = time.time()

    cfg = load_macro_config()
    secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
    db_config = cfg.build_db_config(secret)
    schema = cfg.db_schema

    # ── Block signals ─────────────────────────────────────────────
    from ..signals.monetary_signals import compute_and_store_monetary_signals
    from ..signals.growth_signals import compute_and_store_growth_signals
    from ..signals.labour_signals import compute_and_store_labour_signals
    from ..signals.rates_signals import compute_and_store_rates_signals
    from ..signals.composite import compute_and_store_composite_signals

    logger.info("── Step 1/5: Monetary signals ──")
    t1 = time.time()
    monetary_panels = compute_and_store_monetary_signals(db_config, schema)
    logger.info("  Done in %.1fs", time.time() - t1)

    logger.info("── Step 2/5: Growth signals ──")
    t2 = time.time()
    growth_panels = compute_and_store_growth_signals(db_config, schema)
    logger.info("  Done in %.1fs", time.time() - t2)

    logger.info("── Step 3/5: Labour signals ──")
    t3 = time.time()
    labour_panels = compute_and_store_labour_signals(db_config, schema)
    logger.info("  Done in %.1fs", time.time() - t3)

    logger.info("── Step 4/5: Rates signals ──")
    t4 = time.time()
    rates_panels = compute_and_store_rates_signals(db_config, schema)
    logger.info("  Done in %.1fs", time.time() - t4)

    # ── Composite ─────────────────────────────────────────────────
    # Pass pre-computed block scores to avoid DB round-trip
    block_panels = {}
    if "monetary_score" in monetary_panels:
        block_panels["monetary_score"] = monetary_panels["monetary_score"]
    if "growth_score" in growth_panels:
        block_panels["growth_score"] = growth_panels["growth_score"]
    if "labour_score" in labour_panels:
        block_panels["labour_score"] = labour_panels["labour_score"]
    if rates_panels and "rates_score" in rates_panels:
        block_panels["rates_score"] = rates_panels["rates_score"]

    logger.info("── Step 5/5: Composite signals ──")
    t5 = time.time()
    compute_and_store_composite_signals(db_config, schema, block_panels=block_panels)
    logger.info("  Done in %.1fs", time.time() - t5)

    elapsed = time.time() - t0
    logger.info("═══════════════════════════════════════════")
    logger.info("  Signals pipeline complete in %.1fs", elapsed)
    logger.info("═══════════════════════════════════════════")


if __name__ == "__main__":
    main()