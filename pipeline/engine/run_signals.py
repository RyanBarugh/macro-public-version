"""
Run Signals Pipeline
=====================

Computes all signal layers in sequence:
    1. Inflation signals  (reads inflation_derived → writes inflation_signals)
    2. Growth signals     (reads growth_derived → writes growth_signals)
    3. Labour signals     (reads labour_derived → writes labour_signals)
    4. Composite signals  (reads block signals → writes composite_signals)

Usage:
    python -m pipeline.engine.run_signals
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from .logger import configure_logging
configure_logging()

import sys
import time

from .config import load_macro_config
from .secrets import get_secret
from .logger import get_logger

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
    from ..signals.inflation_signals import compute_and_store_inflation_signals
    from ..signals.growth_signals import compute_and_store_growth_signals
    from ..signals.labour_signals import compute_and_store_labour_signals
    from ..signals.composite import compute_and_store_composite_signals

    logger.info("── Step 1/4: Inflation signals ──")
    t1 = time.time()
    inflation_panels = compute_and_store_inflation_signals(db_config, schema)
    logger.info("  Done in %.1fs", time.time() - t1)

    logger.info("── Step 2/4: Growth signals ──")
    t2 = time.time()
    growth_panels = compute_and_store_growth_signals(db_config, schema)
    logger.info("  Done in %.1fs", time.time() - t2)

    logger.info("── Step 3/4: Labour signals ──")
    t3 = time.time()
    labour_panels = compute_and_store_labour_signals(db_config, schema)
    logger.info("  Done in %.1fs", time.time() - t3)

    # ── Composite ─────────────────────────────────────────────────
    # Pass pre-computed block scores to avoid DB round-trip
    block_panels = {}
    if "inflation_score" in inflation_panels:
        block_panels["inflation_score"] = inflation_panels["inflation_score"]
    if "growth_score" in growth_panels:
        block_panels["growth_score"] = growth_panels["growth_score"]
    if "labour_score" in labour_panels:
        block_panels["labour_score"] = labour_panels["labour_score"]

    logger.info("── Step 4/4: Composite signals ──")
    t4 = time.time()
    compute_and_store_composite_signals(db_config, schema, block_panels=block_panels)
    logger.info("  Done in %.1fs", time.time() - t4)

    elapsed = time.time() - t0
    logger.info("═══════════════════════════════════════════")
    logger.info("  Signals pipeline complete in %.1fs", elapsed)
    logger.info("═══════════════════════════════════════════")


if __name__ == "__main__":
    main()