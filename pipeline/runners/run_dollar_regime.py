"""
Run Dollar Regime Pipeline (standalone)
========================================

Computes dollar quadrant (4-quadrant dollar smile classification).

Prerequisites: RORO v2 must have run first (reads roro2_score_ema10).

Usage:
    python -m pipeline.derived.run_dollar_regime
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)

import time

from ..engine.config import load_macro_config
from ..engine.secrets import get_secret

logger = logging.getLogger(__name__)


def main():
    logger.info("═══════════════════════════════════════════")
    logger.info("  Dollar Regime")
    logger.info("═══════════════════════════════════════════")

    t0 = time.time()

    cfg = load_macro_config()
    secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
    db_config = cfg.build_db_config(secret)
    schema = cfg.db_schema

    from ..derived.regime.dollar_regime import compute_and_store_dollar_regime

    logger.info("── Computing Dollar Regime ──")
    df = compute_and_store_dollar_regime(db_config, schema)

    elapsed = time.time() - t0

    logger.info("═══════════════════════════════════════════")
    logger.info("  Complete in %.1fs (%d rows)", elapsed, len(df))
    logger.info("───────────────────────────────────────────")

    if len(df) > 0:
        latest = df.iloc[-1]

        logger.info("  Real yield axis: %+.3f", latest['real_yield_axis'])
        logger.info("  Risk axis:       %+.3f", latest['risk_axis'])
        logger.info("  Dollar score:    %+.3f", latest['dollar_score'])
        logger.info("  Quadrant:        %s (confidence: %.2f)",
                    latest['dollar_quadrant'], latest['quadrant_confidence'])

        # Quadrant history
        logger.info("")
        logger.info("  QUADRANT HISTORY (last 30 days)")
        last_30 = df.tail(30)
        quad_counts = last_30['dollar_quadrant'].value_counts()
        logger.info("    %s", dict(quad_counts))

    logger.info("═══════════════════════════════════════════")


if __name__ == "__main__":
    main()