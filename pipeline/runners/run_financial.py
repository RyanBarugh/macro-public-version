"""
Run Financial Conditions Pipeline (standalone)
===============================================

Computes FC composite + regime only.

Usage:
    python -m pipeline.derived.run_financial_conditions
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
    logger.info("  Financial Conditions (FC Only)")
    logger.info("═══════════════════════════════════════════")

    t0 = time.time()

    cfg = load_macro_config()
    secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
    db_config = cfg.build_db_config(secret)
    schema = cfg.db_schema

    from ..derived.regime.financial_conditions import compute_and_store_financial_conditions

    logger.info("── Computing FC layer ──")
    df = compute_and_store_financial_conditions(db_config, schema)

    elapsed = time.time() - t0

    logger.info("═══════════════════════════════════════════")
    logger.info("  Complete in %.1fs (%d rows)", elapsed, len(df))
    logger.info("───────────────────────────────────────────")

    if len(df) > 0:
        latest = df.iloc[-1]

        logger.info("  FC score:     %.3f (EMA10: %.3f)", latest['fc_score'], latest['fc_score_ema10'])
        logger.info("  FC regime:    %s (day %s)", latest['fc_regime'], latest['fc_regime_days'])
        logger.info("  FC pctl/dir:  P%.0f / %+.3f", latest['fc_percentile'], latest['fc_direction'])
        logger.info("  Buckets:      rates=%.2f  credit=%.2f  liq=%.2f  lev=%.2f  fund=%.2f",
                    latest['fc_rates'], latest['fc_credit'], latest['fc_liquidity'],
                    latest['fc_leverage'], latest['fc_funding'])

    logger.info("═══════════════════════════════════════════")


if __name__ == "__main__":
    main()