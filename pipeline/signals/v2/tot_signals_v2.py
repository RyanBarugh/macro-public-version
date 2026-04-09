"""
ToT Signals v2 — JPMaQS-Aligned Block Scoring (2 factors)
============================================================

Reads ToT momentum metrics from macro.tot_derived, scores them
using panel-pooled MAD z-scores (neutral="zero"), composites into
factors and a block score with re-scoring at every aggregation step.

Follows the identical architecture to growth_signals_v2.py,
labour_signals_v2.py, monetary_signals_v2.py, rates_signals_v2.py.

Scoring pipeline:
    1. Load each constituent into wide panel (dates × 8 currencies)
    2. FFill to daily business days
    3. Z-score: make_zn_scores(panel, neutral="zero", pan_weight=0.3)
    4. Equal-weight within factors → rescore (if >1 constituent)
    5. Equal-weight across factors → rescore (if >1 factor)
    6. → tot_score_v2 (8 currencies)

Factor structure:
    F1 (annual_momentum):   tot_12m_chg
    F2 (quarterly_momentum): tot_3m_chg

Both constituents are change metrics — already centered around zero.
The 12m change is the most academically validated ToT-FX predictor
(Macrosynergy, Chen & Rogoff). The 3m change adds responsiveness
for faster-moving commodity shocks.

NOTE: No percentile rank, no rolling window. Uses the same expanding
MAD z-score with cross-sectional demeaning as every other block.
The design document's percentile rank proposal was written before the
v2 scoring engine was finalised and is superseded by this approach.

Reads from:  macro.tot_derived
Writes to:   macro.tot_signals_v2

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.tot_signals_v2 (
        currency    TEXT          NOT NULL,
        series_id   TEXT          NOT NULL,
        time        DATE          NOT NULL,
        value       FLOAT,
        updated_at  TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_tot_signals_v2_time
        ON macro.tot_signals_v2(time);
    CREATE INDEX IF NOT EXISTS idx_tot_signals_v2_currency
        ON macro.tot_signals_v2(currency);
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import warnings
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

from pipeline.engine.logger import get_logger
from ..zn_scores_v2 import make_zn_scores, linear_composite, rescore, ffill_to_daily

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_TABLE = "tot_signals_v2"
SOURCE_TABLE = "tot_derived"

CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]

# Factor definitions.
# Signs: +1 = higher value is bullish for the currency.
# Improving ToT (positive change) is currency-positive → sign = +1.
FACTORS = {
    "annual_momentum": {
        "constituents": ["tot_12m_chg"],
        "signs":        [+1],
    },
    "quarterly_momentum": {
        "constituents": ["tot_3m_chg"],
        "signs":        [+1],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_panel(
    metric_suffix: str,
    conn,
    schema: str,
    sign: int = 1,
) -> pd.DataFrame:
    """
    Load a single ToT metric into a wide panel (dates × 8 currencies).

    Reads from tot_derived, pivots on estimated_release_date for PIT
    correctness, then forward-fills to daily business days.
    """
    # Build the list of series_ids to load
    series_ids = [f"{ccy}_{metric_suffix}" for ccy in CURRENCIES]

    placeholders = ",".join(["%s"] * len(series_ids))
    sql = f"""
        SELECT currency, series_id, time, value, estimated_release_date
        FROM {schema}.{SOURCE_TABLE}
        WHERE series_id IN ({placeholders})
        ORDER BY time
    """
    df = pd.read_sql(sql, conn, params=series_ids)

    if df.empty:
        logger.warning("  No data for %s in %s.%s", metric_suffix, schema, SOURCE_TABLE)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value"])

    # PIT gating: ToT is derived from daily commodity prices — no real
    # publication lag. Use estimated_release_date if populated, otherwise
    # fall back to observation date. Consistent with rates_signals_v2.
    if "estimated_release_date" in df.columns:
        df["estimated_release_date"] = pd.to_datetime(
            df["estimated_release_date"], errors="coerce"
        )
        has_rd = df["estimated_release_date"].notna()
        if has_rd.sum() == 0:
            logger.warning(
                "  PIT: %s — no release dates, falling back to observation date",
                metric_suffix,
            )
            df["pivot_date"] = df["time"]
        else:
            n_dropped = (~has_rd).sum()
            if n_dropped > 0:
                logger.info(
                    "  PIT: %s — %d rows missing release date, dropped",
                    metric_suffix, n_dropped,
                )
            df = df[has_rd].copy()
            df["pivot_date"] = df["estimated_release_date"]
    else:
        df["pivot_date"] = df["time"]
        logger.info("  PIT: %s — no release date column, using observation date", metric_suffix)

    # Pivot to wide: dates × currencies
    panel = df.pivot_table(
        index="pivot_date", columns="currency", values="value", aggfunc="last"
    )
    panel = panel.sort_index()

    # Ensure all 8 currencies present (fill missing with NaN)
    for ccy in CURRENCIES:
        if ccy not in panel.columns:
            panel[ccy] = np.nan
    panel = panel[CURRENCIES]

    # FFill to daily business days
    panel = ffill_to_daily(panel)

    if sign == -1:
        panel = panel * -1

    logger.info(
        "  Loaded %s: %d dates × %d currencies, range %s to %s",
        metric_suffix, len(panel), panel.notna().any(axis=0).sum(),
        panel.index[0].date(), panel.index[-1].date(),
    )
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _score_factor(
    factor_name: str,
    factor_def: dict,
    conn,
    schema: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Score a single factor: load constituents → z-score → composite
    → rescore (if multi-constituent).

    All ToT constituents are change metrics — neutral="zero" for everything.
    """
    constituent_zn = {}

    for suffix, sign in zip(factor_def["constituents"], factor_def["signs"]):
        panel = _load_panel(suffix, conn, schema, sign=sign)
        if panel.empty:
            logger.warning("  Skipping constituent %s — no data", suffix)
            continue

        # All ToT metrics are changes — already centered around zero
        zn = make_zn_scores(panel, neutral="zero", pan_weight=0.3)
        constituent_zn[suffix] = zn
        logger.info(
            "  Z-scored %s: mean|z|=%.3f, range [%.2f, %.2f]",
            suffix,
            np.nanmean(np.abs(zn.values)),
            np.nanmin(zn.values), np.nanmax(zn.values),
        )

    if not constituent_zn:
        logger.warning("  Factor %s: no valid constituents", factor_name)
        return pd.DataFrame(), {}

    # Single constituent: no averaging happened, no compression to undo
    if len(constituent_zn) == 1:
        factor_score = list(constituent_zn.values())[0]
    else:
        composite = linear_composite(constituent_zn)
        factor_score = rescore(composite)

    logger.info(
        "  Factor %s: mean|z|=%.3f, range [%.2f, %.2f]",
        factor_name,
        np.nanmean(np.abs(factor_score.values)),
        np.nanmin(factor_score.values), np.nanmax(factor_score.values),
    )

    return factor_score, constituent_zn


def compute_tot_signals_v2(
    conn,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Full ToT block v2 scoring pipeline.

    Returns dict of all scored panels:
        "zn_{suffix}"       — z-scored constituents (8 currencies)
        "{factor_name}"     — factor scores (rescored if multi-constituent)
        "tot_score_v2"      — final block composite (rescored if multi-factor)
    """
    logger.info("═══ ToT Signals v2 (2-factor) ═══")

    all_panels = {}
    factor_scores = {}

    for factor_name, factor_def in FACTORS.items():
        logger.info("Factor: %s", factor_name)
        factor_score, constituent_zn = _score_factor(
            factor_name, factor_def, conn, schema,
        )

        for suffix, zn in constituent_zn.items():
            all_panels[f"zn_{suffix}"] = zn

        if not factor_score.empty:
            factor_scores[factor_name] = factor_score
            all_panels[factor_name] = factor_score

    if not factor_scores:
        logger.error("No valid factors — cannot compute ToT block v2 score")
        return all_panels

    # Block composite: equal-weight across factors → rescore if multi-factor
    if len(factor_scores) == 1:
        block_score = list(factor_scores.values())[0]
    else:
        block_composite = linear_composite(factor_scores)
        block_score = rescore(block_composite)

    all_panels["tot_score_v2"] = block_score

    logger.info(
        "═══ ToT block v2 score: mean|z|=%.3f, range [%.2f, %.2f] ═══",
        np.nanmean(np.abs(block_score.values)),
        np.nanmin(block_score.values), np.nanmax(block_score.values),
    )

    return all_panels


# ═══════════════════════════════════════════════════════════════════════════════
# WIDE → LONG CONVERSION FOR DB STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _panels_to_long(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert dict of wide panels to long format for DB upsert."""
    frames = []
    for series_id, panel in panels.items():
        if panel.empty:
            continue
        for ccy in panel.columns:
            s = panel[ccy].dropna()
            if s.empty:
                continue
            df = pd.DataFrame({
                "currency":  ccy,
                "series_id": f"{ccy}_{series_id}",
                "time":      s.index,
                "value":     s.values,
            })
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DB UPSERT
# ═══════════════════════════════════════════════════════════════════════════════

def _upsert(df: pd.DataFrame, conn, schema: str) -> None:
    if df.empty:
        return

    now = datetime.now(timezone.utc)
    rows = [
        (r.currency, r.series_id, r.time, r.value, now)
        for r in df.itertuples(index=False)
    ]

    sql = f"""
        INSERT INTO {schema}.{TARGET_TABLE} (
            currency, series_id, time, value, updated_at
        )
        VALUES %s
        ON CONFLICT (currency, series_id, time)
        DO UPDATE SET
            value      = EXCLUDED.value,
            updated_at = EXCLUDED.updated_at
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()

    logger.info("Upsert complete: %s.%s rows=%d", schema, TARGET_TABLE, len(rows))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_store_tot_signals_v2(
    conn,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Compute ToT signals and store to database.

    Parameters
    ----------
    conn : psycopg2 connection
        Open connection to the database.
    schema : str
        Database schema (default "macro").

    Returns
    -------
    dict of scored panels (for inspection / composite integration).
    """
    panels = compute_tot_signals_v2(conn, schema)

    if not panels:
        logger.warning("No ToT signals produced — nothing to store")
        return panels

    long_df = _panels_to_long(panels)

    try:
        _upsert(long_df, conn, schema)
        logger.info("✓ ToT signals v2 stored: %d rows", len(long_df))
    except psycopg2.errors.UndefinedTable:
        logger.error("Table %s.%s does not exist — run setup SQL first.", schema, TARGET_TABLE)
        raise

    return panels