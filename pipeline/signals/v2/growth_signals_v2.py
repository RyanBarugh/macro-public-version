"""
Growth Signals v2 — JPMaQS-Aligned Block Scoring (3 factors)
==============================================================

Reads excess/change metrics from macro.growth_derived, scores them
using panel-pooled MAD z-scores (neutral="zero"), composites into
factors and a block score with re-scoring at every aggregation step.

Changes from v1:
  - Excess over trend metrics instead of raw YoY levels
  - All neutral="zero" (all metrics centered by construction)
  - Added F2 (BCI) — absolute signals, not benchmark-relative
  - 3 factors instead of 2 (labour moved to labour_signals_v2)
  - Re-scoring at every aggregation step

Scoring pipeline:
    1. Load each constituent into wide panel (dates × 8 currencies)
    2. PIT gate: pivot on estimated_release_date (drop rows without)
    3. FFill to daily business days
    4. Z-score: make_zn_scores(panel, neutral="zero")
    5. Equal-weight within factors → rescore (if >1 constituent)
    6. Equal-weight across factors → rescore (if >1 factor)
    7. → growth_score_v2 (8 currencies)

Factor structure:
    F1 (output):        gdp_excess_trend, ip_excess_trend, ip_6m_excess_trend
    F2 (bci):           bci_3m_chg, bci_qoq_chg
    F3 (consumption):   retail_excess_trend

NOTE: Labour factors (emp_excess, unemp_gap, excess_wages) are scored
      in labour_signals_v2.py — NOT here. Including them in both blocks
      would double-count labour data in the composite.

Reads from:  macro.growth_derived
Writes to:   macro.growth_signals_v2

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.growth_signals_v2 (
        currency    TEXT          NOT NULL,
        series_id   TEXT          NOT NULL,
        time        DATE          NOT NULL,
        value       FLOAT,
        updated_at  TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_growth_signals_v2_time
        ON macro.growth_signals_v2(time);
    CREATE INDEX IF NOT EXISTS idx_growth_signals_v2_currency
        ON macro.growth_signals_v2(currency);
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


# ───────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────

TARGET_TABLE = "growth_signals_v2"

# Sign: +1 = higher value is bullish for the currency.
# All growth/excess metrics: stronger = bullish → +1.
#
# Labour factors (emp_excess, unemp_gap, excess_wages) are in
# labour_signals_v2 — not included here to avoid double-counting.
FACTORS = {
    "f_output": {
        "constituents": ["gdp_excess_trend", "ip_excess_trend", "ip_6m_excess_trend"],
        "signs":        [1, 1, 1],
        "source_table": "growth_derived",
    },
    "f_bci": {
        "constituents": ["bci_3m_chg", "bci_qoq_chg"],
        "signs":        [1, 1],
        "source_table": "growth_derived",
    },
    "f_consumption": {
        "constituents": ["retail_excess_trend"],
        "signs":        [1],
        "source_table": "growth_derived",
    },
}

CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]


# ───────────────────────────────────────────────────────────────────
# DB helpers
# ───────────────────────────────────────────────────────────────────

def _load_panel(
    metric_suffix: str,
    conn,
    schema: str,
    source_table: str,
    sign: int = 1,
) -> pd.DataFrame:
    """
    Load a single derived metric into wide format (dates × currencies).

    PIT gating: pivots on estimated_release_date where available,
    falling back to observation date (time) for rows without release
    dates. This ensures data only enters the scoring panel when it
    was publicly available.
    """
    series_ids = [f"{ccy}_{metric_suffix}" for ccy in CURRENCIES]
    placeholders = ",".join(["%s"] * len(series_ids))

    sql = f"""
        SELECT currency, time, value, estimated_release_date
        FROM {schema}.{source_table}
        WHERE series_id IN ({placeholders})
        ORDER BY currency, time
    """

    df = pd.read_sql(sql, conn, params=series_ids)
    if df.empty:
        logger.warning("No data for metric suffix: %s (table: %s)", metric_suffix, source_table)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # PIT gating: strict — drop rows without estimated_release_date
    if "estimated_release_date" in df.columns:
        df["estimated_release_date"] = pd.to_datetime(
            df["estimated_release_date"], errors="coerce"
        )
        has_rd = df["estimated_release_date"].notna()
        if has_rd.sum() == 0:
            logger.warning(
                "  PIT: %s — no release dates at all, falling back to observation date",
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

    panel = df.pivot_table(
        index="pivot_date", columns="currency", values="value", aggfunc="last"
    )
    panel = panel.sort_index()

    available = [c for c in CURRENCIES if c in panel.columns]
    panel = panel[available]

    # FFill to daily business days (JPMaQS storage convention).
    # Macrosynergy forward-fills all data to daily before scoring —
    # this ensures full cross-section at every date for cs_demean.
    panel = ffill_to_daily(panel)

    if sign == -1:
        panel = panel * -1

    logger.info(
        "  Loaded %s (%s): %d dates × %d currencies, range %s to %s",
        metric_suffix, source_table, len(panel), len(panel.columns),
        panel.index[0].date(), panel.index[-1].date(),
    )
    return panel


# ───────────────────────────────────────────────────────────────────
# Scoring pipeline
# ───────────────────────────────────────────────────────────────────

def _score_factor(
    factor_name: str,
    factor_def: dict,
    conn,
    schema: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Score a single factor: load constituents → z-score → composite
    → rescore (if multi-constituent).

    All v2 constituents are excess or change metrics — neutral="zero"
    for everything. No routing needed.
    """
    source_table = factor_def["source_table"]
    constituent_zn = {}

    for suffix, sign in zip(factor_def["constituents"], factor_def["signs"]):
        panel = _load_panel(suffix, conn, schema, source_table, sign=sign)
        if panel.empty:
            logger.warning("  Skipping constituent %s — no data", suffix)
            continue

        # All v2 metrics are excess/change — already centered around zero
        zn = make_zn_scores(panel, neutral="zero", pan_weight=0.8)
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


def compute_growth_signals_v2(
    conn,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Full growth block v2 scoring pipeline.

    Returns dict of all scored panels:
        "zn_{suffix}"       — z-scored constituents (8 currencies)
        "{factor_name}"     — factor scores (rescored if multi-constituent)
        "growth_score_v2"   — final block composite (rescored if multi-factor)
    """
    logger.info("═══ Growth Signals v2 (3-factor) ═══")

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
        logger.error("No valid factors — cannot compute growth block v2 score")
        return all_panels

    # Block composite: equal-weight across factors → rescore if multi-factor
    if len(factor_scores) == 1:
        block_score = list(factor_scores.values())[0]
    else:
        block_composite = linear_composite(factor_scores)
        block_score = rescore(block_composite)

    all_panels["growth_score_v2"] = block_score

    logger.info(
        "═══ Growth block v2 score: mean|z|=%.3f, range [%.2f, %.2f] ═══",
        np.nanmean(np.abs(block_score.values)),
        np.nanmin(block_score.values), np.nanmax(block_score.values),
    )

    return all_panels


# ───────────────────────────────────────────────────────────────────
# Wide → long conversion for DB storage
# ───────────────────────────────────────────────────────────────────

def _panels_to_long(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert dict of wide panels to long format for DB upsert.
    """
    frames = []
    for key, panel in panels.items():
        for ccy in panel.columns:
            s = panel[ccy].dropna()
            if s.empty:
                continue
            frames.append(pd.DataFrame({
                "currency":  ccy,
                "series_id": f"{ccy}_{key}",
                "time":      s.index,
                "value":     s.values.round(4),
            }))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["time"] = pd.to_datetime(result["time"]).dt.date
    return result


# ───────────────────────────────────────────────────────────────────
# Upsert
# ───────────────────────────────────────────────────────────────────

def _upsert(df: pd.DataFrame, conn, schema: str) -> None:
    if df.empty:
        return

    now = datetime.now(timezone.utc)
    cols = ["currency", "series_id", "time", "value"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]

    logger.info("Upserting %d rows into %s.%s", len(rows), schema, TARGET_TABLE)

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

    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
        logger.info("Upsert complete: %s.%s rows=%d", schema, TARGET_TABLE, len(rows))
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        logger.error(
            "Table %s.%s does not exist — run setup SQL first.",
            schema, TARGET_TABLE,
        )
        raise
    except Exception:
        conn.rollback()
        logger.exception("Upsert failed: %s.%s", schema, TARGET_TABLE)
        raise


# ───────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────

def compute_and_store_growth_signals_v2(
    conn,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Compute growth block v2 signals and write to DB.

    Returns the dict of all scored panels (useful for backtesting
    without needing to re-read from DB).
    """
    try:
        panels = compute_growth_signals_v2(conn, schema)

        if panels:
            long_df = _panels_to_long(panels)
            _upsert(long_df, conn, schema)
            logger.info(
                "✓ Growth signals v2 complete: %d panels, %d total rows",
                len(panels), len(long_df),
            )

        return panels

    except Exception as e:
        logger.error("Growth signals v2 failed: %s", e, exc_info=True)
        raise