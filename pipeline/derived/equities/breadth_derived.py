"""
derived/equities/breadth_derived.py
====================================

Equity market breadth metrics derived from constituent prices.

Single connection pattern: receives `conn` from the orchestrator.

Reads from:   equity.prices + equity.constituents
Writes to:    equity.breadth

Metrics computed (per date):
    pct_above_200d  — % of constituents above 200-day MA
    pct_above_50d   — % of constituents above 50-day MA
    nh_nl_10d       — New highs minus new lows (52-week), 10-day MA

Downstream consumers:
    - RORO v2 reads from equity.breadth for the breadth bucket (15%)
      and VCP eligibility gates.

Design:
    - Pure derived computation: reads raw prices, outputs analytical metrics
    - Same pattern as growth_derived.py, rates_derived.py, etc.
    - All functions receive `conn` — no own DB connections
    - No scoring, no z-scores — that belongs in the RORO v2 bucket

Usage:
    from pipeline.derived.equities.breadth_derived import compute_and_store_breadth
    compute_and_store_breadth(conn=conn, schema="equity")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from datetime import datetime, timezone

import logging
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum number of constituents with valid data before we produce a reading
MIN_VALID_CONSTITUENTS = 100


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

_CREATE_BREADTH_TABLE = """
CREATE TABLE IF NOT EXISTS {schema}.breadth (
    time                DATE NOT NULL,
    index_member        TEXT NOT NULL DEFAULT 'sp500',
    pct_above_200d      DOUBLE PRECISION,
    pct_above_50d       DOUBLE PRECISION,
    nh_nl_10d           DOUBLE PRECISION,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, index_member)
);

CREATE INDEX IF NOT EXISTS idx_breadth_index_member
    ON {schema}.breadth(index_member);
"""


def _ensure_breadth_table(conn, schema: str) -> None:
    """Create equity.breadth table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(_CREATE_BREADTH_TABLE.format(schema=schema))
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD PRICES
# ═══════════════════════════════════════════════════════════════════════════════

def _load_constituent_prices(
    conn,
    schema: str,
    index_filter: str = "sp500",
) -> pd.DataFrame:
    """
    Load daily close prices for active constituents of a given index.
    Returns wide DataFrame: date × ticker.
    """
    sql = f"""
        SELECT p.ticker, p.time, p.close
        FROM {schema}.prices p
        JOIN {schema}.constituents c ON p.ticker = c.ticker
        WHERE c.index_member = %s AND c.active = TRUE
        ORDER BY p.time
    """
    df = pd.read_sql(sql, conn, params=[index_filter])

    if df.empty:
        logger.error("No price data found for index_member=%s", index_filter)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"])
    prices = df.pivot(index="time", columns="ticker", values="close")
    prices = prices.sort_index()

    logger.info(
        "Loaded %d constituents, %d dates (%s to %s)",
        prices.shape[1], prices.shape[0],
        prices.index.min().strftime("%Y-%m-%d"),
        prices.index.max().strftime("%Y-%m-%d"),
    )

    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# BREADTH COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute breadth metrics from a wide price DataFrame.

    Returns DataFrame with columns:
        pct_above_200d  — % of constituents above 200-day MA
        pct_above_50d   — % of constituents above 50-day MA
        nh_nl_10d       — new highs minus new lows (52-week), 10-day MA
    """
    breadth = pd.DataFrame(index=prices.index)

    # ── % above 200-day MA ────────────────────────────────────────────────
    ma200 = prices.rolling(200, min_periods=200).mean()
    above_200 = (prices > ma200).astype(float)
    valid_200 = above_200.notna().sum(axis=1)
    breadth["pct_above_200d"] = (
        above_200.sum(axis=1) / valid_200.replace(0, np.nan) * 100
    )
    breadth.loc[valid_200 < MIN_VALID_CONSTITUENTS, "pct_above_200d"] = np.nan

    # ── % above 50-day MA ─────────────────────────────────────────────────
    ma50 = prices.rolling(50, min_periods=50).mean()
    above_50 = (prices > ma50).astype(float)
    valid_50 = above_50.notna().sum(axis=1)
    breadth["pct_above_50d"] = (
        above_50.sum(axis=1) / valid_50.replace(0, np.nan) * 100
    )
    breadth.loc[valid_50 < MIN_VALID_CONSTITUENTS, "pct_above_50d"] = np.nan

    # ── New highs minus new lows (52-week), 10-day MA ─────────────────────
    rolling_high = prices.rolling(252, min_periods=252).max()
    rolling_low = prices.rolling(252, min_periods=252).min()
    new_highs = (prices >= rolling_high * 0.999).sum(axis=1)
    new_lows = (prices <= rolling_low * 1.001).sum(axis=1)
    nh_nl = new_highs - new_lows
    breadth["nh_nl_10d"] = nh_nl.rolling(10, min_periods=10).mean()

    return breadth


# ═══════════════════════════════════════════════════════════════════════════════
# UPSERT
# ═══════════════════════════════════════════════════════════════════════════════

def _upsert_breadth(
    conn,
    schema: str,
    breadth: pd.DataFrame,
    index_filter: str = "sp500",
) -> int:
    """Write breadth metrics to equity.breadth table. Returns rows written."""
    now = datetime.now(timezone.utc)

    rows = []
    for date, row in breadth.iterrows():
        if row.isna().all():
            continue
        rows.append((
            date.date(),
            index_filter,
            float(row["pct_above_200d"]) if pd.notna(row["pct_above_200d"]) else None,
            float(row["pct_above_50d"]) if pd.notna(row["pct_above_50d"]) else None,
            float(row["nh_nl_10d"]) if pd.notna(row["nh_nl_10d"]) else None,
            now,
        ))

    if not rows:
        logger.warning("No breadth rows to write")
        return 0

    upsert_sql = f"""
        INSERT INTO {schema}.breadth
            (time, index_member, pct_above_200d, pct_above_50d, nh_nl_10d, updated_at)
        VALUES %s
        ON CONFLICT (time, index_member)
        DO UPDATE SET
            pct_above_200d = EXCLUDED.pct_above_200d,
            pct_above_50d = EXCLUDED.pct_above_50d,
            nh_nl_10d = EXCLUDED.nh_nl_10d,
            updated_at = EXCLUDED.updated_at
    """

    with conn.cursor() as cur:
        execute_values(cur, upsert_sql, rows, page_size=1000)
    conn.commit()

    logger.info("Upserted %d rows to %s.breadth", len(rows), schema)
    return len(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — matches macro derived pattern: func(conn=conn, schema=schema)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_store_breadth(
    conn,
    schema: str = "equity",
    index_filter: str = "sp500",
) -> None:
    """
    End-to-end: load prices → compute breadth → write to equity.breadth.

    Signature matches macro derived pattern:
        compute_and_store_breadth(conn=conn, schema="equity")
    """
    logger.info("Computing breadth for index_member=%s...", index_filter)

    _ensure_breadth_table(conn, schema)

    # ── Load prices ───────────────────────────────────────────────────────
    prices = _load_constituent_prices(conn, schema, index_filter=index_filter)
    if prices.empty:
        logger.error("No prices loaded — cannot compute breadth")
        return

    # ── Compute ───────────────────────────────────────────────────────────
    breadth = _compute_breadth(prices)

    # ── Write to equity.breadth ───────────────────────────────────────────
    rows_written = _upsert_breadth(conn, schema, breadth, index_filter=index_filter)

    # ── Log latest values ─────────────────────────────────────────────────
    valid_idx = breadth.dropna(subset=["pct_above_200d"]).index
    if len(valid_idx) > 0:
        latest = valid_idx[-1]
        logger.info(
            "Latest breadth (%s): %%>200d=%.1f%%, %%>50d=%.1f%%, NH-NL=%.1f",
            latest.strftime("%Y-%m-%d"),
            breadth["pct_above_200d"].loc[latest],
            breadth["pct_above_50d"].loc[latest],
            breadth["nh_nl_10d"].loc[latest],
        )

    logger.info(
        "✓ Breadth derived complete: %d rows written to %s.breadth",
        rows_written, schema,
    )