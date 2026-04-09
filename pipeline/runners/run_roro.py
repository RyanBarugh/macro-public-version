"""
Run RoRo V2 Pipeline (standalone)
==================================

7-bucket RORO regime detection with VCP gradient filter.

Loads data from macro.series_data + macro.financial_conditions,
computes RORO v2 composite, classifies regime, writes to macro.roro_v2.

Usage:
    python -m pipeline.engine.run_roro_v2
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from ..engine.logger import configure_logging
configure_logging()

import time
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone

from ..engine.config import load_macro_config
from ..engine.secrets import get_secret
from ..engine.logger import get_logger

logger = get_logger(__name__)


def _load_wide_series(db_config, schema: str, series_ids: list[str]) -> pd.DataFrame:
    """Load series from series_data into wide format (date × series_id)."""
    placeholders = ','.join(['%s'] * len(series_ids))
    sql = f"""
        SELECT series_id, time, value
        FROM {schema}.series_data
        WHERE series_id IN ({placeholders})
        ORDER BY time
    """
    conn = psycopg2.connect(
        host=db_config.host,
        port=db_config.port,
        dbname=db_config.dbname,
        user=db_config.user,
        password=db_config.password,
        sslmode=db_config.sslmode,
    )
    try:
        df = pd.read_sql(sql, conn, params=series_ids)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    df['time'] = pd.to_datetime(df['time'])
    wide = df.pivot(index='time', columns='series_id', values='value')
    return wide


def _load_fc_scores(db_config, schema: str) -> pd.Series:
    """Load FC composite score from financial_conditions table."""
    sql = f"""
        SELECT time, fc_score
        FROM {schema}.financial_conditions
        ORDER BY time
    """
    conn = psycopg2.connect(
        host=db_config.host,
        port=db_config.port,
        dbname=db_config.dbname,
        user=db_config.user,
        password=db_config.password,
        sslmode=db_config.sslmode,
    )
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    if df.empty:
        logger.warning("No FC scores found — fc_momentum bucket will be NaN")
        return None

    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    return df['fc_score']


def _load_oanda_prices(db_config, schema: str) -> pd.DataFrame:
    """
    Load gold and copper from Oanda candles table.
    Tries common table names: oanda_candles, candles, oanda_instruments.
    Returns empty DataFrame if none found (graceful degradation).
    """
    oanda_map = {
        'XAU_USD': 'roro_gold_daily',
        'XCU_USD': 'roro_copper_daily',
    }
    instruments = list(oanda_map.keys())

    # Try common table names — prices.price_candles is the actual table
    candidate_tables = [
        ("prices.price_candles", "instrument", "close"),
        (f"{schema}.candles", "instrument", "close"),
        (f"public.candles", "instrument", "close"),
    ]

    conn = psycopg2.connect(
        host=db_config.host,
        port=db_config.port,
        dbname=db_config.dbname,
        user=db_config.user,
        password=db_config.password,
        sslmode=db_config.sslmode,
    )

    try:
        for table, inst_col, price_col in candidate_tables:
            try:
                placeholders = ','.join(['%s'] * len(instruments))
                sql = f"""
                    SELECT {inst_col} AS instrument, time, {price_col} AS close
                    FROM {table}
                    WHERE {inst_col} IN ({placeholders})
                      AND granularity = 'D'
                    ORDER BY time
                """
                df = pd.read_sql(sql, conn, params=instruments)
                if not df.empty:
                    logger.info("Loaded Oanda data from %s", table)
                    df['time'] = pd.to_datetime(df['time'])
                    df['series_id'] = df['instrument'].map(
                        lambda x: oanda_map.get(x.replace('oanda_', ''), oanda_map.get(x, x))
                    )
                    wide = df.pivot(index='time', columns='series_id', values='close')
                    logger.info("Loaded %d Oanda instruments: %s", len(wide.columns), list(wide.columns))
                    return wide
            except Exception:
                continue  # try next table

        # No table worked — try without granularity filter
        for table, inst_col, price_col in candidate_tables:
            try:
                placeholders = ','.join(['%s'] * len(instruments))
                sql = f"""
                    SELECT {inst_col} AS instrument, time, {price_col} AS close
                    FROM {table}
                    WHERE {inst_col} IN ({placeholders})
                    ORDER BY time
                """
                df = pd.read_sql(sql, conn, params=instruments)
                if not df.empty:
                    logger.info("Loaded Oanda data from %s (no granularity filter)", table)
                    df['time'] = pd.to_datetime(df['time'])
                    df['series_id'] = df['instrument'].map(
                        lambda x: oanda_map.get(x.replace('oanda_', ''), oanda_map.get(x, x))
                    )
                    wide = df.pivot(index='time', columns='series_id', values='close')
                    logger.info("Loaded %d Oanda instruments: %s", len(wide.columns), list(wide.columns))
                    return wide
            except Exception:
                continue

        logger.warning(
            "Could not find Oanda gold/copper data in any table. "
            "Cross-asset and sector rotation will use fallback signals. "
            "Tried: %s", [t[0] for t in candidate_tables]
        )
        return pd.DataFrame()

    finally:
        conn.close()


def _upsert_roro_v2(df: pd.DataFrame, db_config, schema: str) -> None:
    """Write RORO v2 output to macro.roro_v2 table."""
    now = datetime.now(timezone.utc)

    # Create table if not exists
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {schema}.roro_v2 (
        time DATE PRIMARY KEY,
        roro2_credit_trend DOUBLE PRECISION,
        roro2_breadth DOUBLE PRECISION,
        roro2_sector_rotation DOUBLE PRECISION,
        roro2_fc_momentum DOUBLE PRECISION,
        roro2_cross_asset DOUBLE PRECISION,
        roro2_equity_trend DOUBLE PRECISION,
        roro2_vol_structure DOUBLE PRECISION,
        roro2_score DOUBLE PRECISION,
        roro2_score_ema10 DOUBLE PRECISION,
        roro2_regime TEXT,
        roro2_regime_days INTEGER,
        vcp_sizing DOUBLE PRECISION,
        vcp_zone TEXT,
        vcp_eligible BOOLEAN,
        vcp_reason TEXT,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    """

    # Select columns to write
    write_cols = [
        'roro2_credit_trend', 'roro2_breadth', 'roro2_sector_rotation',
        'roro2_fc_momentum', 'roro2_cross_asset', 'roro2_equity_trend',
        'roro2_vol_structure', 'roro2_score', 'roro2_score_ema10',
        'roro2_regime', 'roro2_regime_days',
        'vcp_sizing', 'vcp_zone', 'vcp_eligible', 'vcp_reason',
    ]
    available_cols = [c for c in write_cols if c in df.columns]

    rows = []
    for time_val, row in df.iterrows():
        values = [time_val.date()]
        for col in available_cols:
            val = row[col]
            if pd.isna(val):
                values.append(None)
            elif isinstance(val, (np.bool_, bool)):
                values.append(bool(val))
            elif isinstance(val, (np.integer,)):
                values.append(int(val))
            elif isinstance(val, (np.floating,)):
                values.append(float(val))
            else:
                values.append(val)
        values.append(now)
        rows.append(tuple(values))

    col_names = ', '.join(available_cols)
    update_clauses = ', '.join([f"{c} = EXCLUDED.{c}" for c in available_cols])
    placeholders_sql = f"""
        INSERT INTO {schema}.roro_v2 (time, {col_names}, updated_at)
        VALUES %s
        ON CONFLICT (time)
        DO UPDATE SET {update_clauses}, updated_at = EXCLUDED.updated_at
    """

    conn = psycopg2.connect(
        host=db_config.host,
        port=db_config.port,
        dbname=db_config.dbname,
        user=db_config.user,
        password=db_config.password,
        sslmode=db_config.sslmode,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
                execute_values(cur, placeholders_sql, rows, page_size=500)
        logger.info("Upserted %d rows to %s.roro_v2", len(rows), schema)
    except Exception:
        logger.exception("Failed to upsert roro_v2")
        raise
    finally:
        conn.close()


def main():
    logger.info("═══════════════════════════════════════════")
    logger.info("  RoRo V2 Pipeline (7-Bucket)")
    logger.info("═══════════════════════════════════════════")

    t0 = time.time()

    cfg = load_macro_config()
    secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
    db_config = cfg.build_db_config(secret)
    schema = cfg.db_schema

    # ── Load all required series ──────────────────────────────────
    from ..derived.regime.rorov2 import compute_roro_v2, SERIES

    all_series_ids = list(SERIES.values())
    logger.info("Loading %d series from DB...", len(all_series_ids))
    data = _load_wide_series(db_config, schema, all_series_ids)

    # Merge Oanda prices (gold, copper) — only columns not already loaded
    oanda = _load_oanda_prices(db_config, schema)
    if not oanda.empty:
        new_cols = [c for c in oanda.columns if c not in data.columns]
        if new_cols:
            data = data.join(oanda[new_cols], how='outer')
            logger.info("Merged %d new Oanda columns: %s", len(new_cols), new_cols)
        else:
            logger.info("Gold/copper already in series_data — skipping Oanda merge")

    data = data.asfreq('B')
    data = data.ffill(limit=7)

    logger.info(
        "Data shape: %s, date range: %s to %s",
        data.shape, data.index.min(), data.index.max()
    )

    # ── Load FC scores for fc_momentum bucket ─────────────────────
    fc_scores = _load_fc_scores(db_config, schema)

    # ── Compute RORO v2 ──────────────────────────────────────────
    df = compute_roro_v2(data, fc_scores=fc_scores)

    # ── Write to DB ──────────────────────────────────────────────
    _upsert_roro_v2(df, db_config, schema)

    elapsed = time.time() - t0
    logger.info("═══════════════════════════════════════════")
    logger.info("  RoRo V2 pipeline complete in %.1fs", elapsed)

    if df is not None and not df.empty:
        last = df.iloc[-1]
        logger.info("  Latest (%s):", df.index[-1].strftime("%Y-%m-%d"))
        logger.info("    Score:    %+.3f (EMA10: %+.3f)",
                     last.get('roro2_score', 0),
                     last.get('roro2_score_ema10', 0))
        logger.info("    Regime:   %s (day %s)",
                     last.get('roro2_regime', '?'),
                     last.get('roro2_regime_days', '?'))
        logger.info("    VCP zone: %s (sizing: %.0f%%)",
                     last.get('vcp_zone', '?'),
                     last.get('vcp_sizing', 0) * 100)

        logger.info("  Bucket breakdown:")
        buckets = [
            ('credit_trend', 0.15), ('breadth', 0.15),
            ('sector_rotation', 0.10), ('fc_momentum', 0.10),
            ('cross_asset', 0.15), ('equity_trend', 0.15),
            ('vol_structure', 0.20),
        ]
        for name, wt in buckets:
            col = f'roro2_{name}'
            val = last.get(col, 0)
            if pd.isna(val):
                val = 0
            logger.info("    %-20s (%2.0f%%): %+.3f → %+.3f",
                         name, wt * 100, val, wt * val)

        # Transition stats
        regime = df['roro2_regime']
        transitions = (regime != regime.shift(1)).sum() - 1
        years = len(df.dropna(subset=['roro2_regime'])) / 252
        if years > 0:
            logger.info("    Transitions: %d total (%.1f/year over %.1f years)",
                         transitions, transitions / years, years)
    else:
        logger.warning("  No output produced")

    logger.info("═══════════════════════════════════════════")


if __name__ == "__main__":
    main()