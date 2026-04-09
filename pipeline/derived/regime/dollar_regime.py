"""
derived/regime/dollar_regime.py
================================

Computes the Dollar Regime: 4-Quadrant classification based on the
dollar smile framework.

2-Axis System:
    X-axis: Real yield momentum — multi-timeframe z-score on 10Y TIPS
            Rising real yields = USD bullish (tighter policy, higher carry)
    Y-axis: Risk appetite — RORO v2 composite score (EMA10)
            Risk-on = USD bearish (money flows out of safe havens)

Dollar Score:
    real_yield_axis - risk_axis
    Positive = net USD bullish, Negative = net USD bearish

4 Quadrants (dollar smile mapping):
    Q1_moderate_bull  — rising yields + risk-on   (US outperformance)
    Q2_strong_bull    — rising yields + risk-off  (flight to USD, left side of smile)
    Q3_bear           — falling yields + risk-on  (carry-friendly, weak USD, middle of smile)
    Q4_ambiguous      — falling yields + risk-off (US-specific stress, rare)

Reads from:
    - macro.series_data (real_yield_10y, broad_usd for momentum z-scores)
    - macro.roro_v2 (roro2_score_ema10 for risk axis)

Writes to: macro.dollar_regime (daily output)

Usage:
    from pipeline.derived.regime.dollar_regime import compute_and_store_dollar_regime
    compute_and_store_dollar_regime(db_config=db_config, schema='macro')
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone

import logging
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── Raw series needed (from macro.series_data) ────────────────────────────────
SERIES = {
    'real_yield_10y':   'roro_tips_10y',            # DFII10
    'broad_usd':        'roro_dxy_daily',            # DTWEXBGS
}

# ── Z-score / momentum config ────────────────────────────────────────────────
Z_WINDOW = 252
MOM_WINDOWS = (20, 63, 126)
MOM_WEIGHTS = (0.30, 0.50, 0.20)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_series(db_config, schema: str, series_ids: list[str]) -> pd.DataFrame:
    """Load raw series from macro.series_data."""
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
        raise ValueError(f"No data found for series: {series_ids}")

    df['time'] = pd.to_datetime(df['time'])
    wide = df.pivot(index='time', columns='series_id', values='value')
    return wide


def _load_roro_score(db_config, schema: str) -> pd.Series:
    """Load roro2_score_ema10 from macro.roro_v2."""
    sql = f"""
        SELECT time, roro2_score_ema10
        FROM {schema}.roro_v2
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
        raise ValueError("No data in roro_v2 table — run RORO pipeline first")

    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    return df['roro2_score_ema10']


# ═══════════════════════════════════════════════════════════════════════════════
# Z-SCORE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_z(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(window // 2, 60)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(-3, 3)


def pct_change_n(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n)


def momentum_composite(series: pd.Series, z_window: int = Z_WINDOW) -> pd.Series:
    """Multi-timeframe momentum z-score (20/63/126 day)."""
    components = []
    for lookback, weight in zip(MOM_WINDOWS, MOM_WEIGHTS):
        ret = pct_change_n(series, lookback)
        z = rolling_z(ret, z_window)
        components.append(z * weight)

    composite = components[0]
    for c in components[1:]:
        composite = composite + c
    return composite


# ═══════════════════════════════════════════════════════════════════════════════
# DOLLAR REGIME COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dollar_regime(
    data: pd.DataFrame,
    roro_score_ema10: pd.Series,
) -> pd.DataFrame:
    """
    Compute dollar quadrant from real yield momentum × risk appetite.

    Args:
        data:              Raw series (real_yield_10y, broad_usd)
        roro_score_ema10:  Smoothed RORO composite from RORO v2

    Returns:
        DataFrame with axes, dollar score, quadrant, and confidence.
    """
    dr = pd.DataFrame(index=data.index)

    # ── Axes (computed from raw series) ──────────────────────────────────
    dr['real_yield_mom_z'] = momentum_composite(
        data[SERIES['real_yield_10y']], z_window=Z_WINDOW
    )
    dr['broad_usd_mom_z'] = momentum_composite(
        data[SERIES['broad_usd']], z_window=Z_WINDOW
    )

    # Align RORO score to our index
    roro_aligned = roro_score_ema10.reindex(data.index).ffill()

    dr['real_yield_axis'] = dr['real_yield_mom_z']
    dr['risk_axis'] = roro_aligned
    dr['dollar_score'] = dr['real_yield_axis'] - dr['risk_axis']

    # ── 4-Quadrant classification ────────────────────────────────────────
    ry = dr['real_yield_axis']
    ri = dr['risk_axis']
    dr['dollar_quadrant'] = np.where(
        (ry > 0) & (ri > 0), 'Q1_moderate_bull',
        np.where(
            (ry > 0) & (ri <= 0), 'Q2_strong_bull',
            np.where(
                (ry <= 0) & (ri > 0), 'Q3_bear',
                'Q4_ambiguous'
            )
        )
    )

    dr['quadrant_confidence'] = np.sqrt(ry**2 + ri**2)

    return dr


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dollar_regime_full(db_config, schema: str = 'macro') -> pd.DataFrame:
    """
    Load all inputs and compute dollar regime.

    Reads from:
        - macro.series_data (real_yield_10y, broad_usd)
        - macro.roro_v2 (roro2_score_ema10)
    """
    logger.info("Loading raw series...")
    series_ids = list(SERIES.values())
    data = _load_series(db_config, schema, series_ids)
    data = data.asfreq('B')
    data = data.ffill(limit=7)

    logger.info("Data shape: %s, date range: %s to %s",
                data.shape, data.index.min(), data.index.max())

    logger.info("Loading RORO v2 score...")
    roro_score = _load_roro_score(db_config, schema)

    logger.info("Computing Dollar Regime...")
    dr = compute_dollar_regime(data, roro_score)

    # ── Assemble output ──────────────────────────────────────────────────
    out = pd.DataFrame(index=data.index)
    out.index.name = 'time'

    out['real_yield_axis'] = dr['real_yield_axis']
    out['real_yield_mom_z'] = dr['real_yield_mom_z']
    out['broad_usd_mom_z'] = dr['broad_usd_mom_z']
    out['risk_axis'] = dr['risk_axis']
    out['dollar_score'] = dr['dollar_score']
    out['dollar_quadrant'] = dr['dollar_quadrant']
    out['quadrant_confidence'] = dr['quadrant_confidence']

    out = out.dropna(subset=['dollar_score'], how='all')

    logger.info("Output: %d rows, %s to %s", len(out), out.index.min(), out.index.max())

    # ── Log summary ──────────────────────────────────────────────────────
    if len(out) > 0:
        latest = out.iloc[-1]
        logger.info("  Real yield axis: %+.3f", latest['real_yield_axis'])
        logger.info("  Risk axis:       %+.3f", latest['risk_axis'])
        logger.info("  Dollar score:    %+.3f", latest['dollar_score'])
        logger.info("  Quadrant:        %s (confidence: %.2f)",
                    latest['dollar_quadrant'], latest['quadrant_confidence'])

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# DB UPSERT
# ═══════════════════════════════════════════════════════════════════════════════

def _upsert_dollar_regime(df: pd.DataFrame, db_config, schema: str) -> None:
    now = datetime.now(timezone.utc)
    columns = [c for c in df.columns]
    col_names = ', '.join(columns)

    rows = []
    for time_val, row in df.iterrows():
        values = [time_val.date()]
        for col in columns:
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

    update_clauses = ', '.join([f"{c} = EXCLUDED.{c}" for c in columns])
    sql = f"""
        INSERT INTO {schema}.dollar_regime (time, {col_names}, updated_at)
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
                execute_values(cur, sql, rows, page_size=500)
        logger.info("Upserted %d rows to %s.dollar_regime", len(rows), schema)
    except Exception:
        logger.exception("Failed to upsert dollar_regime")
        raise
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_store_dollar_regime(db_config, schema: str = 'macro') -> pd.DataFrame:
    df = compute_dollar_regime_full(db_config, schema)
    _upsert_dollar_regime(df, db_config, schema)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    from ...engine.config import load_macro_config
    from ...engine.secrets import get_secret

    cfg = load_macro_config()
    secret = get_secret(cfg.secrets_manager_secret_id, cfg.region)
    db_config = cfg.build_db_config(secret)

    logger.info("═══════════════════════════════════════════")
    logger.info("  Dollar Regime")
    logger.info("═══════════════════════════════════════════")

    import time as _time
    t0 = _time.time()

    df = compute_and_store_dollar_regime(db_config, cfg.db_schema)

    elapsed = _time.time() - t0
    logger.info("═══════════════════════════════════════════")
    logger.info("  Complete in %.1fs", elapsed)
    logger.info("  Latest quadrant: %s (confidence %.2f)",
                df['dollar_quadrant'].iloc[-1], df['quadrant_confidence'].iloc[-1])
    logger.info("═══════════════════════════════════════════")