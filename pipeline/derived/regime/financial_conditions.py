"""
derived/regime/financial_conditions.py
=======================================

Computes the Financial Conditions (FC) composite score and regime.

5-Bucket Architecture:
    Bucket 1 — RATES (25%):     cost of risk-free capital across the curve
    Bucket 2 — CREDIT (30%):    cost of risky credit (strongest leading indicator)
    Bucket 3 — LIQUIDITY (25%): central bank liquidity provision
    Bucket 4 — LEVERAGE (10%):  structural vulnerability (10-13mo recession lead)
    Bucket 5 — FUNDING (10%):   short-term wholesale funding stress

Regime Classification:
    Four-state percentile + direction classifier:
        loose_easing      — pctl < 50, direction ≤ 0  → full risk, carry on
        loose_tightening  — pctl < 75, direction > 0  → early warning
        tight_tightening  — pctl ≥ 75, direction > 0  → defensive
        tight_easing      — pctl ≥ 75, direction ≤ 0  → recovery

Reads from: macro.series_data (raw inputs)
Writes to:  macro.financial_conditions (daily output)

Usage:
    from pipeline.derived.regime.financial_conditions import compute_and_store_financial_conditions
    compute_and_store_financial_conditions(db_config=db_config, schema='macro')
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

# ── Series IDs — mapped to actual pipeline series ─────────────────────────────
SERIES = {
    # ── Rates bucket ──────────────────────────────────────────────────────
    'real_yield_10y':   'roro_tips_10y',            # DFII10
    'yield_2y':         'usd_yield_2y',             # DGS2 — via FRED
    'fed_funds':        'usd_fed_funds_rate',        # DFF

    # ── Credit bucket ─────────────────────────────────────────────────────
    'hy_spread':        'usd_hy_spread_oas',         # BAMLH0A0HYM2
    'bbb_spread':       'usd_bbb_spread_oas',        # BAMLC0A4CBBB

    # ── Liquidity bucket ──────────────────────────────────────────────────
    'fed_assets':       'fc_fed_total_assets',       # WALCL
    'on_rrp':           'fc_on_rrp',                 # RRPONTSYD

    # ── Leverage bucket ───────────────────────────────────────────────────
    'nfci_leverage':    'usd_nfci_leverage',         # NFCINONFINLEVERAGE

    # ── Funding bucket ────────────────────────────────────────────────────
    'cp_3m_rate':       'usd_cp_3m_rate',            # RIFSPPFAAD90NB
    'tbill_3m':         'usd_tbill_3m',              # DTB3
}

# ── FC 5-bucket architecture ─────────────────────────────────────────────────
# Research-grounded: Arrigoni et al. (2022) shows equal weights within buckets
# match sophisticated methods at this scale. Bucket weights from GS FCI
# multiplier estimates and FCI-G research.

FC_BUCKET_WEIGHTS = {
    'rates':     0.25,
    'credit':    0.30,
    'liquidity': 0.25,
    'leverage':  0.10,
    'funding':   0.10,
}

FC_Z_WINDOW_DEFAULT = 252       # 1-year for daily signals
FC_Z_WINDOW_LIQUIDITY = 504     # 2-year for Fed BS to avoid QT stuck-at-extreme

# ── FC regime: four-state percentile + direction classifier ───────────────────
FC_PERCENTILE_WINDOW = 1260     # 5-year rolling window for percentile rank
FC_DIRECTION_LOOKBACK = 63     # 63-day (3-month) rate of change, per FCI-G cadence
FC_TIGHT_THRESHOLD = 75         # percentile above which conditions are "tight"
FC_LOOSE_THRESHOLD = 50         # percentile below which conditions are "loose"
FC_PERSISTENCE_DAYS = 5         # minimum days before regime switch commits (tightening direction)
FC_PERSISTENCE_DAYS_EASING = 2  # faster release for easing transitions (asymmetric)

# Easing transitions — these get the shorter persistence window
FC_EASING_TRANSITIONS = {
    ('tight_tightening', 'tight_easing'),
    ('tight_easing', 'loose_easing'),
    ('tight_tightening', 'loose_easing'),
    ('tight_tightening', 'loose_tightening'),
    ('loose_tightening', 'loose_easing'),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_series(engine_or_config, schema: str, series_ids: list[str]) -> pd.DataFrame:
    placeholders = ','.join(['%s'] * len(series_ids))
    sql = f"""
        SELECT series_id, time, value
        FROM {schema}.series_data
        WHERE series_id IN ({placeholders})
        ORDER BY time
    """

    if hasattr(engine_or_config, 'host'):
        conn = psycopg2.connect(
            host=engine_or_config.host,
            port=engine_or_config.port,
            dbname=engine_or_config.dbname,
            user=engine_or_config.user,
            password=engine_or_config.password,
            sslmode=engine_or_config.sslmode,
        )
        try:
            df = pd.read_sql(sql, conn, params=series_ids)
        finally:
            conn.close()
    else:
        df = pd.read_sql(sql, engine_or_config, params=series_ids)

    if df.empty:
        raise ValueError(f"No data found for series: {series_ids}")

    df['time'] = pd.to_datetime(df['time'])
    wide = df.pivot(index='time', columns='series_id', values='value')
    return wide


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


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: FINANCIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fc(data: pd.DataFrame) -> pd.DataFrame:
    """
    Financial Conditions — 5-bucket orthogonal architecture.

    Buckets:
        rates     (25%): 10Y TIPS real yield, 2Y nominal yield, fed funds
        credit    (30%): HY OAS, BBB OAS
        liquidity (25%): Fed balance sheet delta, ON RRP
        leverage  (10%): NFCI nonfinancial leverage subindex
        funding   (10%): CP-Treasury spread (modern TED)

    All z-scores direction-aligned: positive = tighter conditions.
    Equal-weight averaging within each bucket, then weighted across buckets.
    """
    fc = pd.DataFrame(index=data.index)

    # ── BUCKET 1: RATES ──────────────────────────────────────────────────
    fc['real_yield_z'] = rolling_z(data[SERIES['real_yield_10y']], FC_Z_WINDOW_DEFAULT)
    fc['yield_2y_z'] = rolling_z(data[SERIES['yield_2y']], FC_Z_WINDOW_DEFAULT)
    fc['fed_funds_z'] = rolling_z(data[SERIES['fed_funds']], FC_Z_WINDOW_DEFAULT)

    fc['_rates'] = fc[['real_yield_z', 'yield_2y_z', 'fed_funds_z']].mean(axis=1)

    # ── BUCKET 2: CREDIT ─────────────────────────────────────────────────
    fc['hy_spread_z'] = rolling_z(data[SERIES['hy_spread']], FC_Z_WINDOW_DEFAULT)
    fc['bbb_spread_z'] = rolling_z(data[SERIES['bbb_spread']], FC_Z_WINDOW_DEFAULT)

    fc['_credit'] = fc[['hy_spread_z', 'bbb_spread_z']].mean(axis=1)

    # ── BUCKET 3: LIQUIDITY ──────────────────────────────────────────────
    fed_raw = data[SERIES['fed_assets']].ffill()
    fed_delta = fed_raw.diff(5)
    fc['fed_assets_chg_z'] = rolling_z(-fed_delta, FC_Z_WINDOW_LIQUIDITY, min_periods=126)

    fc['on_rrp_z'] = rolling_z(-data[SERIES['on_rrp']], FC_Z_WINDOW_DEFAULT)

    fc['_liquidity'] = fc[['fed_assets_chg_z', 'on_rrp_z']].mean(axis=1)

    # ── BUCKET 4: LEVERAGE ───────────────────────────────────────────────
    nfci_lev = data[SERIES['nfci_leverage']].ffill()
    fc['nfci_leverage_z'] = rolling_z(nfci_lev, FC_Z_WINDOW_DEFAULT)

    fc['_leverage'] = fc['nfci_leverage_z']

    # ── BUCKET 5: FUNDING ────────────────────────────────────────────────
    cp_rate = data[SERIES['cp_3m_rate']]
    tbill = data[SERIES['tbill_3m']]
    cp_treasury_spread = cp_rate - tbill
    fc['cp_treasury_z'] = rolling_z(cp_treasury_spread, FC_Z_WINDOW_DEFAULT)

    fc['_funding'] = fc['cp_treasury_z']

    # ── COMPOSITE: weighted sum of bucket averages ────────────────────────
    fc['fc_score'] = (
        FC_BUCKET_WEIGHTS['rates']     * fc['_rates'].fillna(0) +
        FC_BUCKET_WEIGHTS['credit']    * fc['_credit'].fillna(0) +
        FC_BUCKET_WEIGHTS['liquidity'] * fc['_liquidity'].fillna(0) +
        FC_BUCKET_WEIGHTS['leverage']  * fc['_leverage'].fillna(0) +
        FC_BUCKET_WEIGHTS['funding']   * fc['_funding'].fillna(0)
    )

    fc['fc_score_ema10'] = ema(fc['fc_score'], 10)

    # Log bucket contributions
    for bucket in FC_BUCKET_WEIGHTS:
        col = f'_{bucket}'
        if col in fc.columns:
            val = fc[col].iloc[-1] if not fc[col].dropna().empty else np.nan
            logger.info("  FC bucket %-10s (%.0f%%): %.3f",
                        bucket, FC_BUCKET_WEIGHTS[bucket] * 100, val if not np.isnan(val) else 0)
    logger.info("  FC composite: %.3f (EMA10: %.3f)",
                fc['fc_score'].iloc[-1], fc['fc_score_ema10'].iloc[-1])

    return fc


# ═══════════════════════════════════════════════════════════════════════════════
# FC REGIME CLASSIFICATION (Percentile + Direction)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_fc_regime(
    fc_score: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Four-state FC regime classification using rolling percentile rank of the
    FC composite score combined with its direction of change.

    Research basis:
        - No major institution (Chicago Fed, OFR, Goldman, Fed Board) uses
          Markov-Switching operationally on its own FC index.
        - Rolling percentile thresholds adapt to structural shifts better than
          fixed thresholds.
        - FC tightening leads risk-off by 2-8 weeks.

    Four states:
        loose_easing:      pctl < 50, direction ≤ 0  → full risk, carry on
        loose_tightening:  pctl < 75, direction > 0  → early warning, tighten stops
        tight_tightening:  pctl ≥ 75, direction > 0  → defensive, cut risk 30-50%
        tight_easing:      pctl ≥ 75, direction ≤ 0  → recovery, re-engage

    Returns:
        fc_regime, fc_regime_days, fc_percentile, fc_direction
    """
    score = fc_score.copy()

    # ── Rolling percentile rank (5-year window) ──────────────────────────
    fc_percentile = score.rolling(
        FC_PERCENTILE_WINDOW,
        min_periods=252,
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    # ── Direction of change ──────────────────────────────────────────────
    fc_direction = score - score.shift(FC_DIRECTION_LOOKBACK)

    # ── Raw four-state classification ────────────────────────────────────
    def _classify_row(pctl, direction):
        if pd.isna(pctl) or pd.isna(direction):
            return 'loose_easing'
        if pctl >= FC_TIGHT_THRESHOLD:
            if direction > 0:
                return 'tight_tightening'
            else:
                return 'tight_easing'
        else:
            if direction > 0:
                return 'loose_tightening'
            else:
                return 'loose_easing'

    raw_regime = pd.Series(
        [_classify_row(p, d) for p, d in zip(fc_percentile, fc_direction)],
        index=score.index,
        dtype=str,
    )

    # ── Persistence filter (asymmetric) ──────────────────────────────────
    regime = pd.Series('loose_easing', index=score.index, dtype=str)
    regime_days = pd.Series(0, index=score.index, dtype='Int64')
    current_regime = 'loose_easing'
    current_days = 0
    pending_regime = None
    pending_count = 0

    for i in range(len(score)):
        candidate = raw_regime.iloc[i]
        current_days += 1

        if candidate != current_regime:
            if candidate == pending_regime:
                pending_count += 1
            else:
                pending_regime = candidate
                pending_count = 1

            transition = (current_regime, candidate)
            required = FC_PERSISTENCE_DAYS_EASING if transition in FC_EASING_TRANSITIONS else FC_PERSISTENCE_DAYS

            if pending_count >= required:
                current_regime = candidate
                current_days = pending_count
                pending_regime = None
                pending_count = 0
        else:
            pending_regime = None
            pending_count = 0

        regime.iloc[i] = current_regime
        regime_days.iloc[i] = current_days

    # ── Diagnostics ──────────────────────────────────────────────────────
    dist = regime.value_counts()
    logger.info("FC regime distribution: %s", dist.to_dict())
    transitions = (regime != regime.shift(1)).sum() - 1
    years = len(regime) / 252
    if years > 0:
        logger.info("FC regime transitions: %d total (%.1f/year over %.1f years)",
                    transitions, transitions / years, years)

    last_valid = fc_percentile.last_valid_index()
    if last_valid is not None:
        logger.info(
            "FC latest: percentile=%.1f, direction=%+.3f, regime=%s (day %d)",
            fc_percentile[last_valid],
            fc_direction[last_valid] if pd.notna(fc_direction[last_valid]) else 0,
            regime[last_valid],
            regime_days[last_valid],
        )

    return regime, regime_days, fc_percentile, fc_direction


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_financial_conditions(db_config, schema: str = 'macro') -> pd.DataFrame:
    logger.info("Loading input series...")
    series_ids = list(SERIES.values())

    data = _load_series(db_config, schema, series_ids)

    data = data.asfreq('B')
    data = data.ffill(limit=7)

    logger.info("Data shape: %s, date range: %s to %s",
                data.shape, data.index.min(), data.index.max())

    logger.info("Computing Financial Conditions...")
    fc = compute_fc(data)

    logger.info("Classifying FC regime (percentile + direction)...")
    fc['fc_regime'], fc['fc_regime_days'], fc['fc_percentile'], fc['fc_direction'] = \
        classify_fc_regime(fc['fc_score'])

    logger.info("Assembling output DataFrame...")
    out = pd.DataFrame(index=data.index)
    out.index.name = 'time'

    # Component z-scores
    out['real_yield_z'] = fc['real_yield_z']
    out['yield_2y_z'] = fc['yield_2y_z']
    out['fed_funds_z'] = fc['fed_funds_z']
    out['hy_spread_z'] = fc['hy_spread_z']
    out['bbb_spread_z'] = fc['bbb_spread_z']
    out['fed_assets_chg_z'] = fc['fed_assets_chg_z']
    out['on_rrp_z'] = fc['on_rrp_z']
    out['nfci_leverage_z'] = fc['nfci_leverage_z']
    out['cp_treasury_z'] = fc['cp_treasury_z']

    # Bucket scores
    out['fc_rates'] = fc['_rates']
    out['fc_credit'] = fc['_credit']
    out['fc_liquidity'] = fc['_liquidity']
    out['fc_leverage'] = fc['_leverage']
    out['fc_funding'] = fc['_funding']

    # Composite + regime
    out['fc_score'] = fc['fc_score']
    out['fc_score_ema10'] = fc['fc_score_ema10']
    out['fc_regime'] = fc['fc_regime']
    out['fc_regime_days'] = fc['fc_regime_days']
    out['fc_percentile'] = fc['fc_percentile']
    out['fc_direction'] = fc['fc_direction']

    out = out.dropna(subset=['fc_score'], how='all')

    logger.info("Output: %d rows, %s to %s", len(out), out.index.min(), out.index.max())
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# DB UPSERT
# ═══════════════════════════════════════════════════════════════════════════════

def _upsert_financial_conditions(df: pd.DataFrame, db_config, schema: str) -> None:
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
        INSERT INTO {schema}.financial_conditions (time, {col_names}, updated_at)
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
        logger.info("Upserted %d rows to %s.financial_conditions", len(rows), schema)
    except Exception:
        logger.exception("Failed to upsert financial_conditions")
        raise
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_store_financial_conditions(db_config, schema: str = 'macro') -> pd.DataFrame:
    df = compute_financial_conditions(db_config, schema)
    _upsert_financial_conditions(df, db_config, schema)
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
    logger.info("  Financial Conditions (FC Only)")
    logger.info("═══════════════════════════════════════════")

    import time as _time
    t0 = _time.time()

    df = compute_and_store_financial_conditions(db_config, cfg.db_schema)

    elapsed = _time.time() - t0
    logger.info("═══════════════════════════════════════════")
    logger.info("  Complete in %.1fs", elapsed)
    logger.info("  Latest FC regime:    %s (day %s)",
                df['fc_regime'].iloc[-1], df['fc_regime_days'].iloc[-1])
    logger.info("═══════════════════════════════════════════")