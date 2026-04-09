"""
derived/equities/rs_derived.py
================================

Relative strength ranking model for the equity pipeline.

Reads from equity.prices + equity.constituents, computes multi-timeframe
RS rankings with sector gating, acceleration, and volume confirmation,
writes to equity.rs_rankings + equity.rs_sector.

Matches breadth_derived.py pattern:
    compute_and_store_rs(conn, schema) → None

Called by core.py run_derived() as a registered derived module.

Strategy context:
    This feeds the VCP breakout scanner. The output answers:
    "Which stocks are in the top 5 by relative strength within
     a top-quartile sector, accelerating, with volume confirmation?"

Design decisions:
    - Percentile ranks (0-100), not raw returns — regime-independent
    - Multi-timeframe weighted: 21d(30%), 63d(25%), 126d(25%), 252d(20%)
    - Excess returns vs SPY benchmark (ticker=SPY must be in constituents)
    - Acceleration = 21d rank > 63d rank (getting stronger, not just strong)
    - Volume confirmation: up-day volume vs down-day volume ratio
    - Sector RS: median stock excess return within each GICS sector
    - Only stores latest snapshot (today's rankings) — not full history
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — all tunable parameters in one place for sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════════

# Multi-timeframe windows: (lookback_days, weight)
RS_WINDOWS = {
    '21d':  (21,  0.30),
    '63d':  (63,  0.25),
    '126d': (126, 0.25),
    '252d': (252, 0.20),
}

# Universe filters
MIN_HISTORY_DAYS = 252
MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 1_000_000  # $1M daily

# Sector filter
TOP_N_SECTORS = 4

# Stock selection
TOP_N_PER_SECTOR = 5

# Volume confirmation lookback
VOLUME_LOOKBACK = 50

# How many days of price history to load (enough for 252d lookback + buffer)
PRICE_LOOKBACK_DAYS = 400


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

_CREATE_RS_RANKINGS = """
CREATE TABLE IF NOT EXISTS {schema}.rs_rankings (
    ticker          TEXT        NOT NULL,
    time            DATE        NOT NULL,
    rs_composite    REAL,
    rs_percentile   REAL,
    rs_21d          REAL,
    rs_63d          REAL,
    rs_126d         REAL,
    rs_252d         REAL,
    acceleration    REAL,
    volume_score    REAL,
    sector          TEXT,
    sector_rs_pct   REAL,
    sector_rank     INTEGER,
    passes_sector   BOOLEAN     DEFAULT FALSE,
    passes_accel    BOOLEAN     DEFAULT FALSE,
    passes_volume   BOOLEAN     DEFAULT FALSE,
    passes_all      BOOLEAN     DEFAULT FALSE,
    is_top_n        BOOLEAN     DEFAULT FALSE,
    updated_at      TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (ticker, time)
);
"""

_CREATE_RS_SECTOR = """
CREATE TABLE IF NOT EXISTS {schema}.rs_sector (
    sector          TEXT        NOT NULL,
    time            DATE        NOT NULL,
    sector_rs       REAL,
    sector_rs_pct   REAL,
    n_stocks        INTEGER,
    top_ticker_1    TEXT,
    top_ticker_2    TEXT,
    top_ticker_3    TEXT,
    updated_at      TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (sector, time)
);
"""

_CREATE_INDEX_RS = """
CREATE INDEX IF NOT EXISTS idx_rs_rankings_passes
    ON {schema}.rs_rankings (time, passes_all)
    WHERE passes_all = TRUE;
"""

_CREATE_INDEX_TOP = """
CREATE INDEX IF NOT EXISTS idx_rs_rankings_top
    ON {schema}.rs_rankings (time, is_top_n)
    WHERE is_top_n = TRUE;
"""


def _ensure_tables(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_RS_RANKINGS.format(schema=schema))
        cur.execute(_CREATE_RS_SECTOR.format(schema=schema))
        cur.execute(_CREATE_INDEX_RS.format(schema=schema))
        cur.execute(_CREATE_INDEX_TOP.format(schema=schema))
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_prices(conn, schema: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load close prices and volumes from equity.prices.
    Returns (prices_wide, volumes_wide) with DatetimeIndex × ticker columns.
    """
    sql = f"""
        SELECT ticker, time, close, volume
        FROM {schema}.prices
        WHERE time >= CURRENT_DATE - INTERVAL '{PRICE_LOOKBACK_DAYS} days'
          AND close IS NOT NULL
        ORDER BY time
    """
    df = pd.read_sql(sql, conn, parse_dates=['time'])

    if df.empty:
        raise RuntimeError("No price data found in equity.prices")

    prices = df.pivot(index='time', columns='ticker', values='close')
    volumes = df.pivot(index='time', columns='ticker', values='volume')

    # Forward-fill gaps (weekends already excluded, this catches holidays)
    prices = prices.ffill(limit=5)
    volumes = volumes.ffill(limit=5)

    logger.info("Loaded prices: %d days × %d tickers", len(prices), len(prices.columns))
    return prices, volumes


def _load_constituents(conn, schema: str) -> pd.Series:
    """Load active constituents with sectors. Returns ticker → sector Series."""
    sql = f"""
        SELECT ticker, sector
        FROM {schema}.constituents
        WHERE active = TRUE
    """
    df = pd.read_sql(sql, conn)
    return df.set_index('ticker')['sector']


# ═══════════════════════════════════════════════════════════════════════════════
# RS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_universe(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    sectors: pd.Series,
) -> pd.Index:
    """Remove tickers that fail minimum data / liquidity / price requirements."""
    # Must have enough history
    has_history = prices.count() >= MIN_HISTORY_DAYS
    # Must be above minimum price
    above_min_price = prices.iloc[-1] >= MIN_PRICE
    # Must have sufficient dollar volume (trailing 20d avg)
    dollar_vol = (prices * volumes).iloc[-20:].mean()
    liquid = dollar_vol >= MIN_AVG_DOLLAR_VOLUME
    # Must exist in constituents
    in_constituents = prices.columns.isin(sectors.index)
    # No NaN in recent data
    recent_valid = prices.iloc[-5:].notna().all()

    mask = has_history & above_min_price & liquid & in_constituents & recent_valid
    eligible = prices.columns[mask]
    logger.info("Universe filter: %d/%d eligible", len(eligible), len(prices.columns))
    return eligible


def _compute_rs_scores(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    sectors: pd.Series,
    eligible: pd.Index,
) -> pd.DataFrame:
    """
    Core RS computation. Returns DataFrame with one row per eligible stock.
    All scores are percentile ranks (0-100).
    """
    # ── Percentile-rank each timeframe cross-sectionally ─────────
    # Pure percentile rank of returns — no benchmark needed
    ranked = {}
    for label, (lookback, _) in RS_WINDOWS.items():
        raw_ret = prices[eligible].pct_change(lookback).iloc[-1]
        ranked[label] = raw_ret.rank(pct=True) * 100

    # ── Weighted composite ───────────────────────────────────────
    composite = pd.Series(0.0, index=eligible)
    for label, (_, weight) in RS_WINDOWS.items():
        composite += ranked[label].reindex(eligible, fill_value=0) * weight

    rs_percentile = composite.rank(pct=True) * 100

    # ── Acceleration: 21d rank > 63d rank ────────────────────────
    acceleration = ranked['21d'].reindex(eligible, fill_value=0) - \
                   ranked['63d'].reindex(eligible, fill_value=0)

    # ── Volume confirmation ──────────────────────────────────────
    volume_score = _compute_volume_score(prices, volumes, eligible)

    # ── Sector RS ────────────────────────────────────────────────
    sector_rs, sector_rs_pct = _compute_sector_rs(
        prices, sectors, eligible
    )

    # ── Assemble ─────────────────────────────────────────────────
    result = pd.DataFrame({
        'ticker': eligible,
        'sector': sectors.reindex(eligible).values,
        'rs_composite': composite.values,
        'rs_percentile': rs_percentile.values,
        'rs_21d': ranked['21d'].reindex(eligible).values,
        'rs_63d': ranked['63d'].reindex(eligible).values,
        'rs_126d': ranked['126d'].reindex(eligible).values,
        'rs_252d': ranked['252d'].reindex(eligible).values,
        'acceleration': acceleration.values,
        'volume_score': volume_score.reindex(eligible).values,
        'sector_rs_pct': [sector_rs_pct.get(sectors.get(t, ''), 0) for t in eligible],
    }, index=eligible)

    # ── Filters ──────────────────────────────────────────────────
    # Sector gate: top N sectors
    n_sectors = len(sector_rs_pct)
    sector_cutoff = 100 - (TOP_N_SECTORS / n_sectors * 100) if n_sectors > 0 else 50
    result['passes_sector'] = result['sector_rs_pct'] >= sector_cutoff
    result['passes_accel'] = result['acceleration'] >= 0
    result['passes_volume'] = result['volume_score'] >= 50
    result['passes_all'] = (
        result['passes_sector'] &
        result['passes_accel'] &
        result['passes_volume']
    )

    # ── Rank within sector for passing stocks ────────────────────
    passing = result[result['passes_all']].copy()
    if not passing.empty:
        passing['sector_rank'] = passing.groupby('sector')['rs_percentile'].rank(
            ascending=False, method='min'
        ).astype(int)
        passing['is_top_n'] = passing['sector_rank'] <= TOP_N_PER_SECTOR
        result = result.join(passing[['sector_rank', 'is_top_n']], how='left')
    else:
        result['sector_rank'] = None
        result['is_top_n'] = False

    result['sector_rank'] = result['sector_rank'].fillna(0).astype(int)
    result['is_top_n'] = result['is_top_n'].fillna(False).infer_objects(copy=False)

    logger.info(
        "RS scores: %d eligible, %d pass all filters, %d top-N",
        len(result),
        result['passes_all'].sum(),
        result['is_top_n'].sum(),
    )

    return result, sector_rs, sector_rs_pct


def _compute_volume_score(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    eligible: pd.Index,
) -> pd.Series:
    """
    Up-volume vs down-volume ratio, percentile-ranked.
    Higher = more institutional accumulation on up-days.
    """
    rets = prices[eligible].pct_change()
    vols = volumes[eligible]

    recent_rets = rets.iloc[-VOLUME_LOOKBACK:]
    recent_vols = vols.iloc[-VOLUME_LOOKBACK:]

    up_mask = recent_rets > 0
    down_mask = recent_rets < 0

    avg_up_vol = (recent_vols * up_mask).sum() / up_mask.sum().clip(lower=1)
    avg_down_vol = (recent_vols * down_mask).sum() / down_mask.sum().clip(lower=1)

    ud_ratio = avg_up_vol / avg_down_vol.clip(lower=1)

    # Volume trend: is volume increasing?
    vol_trend = vols[eligible].iloc[-20:].mean() / vols[eligible].iloc[-50:].mean()

    raw_score = ud_ratio * 0.6 + vol_trend.reindex(ud_ratio.index, fill_value=1) * 0.4
    return raw_score.rank(pct=True) * 100


def _compute_sector_rs(
    prices: pd.DataFrame,
    sectors: pd.Series,
    eligible: pd.Index,
) -> tuple[dict, dict]:
    """
    Sector-level RS: median 63d excess return within each sector.
    Returns (sector_rs_raw, sector_rs_percentile) dicts.
    """
    lookback = 63
    stock_rets = prices[eligible].pct_change(lookback).iloc[-1]

    sector_map = sectors.reindex(eligible).dropna()

    sector_median = {}
    for sector in sector_map.unique():
        tickers = sector_map[sector_map == sector].index
        tickers_in_prices = tickers.intersection(stock_rets.index)
        if len(tickers_in_prices) > 0:
            sector_median[sector] = stock_rets.loc[tickers_in_prices].median()

    if not sector_median:
        return {}, {}

    median_series = pd.Series(sector_median)
    sector_pct = (median_series.rank(pct=True) * 100).to_dict()

    return sector_median, sector_pct


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITE
# ═══════════════════════════════════════════════════════════════════════════════

def _write_rankings(conn, schema: str, result: pd.DataFrame, as_of: datetime) -> None:
    """Upsert stock-level RS rankings."""
    today = as_of.date() if hasattr(as_of, 'date') else as_of
    now = datetime.now(timezone.utc)

    rows = []
    for _, r in result.iterrows():
        rows.append((
            r['ticker'],
            today,
            _safe_float(r.get('rs_composite')),
            _safe_float(r.get('rs_percentile')),
            _safe_float(r.get('rs_21d')),
            _safe_float(r.get('rs_63d')),
            _safe_float(r.get('rs_126d')),
            _safe_float(r.get('rs_252d')),
            _safe_float(r.get('acceleration')),
            _safe_float(r.get('volume_score')),
            r.get('sector', ''),
            _safe_float(r.get('sector_rs_pct')),
            int(r.get('sector_rank', 0)),
            bool(r.get('passes_sector', False)),
            bool(r.get('passes_accel', False)),
            bool(r.get('passes_volume', False)),
            bool(r.get('passes_all', False)),
            bool(r.get('is_top_n', False)),
            now,
        ))

    sql = f"""
        INSERT INTO {schema}.rs_rankings
            (ticker, time, rs_composite, rs_percentile,
             rs_21d, rs_63d, rs_126d, rs_252d,
             acceleration, volume_score,
             sector, sector_rs_pct, sector_rank,
             passes_sector, passes_accel, passes_volume,
             passes_all, is_top_n, updated_at)
        VALUES %s
        ON CONFLICT (ticker, time)
        DO UPDATE SET
            rs_composite = EXCLUDED.rs_composite,
            rs_percentile = EXCLUDED.rs_percentile,
            rs_21d = EXCLUDED.rs_21d,
            rs_63d = EXCLUDED.rs_63d,
            rs_126d = EXCLUDED.rs_126d,
            rs_252d = EXCLUDED.rs_252d,
            acceleration = EXCLUDED.acceleration,
            volume_score = EXCLUDED.volume_score,
            sector = EXCLUDED.sector,
            sector_rs_pct = EXCLUDED.sector_rs_pct,
            sector_rank = EXCLUDED.sector_rank,
            passes_sector = EXCLUDED.passes_sector,
            passes_accel = EXCLUDED.passes_accel,
            passes_volume = EXCLUDED.passes_volume,
            passes_all = EXCLUDED.passes_all,
            is_top_n = EXCLUDED.is_top_n,
            updated_at = EXCLUDED.updated_at
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    logger.info("Upserted %d rows to %s.rs_rankings", len(rows), schema)


def _write_sector_rs(
    conn, schema: str,
    sector_rs: dict, sector_rs_pct: dict,
    sectors: pd.Series, result: pd.DataFrame,
    as_of: datetime,
) -> None:
    """Upsert sector-level RS summary."""
    today = as_of.date() if hasattr(as_of, 'date') else as_of
    now = datetime.now(timezone.utc)

    rows = []
    for sector in sector_rs:
        # Find top 3 tickers in this sector
        sector_stocks = result[
            (result['sector'] == sector) & (result['passes_all'] == True)
        ].sort_values('rs_percentile', ascending=False)

        top_tickers = sector_stocks['ticker'].head(3).tolist()
        while len(top_tickers) < 3:
            top_tickers.append(None)

        # Count stocks in sector
        n_stocks = int((sectors == sector).sum())

        rows.append((
            sector,
            today,
            _safe_float(sector_rs.get(sector)),
            _safe_float(sector_rs_pct.get(sector)),
            n_stocks,
            top_tickers[0],
            top_tickers[1],
            top_tickers[2],
            now,
        ))

    sql = f"""
        INSERT INTO {schema}.rs_sector
            (sector, time, sector_rs, sector_rs_pct, n_stocks,
             top_ticker_1, top_ticker_2, top_ticker_3, updated_at)
        VALUES %s
        ON CONFLICT (sector, time)
        DO UPDATE SET
            sector_rs = EXCLUDED.sector_rs,
            sector_rs_pct = EXCLUDED.sector_rs_pct,
            n_stocks = EXCLUDED.n_stocks,
            top_ticker_1 = EXCLUDED.top_ticker_1,
            top_ticker_2 = EXCLUDED.top_ticker_2,
            top_ticker_3 = EXCLUDED.top_ticker_3,
            updated_at = EXCLUDED.updated_at
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=100)
    conn.commit()
    logger.info("Upserted %d sectors to %s.rs_sector", len(rows), schema)


def _safe_float(val) -> float | None:
    """Convert to float, handling NaN/None safely for Postgres."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def _log_summary(result: pd.DataFrame, sector_rs_pct: dict) -> None:
    """Log a human-readable summary of today's rankings."""
    top = result[result['is_top_n'] == True].sort_values('rs_percentile', ascending=False)

    if top.empty:
        logger.info("No stocks passed all filters today")
        return

    logger.info("")
    logger.info("── RS RANKINGS SUMMARY ────────────────────────────────")
    logger.info("  %d stocks pass all filters, %d are top-%d per sector",
                result['passes_all'].sum(), len(top), TOP_N_PER_SECTOR)

    # Sector summary
    sector_order = sorted(sector_rs_pct.items(), key=lambda x: x[1], reverse=True)
    logger.info("")
    logger.info("  %-28s %8s %6s", "Sector", "RS pct", "Status")
    logger.info("  " + "─" * 46)
    for sector, pct in sector_order:
        n_sectors = len(sector_rs_pct)
        cutoff = 100 - (TOP_N_SECTORS / n_sectors * 100) if n_sectors > 0 else 50
        status = "ACTIVE" if pct >= cutoff else "—"
        logger.info("  %-28s %7.1f%% %6s", sector, pct, status)

    # Top stocks per active sector
    logger.info("")
    for sector in top['sector'].unique():
        sector_stocks = top[top['sector'] == sector].head(TOP_N_PER_SECTOR)
        logger.info("  [%s]", sector)
        for _, row in sector_stocks.iterrows():
            logger.info(
                "    %-8s RS:%5.1f  Accel:%+5.1f  Vol:%5.1f  #%d",
                row['ticker'],
                row['rs_percentile'],
                row['acceleration'],
                row['volume_score'],
                row['sector_rank'],
            )
    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — called by core.py run_derived()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_store_rs(conn, schema: str) -> None:
    """
    Main entry point. Matches breadth_derived.py signature:
        compute_and_store_rs(conn=conn, schema=db_schema)

    Steps:
        1. Ensure tables exist
        2. Load prices + constituents from DB
        3. Filter universe
        4. Compute RS scores, sector RS, apply gates
        5. Write to equity.rs_rankings + equity.rs_sector
        6. Log summary
    """
    logger.info("Computing relative strength rankings...")

    # ── Tables ────────────────────────────────────────────────────
    _ensure_tables(conn, schema)

    # ── Load data ─────────────────────────────────────────────────
    prices, volumes = _load_prices(conn, schema)
    sectors = _load_constituents(conn, schema)

    if prices.empty or sectors.empty:
        logger.warning("Insufficient data for RS computation — skipping")
        return

    # ── Filter universe ───────────────────────────────────────────
    eligible = _filter_universe(prices, volumes, sectors)

    if len(eligible) < 50:
        logger.warning(
            "Only %d eligible stocks — need at least 50 for meaningful ranks",
            len(eligible),
        )
        return

    # ── Compute ───────────────────────────────────────────────────
    result, sector_rs, sector_rs_pct = _compute_rs_scores(
        prices, volumes, sectors, eligible
    )

    # ── Write ─────────────────────────────────────────────────────
    # Use the latest date in prices as the "as of" date
    as_of = prices.index[-1]

    _write_rankings(conn, schema, result, as_of)
    _write_sector_rs(conn, schema, sector_rs, sector_rs_pct, sectors, result, as_of)

    # ── Summary ───────────────────────────────────────────────────
    _log_summary(result, sector_rs_pct)

    logger.info("RS rankings complete: %d stocks scored", len(result))