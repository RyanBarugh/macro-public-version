"""
derived/equities/vcp_derived.py
=================================

VCP (Volatility Contraction Pattern) scanner for the equity pipeline.

Reads from equity.prices + equity.rs_rankings, detects Minervini-style
VCP setups, writes to equity.vcp_setups.

Matches breadth_derived.py / rs_derived.py pattern:
    compute_and_store_vcp(conn, schema) → None

Called by core.py run_derived() as a registered derived module.
Runs AFTER rs_derived (needs rs_rankings data).

VCP detection criteria (Minervini / O'Neil):
    1. Stage 2 uptrend — price > 150 SMA > 200 SMA, both rising
    2. Base depth — within 35% of 52-week high
    3. Contracting volatility — at least 2 pullbacks, each shallower
    4. Volume dry-up — recent volume below average
    5. Tight pivot — final contraction range < 15%
    6. RS filter — only scan stocks passing RS gates (optional)

Output: one row per detected VCP setup with quality score 0-100.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Stage 2 trend
SMA_FAST = 50
SMA_MID = 150
SMA_SLOW = 200
SMA_RISING_LOOKBACK = 20         # SMA must be higher than N days ago

# Base characteristics
MAX_BASE_DEPTH_PCT = 35          # max % below 52-week high
MIN_BASE_DAYS = 15               # minimum base length
MAX_BASE_DAYS = 180              # maximum base length
HIGH_LOOKBACK = 252              # 52-week high window

# Contraction detection
MIN_CONTRACTIONS = 2             # need at least 2 pullbacks
ATR_PERIOD = 14                  # ATR calculation period
ATR_CONTRACTION_RATIO = 0.75    # each contraction's ATR must be < 75% of prior

# Volume dry-up
VOLUME_DRY_RATIO = 0.80         # recent vol < 80% of 50d average
VOLUME_RECENT_DAYS = 10
VOLUME_AVG_DAYS = 50

# Pivot tightness
MAX_PIVOT_RANGE_PCT = 15         # final contraction range < 15%
PIVOT_LOOKBACK = 10              # last N days define the pivot area

# RS integration
REQUIRE_RS_PASS = False          # if True, only scan stocks passing RS gates
MIN_RS_PERCENTILE = 70           # minimum RS if filtering

# Price lookback for all computations
PRICE_LOOKBACK_DAYS = 400


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

_CREATE_VCP = """
CREATE TABLE IF NOT EXISTS {schema}.vcp_setups (
    ticker              TEXT        NOT NULL,
    time                DATE        NOT NULL,
    vcp_score           REAL,
    stage2              BOOLEAN     DEFAULT FALSE,
    base_depth_pct      REAL,
    base_days           INTEGER,
    n_contractions      INTEGER,
    atr_contraction     REAL,
    volume_ratio        REAL,
    pivot_range_pct     REAL,
    rs_percentile       REAL,
    sector              TEXT,
    price               REAL,
    sma50               REAL,
    sma150              REAL,
    sma200              REAL,
    high_52w            REAL,
    pct_from_high       REAL,
    updated_at          TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (ticker, time)
);
"""

_CREATE_IDX = """
CREATE INDEX IF NOT EXISTS idx_vcp_setups_score
    ON {schema}.vcp_setups (time, vcp_score DESC)
    WHERE vcp_score > 0;
"""


def _ensure_tables(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_VCP.format(schema=schema))
        cur.execute(_CREATE_IDX.format(schema=schema))
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_prices(conn, schema: str) -> tuple:
    """Load OHLCV data. Returns (close, high, low, volume) DataFrames."""
    sql = f"""
        SELECT ticker, time, open, high, low, close, volume
        FROM {schema}.prices
        WHERE time >= CURRENT_DATE - INTERVAL '{PRICE_LOOKBACK_DAYS} days'
          AND close IS NOT NULL
        ORDER BY time
    """
    df = pd.read_sql(sql, conn, parse_dates=['time'])
    if df.empty:
        raise RuntimeError("No price data found")

    close = df.pivot(index='time', columns='ticker', values='close').ffill(limit=5)
    high = df.pivot(index='time', columns='ticker', values='high').ffill(limit=5)
    low = df.pivot(index='time', columns='ticker', values='low').ffill(limit=5)
    volume = df.pivot(index='time', columns='ticker', values='volume').ffill(limit=5)

    logger.info("Loaded OHLCV: %d days x %d tickers", len(close), len(close.columns))
    return close, high, low, volume


def _load_rs(conn, schema: str) -> pd.DataFrame:
    """Load latest RS rankings. Returns DataFrame indexed by ticker."""
    sql = f"""
        SELECT ticker, rs_percentile, sector, passes_all
        FROM {schema}.rs_rankings
        WHERE time = (SELECT MAX(time) FROM {schema}.rs_rankings)
    """
    try:
        df = pd.read_sql(sql, conn)
        if df.empty:
            return pd.DataFrame()
        return df.set_index('ticker')
    except Exception:
        logger.warning("Could not load RS rankings — scanning full universe")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# VCP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _check_stage2(close_series: pd.Series) -> dict:
    """
    Check if stock is in Stage 2 uptrend.
    Returns dict with stage2 bool and SMA values.
    """
    if len(close_series) < SMA_SLOW + SMA_RISING_LOOKBACK:
        return {'stage2': False}

    price = close_series.iloc[-1]
    sma50 = close_series.rolling(SMA_FAST).mean().iloc[-1]
    sma150 = close_series.rolling(SMA_MID).mean().iloc[-1]
    sma200 = close_series.rolling(SMA_SLOW).mean().iloc[-1]

    # SMA200 rising
    sma200_prev = close_series.rolling(SMA_SLOW).mean().iloc[-(SMA_RISING_LOOKBACK + 1)]

    # Stage 2 criteria
    stage2 = (
        price > sma150 and
        price > sma200 and
        sma150 > sma200 and
        sma200 > sma200_prev  # 200 SMA rising
    )

    return {
        'stage2': bool(stage2),
        'price': float(price),
        'sma50': float(sma50),
        'sma150': float(sma150),
        'sma200': float(sma200),
        'sma200_rising': bool(sma200 > sma200_prev),
    }


def _find_contractions(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> dict:
    """
    Detect volatility contractions in recent price action.

    Method:
        1. Find the 52-week high
        2. Identify the base period (from high to present)
        3. Split base into segments using local swing highs
        4. Measure each segment's range (high-low as % of high)
        5. Check if ranges are contracting

    Returns dict with contraction metrics.
    """
    if len(close) < HIGH_LOOKBACK:
        return {'n_contractions': 0, 'base_days': 0}

    # 52-week high
    recent = close.iloc[-HIGH_LOOKBACK:]
    high_52w = recent.max()
    high_52w_idx = recent.idxmax()
    high_52w_pos = recent.index.get_loc(high_52w_idx)

    current_price = close.iloc[-1]
    pct_from_high = (high_52w - current_price) / high_52w * 100

    # Too far from high — not a base
    if pct_from_high > MAX_BASE_DEPTH_PCT:
        return {
            'n_contractions': 0,
            'base_days': 0,
            'high_52w': float(high_52w),
            'pct_from_high': float(pct_from_high),
            'base_depth_pct': float(pct_from_high),
        }

    # Base period: from 52w high to now
    base_start = max(0, high_52w_pos)
    base_close = close.iloc[-HIGH_LOOKBACK:].iloc[base_start:]
    base_high = high.iloc[-HIGH_LOOKBACK:].iloc[base_start:]
    base_low = low.iloc[-HIGH_LOOKBACK:].iloc[base_start:]
    base_days = len(base_close)

    if base_days < MIN_BASE_DAYS:
        return {
            'n_contractions': 0,
            'base_days': int(base_days),
            'high_52w': float(high_52w),
            'pct_from_high': float(pct_from_high),
            'base_depth_pct': float(pct_from_high),
        }

    # ── Detect contractions using rolling ATR ─────────────────────
    # Split base into equal segments and measure volatility of each
    n_segments = min(5, max(2, base_days // 20))
    seg_len = base_days // n_segments

    segment_ranges = []
    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, base_days)
        seg_h = base_high.iloc[start:end]
        seg_l = base_low.iloc[start:end]
        seg_c = base_close.iloc[start:end]

        if len(seg_c) < 5:
            continue

        # Segment range as % of segment high
        seg_range_pct = (seg_h.max() - seg_l.min()) / seg_h.max() * 100
        segment_ranges.append(float(seg_range_pct))

    # Count contractions: each segment tighter than previous
    n_contractions = 0
    if len(segment_ranges) >= 2:
        for i in range(1, len(segment_ranges)):
            if segment_ranges[i] < segment_ranges[i - 1] * ATR_CONTRACTION_RATIO:
                n_contractions += 1

    # ATR contraction ratio: latest segment / first segment
    atr_contraction = 1.0
    if len(segment_ranges) >= 2 and segment_ranges[0] > 0:
        atr_contraction = segment_ranges[-1] / segment_ranges[0]

    # Pivot range: range of last N days as % of current price
    pivot_high = base_high.iloc[-PIVOT_LOOKBACK:].max()
    pivot_low = base_low.iloc[-PIVOT_LOOKBACK:].min()
    pivot_range_pct = (pivot_high - pivot_low) / pivot_high * 100 if pivot_high > 0 else 99

    return {
        'n_contractions': int(n_contractions),
        'base_days': int(base_days),
        'base_depth_pct': float(pct_from_high),
        'high_52w': float(high_52w),
        'pct_from_high': float(pct_from_high),
        'atr_contraction': float(atr_contraction),
        'pivot_range_pct': float(pivot_range_pct),
        'segment_ranges': segment_ranges,
    }


def _check_volume(volume: pd.Series) -> float:
    """
    Volume dry-up ratio: recent average / longer average.
    Lower = more dry-up (bullish for VCP).
    """
    if len(volume) < VOLUME_AVG_DAYS:
        return 1.0

    recent_avg = volume.iloc[-VOLUME_RECENT_DAYS:].mean()
    longer_avg = volume.iloc[-VOLUME_AVG_DAYS:].mean()

    if longer_avg <= 0:
        return 1.0

    return float(recent_avg / longer_avg)


def _score_vcp(
    stage2: bool,
    n_contractions: int,
    atr_contraction: float,
    volume_ratio: float,
    pivot_range_pct: float,
    base_depth_pct: float,
    rs_percentile: float,
) -> float:
    """
    Score the VCP setup quality 0-100.

    Weighted components:
        - Stage 2 uptrend:    gate (0 if not stage 2)
        - Contractions:       25 pts (more = better, max at 3+)
        - ATR contraction:    20 pts (lower ratio = tighter)
        - Volume dry-up:      20 pts (lower ratio = drier)
        - Pivot tightness:    20 pts (smaller range = tighter)
        - RS strength:        15 pts (higher RS = stronger)
    """
    if not stage2:
        return 0.0

    score = 0.0

    # Contractions: 0/1/2/3+  → 0/12/20/25
    if n_contractions >= 3:
        score += 25
    elif n_contractions >= 2:
        score += 20
    elif n_contractions >= 1:
        score += 12

    # ATR contraction: ratio 1.0 = no contraction, 0.3 = very tight
    # Map 1.0→0, 0.3→20
    atr_pts = max(0, min(20, (1.0 - atr_contraction) / 0.7 * 20))
    score += atr_pts

    # Volume dry-up: ratio 1.0 = normal, 0.5 = very dry
    # Map 1.0→0, 0.5→20
    vol_pts = max(0, min(20, (1.0 - volume_ratio) / 0.5 * 20))
    score += vol_pts

    # Pivot tightness: 15% = 0, 3% = 20
    # Map 15→0, 3→20
    pivot_pts = max(0, min(20, (15 - pivot_range_pct) / 12 * 20))
    score += pivot_pts

    # RS: percentile 50→0, 100→15
    rs_pts = max(0, min(15, (rs_percentile - 50) / 50 * 15))
    score += rs_pts

    return round(score, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_all(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    rs: pd.DataFrame,
) -> list[dict]:
    """Scan all tickers for VCP setups. Returns list of result dicts."""

    # Determine which tickers to scan
    tickers = close.columns.tolist()

    # Optionally filter to RS-passing stocks only
    if REQUIRE_RS_PASS and not rs.empty:
        rs_passing = rs[rs['passes_all'] == True].index
        tickers = [t for t in tickers if t in rs_passing]
        logger.info("Scanning %d tickers (RS-filtered)", len(tickers))
    else:
        logger.info("Scanning %d tickers (full universe)", len(tickers))

    results = []
    stage2_count = 0
    vcp_count = 0

    for ticker in tickers:
        if ticker not in close.columns:
            continue

        c = close[ticker].dropna()
        h = high[ticker].dropna() if ticker in high.columns else c
        l = low[ticker].dropna() if ticker in low.columns else c
        v = volume[ticker].dropna() if ticker in volume.columns else pd.Series(dtype=float)

        if len(c) < SMA_SLOW + SMA_RISING_LOOKBACK:
            continue

        # Step 1: Stage 2 check
        trend = _check_stage2(c)
        if not trend['stage2']:
            # Still record non-stage2 stocks with minimal data for the table
            results.append({
                'ticker': ticker,
                'stage2': False,
                'vcp_score': 0,
                'base_depth_pct': None,
                'base_days': 0,
                'n_contractions': 0,
                'atr_contraction': None,
                'volume_ratio': None,
                'pivot_range_pct': None,
                'price': trend.get('price'),
                'sma50': trend.get('sma50'),
                'sma150': trend.get('sma150'),
                'sma200': trend.get('sma200'),
                'high_52w': None,
                'pct_from_high': None,
            })
            continue

        stage2_count += 1

        # Step 2: Find contractions
        contraction = _find_contractions(c, h, l)

        # Step 3: Volume dry-up
        vol_ratio = _check_volume(v) if len(v) >= VOLUME_AVG_DAYS else 1.0

        # Step 4: Get RS data
        rs_pct = 50.0
        sector = ''
        if not rs.empty and ticker in rs.index:
            rs_pct = float(rs.loc[ticker, 'rs_percentile'] or 50)
            sector = str(rs.loc[ticker, 'sector'] or '')

        # Step 5: Score
        score = _score_vcp(
            stage2=True,
            n_contractions=contraction.get('n_contractions', 0),
            atr_contraction=contraction.get('atr_contraction', 1.0),
            volume_ratio=vol_ratio,
            pivot_range_pct=contraction.get('pivot_range_pct', 99),
            base_depth_pct=contraction.get('base_depth_pct', 99),
            rs_percentile=rs_pct,
        )

        if score > 0:
            vcp_count += 1

        results.append({
            'ticker': ticker,
            'stage2': True,
            'vcp_score': score,
            'base_depth_pct': contraction.get('base_depth_pct'),
            'base_days': contraction.get('base_days', 0),
            'n_contractions': contraction.get('n_contractions', 0),
            'atr_contraction': contraction.get('atr_contraction'),
            'volume_ratio': vol_ratio,
            'pivot_range_pct': contraction.get('pivot_range_pct'),
            'price': trend.get('price'),
            'sma50': trend.get('sma50'),
            'sma150': trend.get('sma150'),
            'sma200': trend.get('sma200'),
            'high_52w': contraction.get('high_52w'),
            'pct_from_high': contraction.get('pct_from_high'),
            'rs_percentile': rs_pct,
            'sector': sector,
        })

    logger.info(
        "Scan complete: %d stage 2, %d with VCP score > 0 out of %d",
        stage2_count, vcp_count, len(tickers),
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITE
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(val):
    """Convert to Python primitive safe for psycopg2."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _write_setups(conn, schema: str, results: list[dict], as_of) -> None:
    """Upsert VCP scan results."""
    today = as_of.date() if hasattr(as_of, 'date') else as_of
    now = datetime.now(timezone.utc)

    # Only write stocks that are stage 2 (keep the table manageable)
    stage2_results = [r for r in results if r.get('stage2')]

    rows = []
    for r in stage2_results:
        rows.append((
            str(r['ticker']),
            today,
            _safe(r.get('vcp_score')),
            bool(r.get('stage2', False)),
            _safe(r.get('base_depth_pct')),
            int(r.get('base_days', 0)),
            int(r.get('n_contractions', 0)),
            _safe(r.get('atr_contraction')),
            _safe(r.get('volume_ratio')),
            _safe(r.get('pivot_range_pct')),
            _safe(r.get('rs_percentile')),
            str(r.get('sector', '')),
            _safe(r.get('price')),
            _safe(r.get('sma50')),
            _safe(r.get('sma150')),
            _safe(r.get('sma200')),
            _safe(r.get('high_52w')),
            _safe(r.get('pct_from_high')),
            now,
        ))

    sql = f"""
        INSERT INTO {schema}.vcp_setups
            (ticker, time, vcp_score, stage2, base_depth_pct, base_days,
             n_contractions, atr_contraction, volume_ratio, pivot_range_pct,
             rs_percentile, sector, price, sma50, sma150, sma200,
             high_52w, pct_from_high, updated_at)
        VALUES %s
        ON CONFLICT (ticker, time)
        DO UPDATE SET
            vcp_score = EXCLUDED.vcp_score,
            stage2 = EXCLUDED.stage2,
            base_depth_pct = EXCLUDED.base_depth_pct,
            base_days = EXCLUDED.base_days,
            n_contractions = EXCLUDED.n_contractions,
            atr_contraction = EXCLUDED.atr_contraction,
            volume_ratio = EXCLUDED.volume_ratio,
            pivot_range_pct = EXCLUDED.pivot_range_pct,
            rs_percentile = EXCLUDED.rs_percentile,
            sector = EXCLUDED.sector,
            price = EXCLUDED.price,
            sma50 = EXCLUDED.sma50,
            sma150 = EXCLUDED.sma150,
            sma200 = EXCLUDED.sma200,
            high_52w = EXCLUDED.high_52w,
            pct_from_high = EXCLUDED.pct_from_high,
            updated_at = EXCLUDED.updated_at
    """

    if rows:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
    logger.info("Upserted %d stage-2 stocks to %s.vcp_setups", len(rows), schema)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def _log_summary(results: list[dict]) -> None:
    """Log the top VCP setups."""
    scored = [r for r in results if r.get('vcp_score', 0) > 30]
    scored.sort(key=lambda r: r.get('vcp_score', 0), reverse=True)

    stage2_count = sum(1 for r in results if r.get('stage2'))

    logger.info("")
    logger.info("── VCP SCAN SUMMARY ────────────────────────────────────")
    logger.info("  %d in stage 2 uptrend, %d with VCP score > 30", stage2_count, len(scored))

    if not scored:
        logger.info("  No actionable VCP setups detected")
        logger.info("  (this is normal in risk-off / bear markets)")
        logger.info("")
        return

    logger.info("")
    logger.info("  %-8s %5s %5s %4s %5s %6s %5s %6s %s",
                "Ticker", "Score", "RS", "Con", "Depth", "Pivot", "Vol", "ATR-C", "Sector")
    logger.info("  " + "─" * 62)

    for r in scored[:20]:
        logger.info(
            "  %-8s %5.1f %5.1f %4d %4.1f%% %5.1f%% %5.2f %5.2f  %s",
            str(r['ticker']),
            r.get('vcp_score', 0),
            r.get('rs_percentile', 0),
            r.get('n_contractions', 0),
            r.get('base_depth_pct', 0),
            r.get('pivot_range_pct', 0),
            r.get('volume_ratio', 1),
            r.get('atr_contraction', 1),
            str(r.get('sector', '')),
        )
    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_store_vcp(conn, schema: str) -> None:
    """
    Main entry point. Matches pipeline pattern:
        compute_and_store_vcp(conn=conn, schema=db_schema)

    Steps:
        1. Ensure tables exist
        2. Load OHLCV + RS rankings
        3. Scan all tickers for VCP setups
        4. Score and write results
        5. Log summary
    """
    logger.info("Scanning for VCP setups...")

    _ensure_tables(conn, schema)

    close, high, low, volume = _load_prices(conn, schema)
    rs = _load_rs(conn, schema)

    results = _scan_all(close, high, low, volume, rs)

    as_of = close.index[-1]
    _write_setups(conn, schema, results, as_of)

    _log_summary(results)

    scored = sum(1 for r in results if r.get('vcp_score', 0) > 30)
    logger.info("VCP scan complete: %d actionable setups", scored)