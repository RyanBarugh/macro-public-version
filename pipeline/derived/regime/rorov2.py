"""
derived/regime/roro_v2.py
=========================

RORO v2: 7-Bucket Risk-On/Risk-Off Regime Detection
====================================================

Upgraded from v1's 6-indicator weighted composite to a 7-bucket
architecture designed around the *types of deterioration* that
precede equity drawdowns and carry trade unwinds.

Bucket Architecture:
    1. Credit Trend       (15%) — spread ROC, not levels (levels live in FC layer)
    2. Breadth             (15%) — SPX % above 200d/50d MA, NH-NL
    3. Sector Rotation     (10%) — cyclicals vs defensives ratio
    4. FC Momentum         (10%) — FC composite score rate of change
    5. Cross-Asset Damage  (15%) — count of major assets below 50d MA
    6. Equity Trend        (15%) — SPX momentum + distance from 200d MA
    7. Vol Structure       (20%) — VIX level + term structure + term structure ROC

Design Principles:
    - FC layer = structural backdrop (tight/loose, tightening/easing)
    - RORO v2 = is risk appetite actively deteriorating NOW across dimensions?
    - No double-counting: credit levels in FC, credit *changes* in RORO
    - All z-scores direction-aligned: NEGATIVE = risk-off (deterioration)
    - Expanding-window z-scores with 2yr burn-in to avoid lookahead bias

Reads from:  macro.series_data (raw inputs), macro.financial_conditions (FC scores)
Writes to:   macro.roro_v2 (daily output)

Usage:
    from pipeline.derived.regime.roro_v2 import compute_roro_v2
    roro = compute_roro_v2(data, fc_scores)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

import logging
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SERIES CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── Existing series (already in your pipeline) ─────────────────────────────
EXISTING_SERIES = {
    # Volatility
    'vix':              'roro_vix_daily',            # VIXCLS via FRED
    'vix3m':            'roro_vix3m_daily',          # VIX3M via EODHD

    # Credit (used for ROC, not levels)
    'hy_spread':        'usd_hy_spread_oas',         # BAMLH0A0HYM2
    'ig_spread':        'usd_ig_spread_oas',         # BAMLC0A0CM

    # Equity
    'sp500':            'roro_spx_daily',            # GSPC via EODHD

    # Commodities
    'gold':             'roro_gold_daily',           # via Oanda
    'copper':           'roro_copper_daily',         # via Oanda
}

# ── New series to add to your pipeline ──────────────────────────────────────
# Add these to your series.json / ingestion config:
#
# FRED series (add to FRED provider):
#   roro_em_corp_oas     → BAMLEMCBPIOAS  (ICE BofA EM Corp Plus OAS, daily)
#
# EODHD series (add to EODHD provider):
#   roro_audjpy          → AUDJPY.FOREX   (or derive from Oanda AUDUSD * USDJPY)
#   roro_eem             → EEM.US          (iShares MSCI Emerging Markets ETF)
#   roro_hyg             → HYG.US          (iShares iBoxx HY Corporate Bond ETF)
#   roro_xlu             → XLU.US          (Utilities Select Sector SPDR)
#   roro_xlp             → XLP.US          (Consumer Staples Select Sector SPDR)
#   roro_xlv             → XLV.US          (Health Care Select Sector SPDR)
#   roro_xlk             → XLK.US          (Technology Select Sector SPDR)
#   roro_xly             → XLY.US          (Consumer Discretionary Select Sector SPDR)
#   roro_xli             → XLI.US          (Industrial Select Sector SPDR)
#
# SPX breadth (requires separate pipeline — see breadth module):
#   roro_spx_pct_above_200d  → computed from SPX constituents
#   roro_spx_pct_above_50d   → computed from SPX constituents
#   roro_spx_nh_nl_10d       → new highs minus new lows, 10d MA

NEW_SERIES = {
    # Credit (EM)
    'em_corp_oas':      'roro_em_corp_oas',          # BAMLEMCBPIOAS

    # Cross-asset
    'audjpy':           'roro_audjpy',               # AUD/JPY
    'eem':              'roro_eem',                   # EM equities ETF
    'hyg':              'roro_hyg_daily',            # HY bond ETF (already in roro.json)

    # Sector rotation
    'xlu':              'roro_xlu',                   # Utilities
    'xlp':              'roro_xlp',                   # Staples
    'xlv':              'roro_xlv',                   # Healthcare
    'xlk':              'roro_xlk',                   # Tech
    'xly':              'roro_xly',                   # Discretionary
    'xli':              'roro_xli',                   # Industrials

    # Breadth (computed — see breadth pipeline module)
    'spx_pct_above_200d':  'roro_spx_pct_above_200d',
    'spx_pct_above_50d':   'roro_spx_pct_above_50d',
    'spx_nh_nl_10d':       'roro_spx_nh_nl_10d',
}

SERIES = {**EXISTING_SERIES, **NEW_SERIES}


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

BUCKET_WEIGHTS = {
    'credit_trend':     0.15,
    'breadth':          0.15,
    'sector_rotation':  0.10,
    'fc_momentum':      0.10,
    'cross_asset':      0.15,
    'equity_trend':     0.15,
    'vol_structure':    0.20,
}

assert abs(sum(BUCKET_WEIGHTS.values()) - 1.0) < 1e-9, "Bucket weights must sum to 1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# Z-SCORE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default lookback for daily rolling z-scores
Z_WINDOW = 252          # 1-year
Z_MIN_PERIODS = 126     # 6-month minimum before producing z-scores

# Credit ROC lookback
CREDIT_ROC_WINDOW = 20  # 20-day (1-month) rate of change on spreads

# Vol term structure ROC lookback
VOL_TERM_ROC_WINDOW = 20

# FC momentum lookback
FC_MOM_WINDOW = 20      # 20-day ROC on FC composite

# Z-score clip bounds
Z_CLIP = 4.0

# Cross-asset trend lookback
TREND_MA_FAST = 50      # 50-day MA for cross-asset trend
TREND_MA_SLOW = 200     # 200-day MA for breadth + equity trend


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Schmitt trigger thresholds for regime classification
# Score is negative = risk-off, positive = risk-on
REGIME_ENTER_THRESHOLD = 0.2      # must cross ±0.40 to enter risk-on/risk-off
REGIME_EXIT_THRESHOLD = 0.2     # must cross back to ±0.15 to exit
REGIME_MIN_DAYS = 3                # minimum days before regime can switch

# EMA smoothing for composite score
EMA_SPAN = 5

# ── VCP gradient thresholds ────────────────────────────────────────────────
# Maps RORO score to position sizing multiplier for VCP/breakout trades
# Score convention: negative = risk-off
VCP_GREEN_THRESHOLD = -0.25        # above this: full size (1.0x)
VCP_YELLOW_THRESHOLD = -1.0        # between green and this: reduced (0.50x)
                                   # below this: minimal/zero (0.0–0.25x)
VCP_VIX_HARD_CUTOFF = 25.0         # VIX above this: automatic 50% reduction
VCP_VIX_FULL_STOP = 30.0           # VIX above this: no new VCP entries

# ── Market breadth prerequisites for VCP eligibility ───────────────────────
VCP_BREADTH_MIN_200D = 50.0        # % SPX above 200d MA must exceed this
VCP_BREADTH_BEAR_ZONE = 30.0       # below this: bear territory, no VCPs at all
VCP_SPX_ABOVE_200D_REQUIRED = True # SPX itself must be above 200d MA
VCP_SPX_ABOVE_50D_REQUIRED = True  # SPX itself must be above 50d MA


# ═══════════════════════════════════════════════════════════════════════════════
# Z-SCORE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_z(
    series: pd.Series,
    window: int = Z_WINDOW,
    min_periods: int = None,
    clip: float = Z_CLIP,
) -> pd.Series:
    """
    Rolling z-score with configurable window and clip bounds.
    Uses rolling (not expanding) to adapt to structural shifts.
    """
    if min_periods is None:
        min_periods = max(window // 2, Z_MIN_PERIODS)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(-clip, clip)


def expanding_z(
    series: pd.Series,
    min_periods: int = 504,
    clip: float = Z_CLIP,
) -> pd.Series:
    """
    Expanding-window z-score for backtest-safe normalization.
    Uses all data up to time t, with 2-year burn-in.
    No lookahead bias by construction.
    """
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(-clip, clip)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window, min_periods=window).mean()


def pct_above_ma(prices: pd.Series, ma_window: int) -> pd.Series:
    """Binary: 1 if price > MA, 0 otherwise."""
    ma = sma(prices, ma_window)
    return (prices > ma).astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET 1: CREDIT TREND DETERIORATION (15%)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_credit_trend(data: pd.DataFrame) -> pd.DataFrame:
    """
    Measures whether credit spreads are grinding wider (deteriorating).
    Uses rate of change on spreads, NOT levels — levels are in the FC layer.

    Direction: negative z = spreads widening = risk-off.

    Ingredients:
        - HY OAS 20d ROC, z-scored
        - IG OAS 20d ROC, z-scored
        - EM Corp OAS 20d ROC, z-scored (if available)

    Equal-weighted within bucket.
    """
    ct = pd.DataFrame(index=data.index)
    components = []

    # HY spread rate of change (widening = positive diff, invert for risk-off = negative)
    if SERIES['hy_spread'] in data.columns:
        hy = data[SERIES['hy_spread']].ffill()
        hy_roc = hy.diff(CREDIT_ROC_WINDOW)
        ct['hy_roc_z'] = rolling_z(-hy_roc, Z_WINDOW)  # negative z = widening
        components.append(ct['hy_roc_z'])

    # IG spread rate of change
    if SERIES['ig_spread'] in data.columns:
        ig = data[SERIES['ig_spread']].ffill()
        ig_roc = ig.diff(CREDIT_ROC_WINDOW)
        ct['ig_roc_z'] = rolling_z(-ig_roc, Z_WINDOW)
        components.append(ct['ig_roc_z'])

    # EM Corp OAS rate of change
    if SERIES['em_corp_oas'] in data.columns:
        em = data[SERIES['em_corp_oas']].ffill()
        em_roc = em.diff(CREDIT_ROC_WINDOW)
        ct['em_roc_z'] = rolling_z(-em_roc, Z_WINDOW)
        components.append(ct['em_roc_z'])
    else:
        logger.warning("EM Corp OAS not available — credit trend bucket degraded")

    if components:
        ct['_credit_trend'] = pd.concat(components, axis=1).mean(axis=1)
    else:
        ct['_credit_trend'] = np.nan
        logger.error("No credit series available — credit trend bucket is NaN")

    return ct


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET 2: BREADTH DETERIORATION (15%)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_breadth(data: pd.DataFrame) -> pd.DataFrame:
    """
    Market breadth — how broad is participation in the equity rally/selloff?
    Breadth divergence (indices at highs, breadth narrowing) is the classic
    early warning for the drawdowns that kill VCP setups.

    Direction: negative z = breadth deteriorating = risk-off.

    Ingredients:
        - % SPX above 200d MA, z-scored
        - % SPX above 50d MA, z-scored
        - New highs minus new lows 10d MA, z-scored

    Falls back gracefully if breadth data not yet in pipeline.
    """
    br = pd.DataFrame(index=data.index)
    components = []

    if SERIES['spx_pct_above_200d'] in data.columns:
        pct200 = data[SERIES['spx_pct_above_200d']]
        # Raw % already has natural bounds (0-100). Z-score captures deviation
        # from recent norm — a drop from 70% to 40% is the signal.
        br['pct_above_200d_z'] = rolling_z(pct200, Z_WINDOW)
        components.append(br['pct_above_200d_z'])

    if SERIES['spx_pct_above_50d'] in data.columns:
        pct50 = data[SERIES['spx_pct_above_50d']]
        br['pct_above_50d_z'] = rolling_z(pct50, Z_WINDOW)
        components.append(br['pct_above_50d_z'])

    if SERIES['spx_nh_nl_10d'] in data.columns:
        nhnl = data[SERIES['spx_nh_nl_10d']]
        br['nh_nl_z'] = rolling_z(nhnl, Z_WINDOW)
        components.append(br['nh_nl_z'])

    if components:
        br['_breadth'] = pd.concat(components, axis=1).mean(axis=1)
    else:
        # Fallback: derive a crude breadth proxy from SPX momentum dispersion
        # This is a placeholder until the full breadth pipeline is built
        logger.warning(
            "Breadth data not available — using SPX multi-timeframe "
            "momentum proxy. Build the breadth pipeline for production."
        )
        if SERIES['sp500'] in data.columns:
            spx = data[SERIES['sp500']]
            # Crude proxy: average of SPX above 50d and above 200d (binary)
            above_50 = pct_above_ma(spx, 50)
            above_200 = pct_above_ma(spx, 200)
            # This produces 0, 0.5, or 1.0 — not great but better than nothing
            proxy = (above_50 + above_200) / 2
            br['breadth_proxy'] = proxy
            # Can't meaningfully z-score a binary/ternary — use raw as bucket score
            # Map: 1.0 → +1, 0.5 → 0, 0.0 → -1
            br['_breadth'] = (proxy - 0.5) * 2
        else:
            br['_breadth'] = np.nan

    return br


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET 3: SECTOR ROTATION — CYCLICALS vs DEFENSIVES (10%)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sector_rotation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclicals-to-defensives ratio captures institutional portfolio rotation
    that happens weeks before VIX reacts. Money moves from tech/discretionary
    into utilities/staples before headline indices crack.

    Ratio = (XLK + XLY + XLI) / (XLU + XLP + XLV)
    Rising ratio = cyclicals outperforming = risk-on.
    Falling ratio = defensives outperforming = risk-off.

    Direction: negative z = defensive rotation = risk-off.
    """
    sr = pd.DataFrame(index=data.index)

    cyclical_keys = ['xlk', 'xly', 'xli']
    defensive_keys = ['xlu', 'xlp', 'xlv']

    cyclicals_available = [k for k in cyclical_keys if SERIES[k] in data.columns]
    defensives_available = [k for k in defensive_keys if SERIES[k] in data.columns]

    if len(cyclicals_available) >= 2 and len(defensives_available) >= 2:
        # Use equal-weight baskets of whatever's available
        cyc = pd.concat(
            [data[SERIES[k]].ffill() for k in cyclicals_available], axis=1
        )
        # Normalize each to base 100 at first valid date for equal weighting
        first_valid = cyc.dropna(how='all').index[0]
        cyc_norm = cyc / cyc.loc[first_valid]
        cyc_avg = cyc_norm.mean(axis=1)

        dfs = pd.concat(
            [data[SERIES[k]].ffill() for k in defensives_available], axis=1
        )
        first_valid_d = dfs.dropna(how='all').index[0]
        dfs_norm = dfs / dfs.loc[first_valid_d]
        dfs_avg = dfs_norm.mean(axis=1)

        ratio = cyc_avg / dfs_avg.replace(0, np.nan)
        sr['cyc_def_ratio'] = ratio
        sr['cyc_def_ratio_z'] = rolling_z(ratio, Z_WINDOW)
        sr['_sector_rotation'] = sr['cyc_def_ratio_z']

        logger.info(
            "Sector rotation: using %d cyclicals, %d defensives",
            len(cyclicals_available), len(defensives_available)
        )
    else:
        logger.warning(
            "Sector ETF data insufficient (need ≥2 each side). "
            "Available: cyclicals=%s, defensives=%s. "
            "Falling back to copper/gold ratio as proxy.",
            cyclicals_available, defensives_available
        )
        # Fallback: copper/gold ratio is a rough cyclical/defensive proxy
        if SERIES['copper'] in data.columns and SERIES['gold'] in data.columns:
            cu = data[SERIES['copper']].ffill()
            au = data[SERIES['gold']].ffill()
            cu_au = cu / au.replace(0, np.nan)
            sr['cyc_def_ratio'] = cu_au
            sr['cyc_def_ratio_z'] = rolling_z(cu_au, Z_WINDOW)
            sr['_sector_rotation'] = sr['cyc_def_ratio_z']
        else:
            sr['_sector_rotation'] = np.nan

    return sr


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET 4: FC MOMENTUM (10%)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fc_momentum(fc_scores: pd.Series) -> pd.DataFrame:
    """
    Bridge between the FC layer and RORO: are financial conditions
    getting WORSE right now, regardless of their absolute level?

    Uses the FC composite score's 20-day rate of change, z-scored.
    FC score convention: positive = tighter conditions.
    So positive ROC = tightening = risk-off.

    Direction: negative z = conditions tightening = risk-off.
    """
    fm = pd.DataFrame(index=fc_scores.index)

    if fc_scores is not None and not fc_scores.empty:
        fc_roc = fc_scores.diff(FC_MOM_WINDOW)
        # Positive ROC = tightening = bad. Invert so negative z = risk-off.
        fm['fc_roc'] = fc_roc
        fm['fc_roc_z'] = rolling_z(-fc_roc, Z_WINDOW)
        fm['_fc_momentum'] = fm['fc_roc_z']
    else:
        logger.error("FC scores not provided — fc_momentum bucket is NaN")
        fm['_fc_momentum'] = np.nan

    return fm


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET 5: CROSS-ASSET TREND DAMAGE (15%)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cross_asset_damage(data: pd.DataFrame) -> pd.DataFrame:
    """
    How many major asset classes are below their own trend?
    When 4 of 5+ are below trend simultaneously, that's a genuine
    cross-asset breakdown — not an isolated sector wobble.

    Assets tracked:
        - S&P 500
        - Copper
        - AUD/JPY (pure risk barometer FX pair)
        - EEM (EM equities)
        - HYG (HY bonds)

    Each scored binary: above 50d MA = 1, below = 0.
    Composite = mean of available signals.
    Then z-scored for composite integration.

    Direction: negative z = more assets below trend = risk-off.
    """
    ca = pd.DataFrame(index=data.index)
    trend_signals = []

    asset_map = {
        'sp500':  SERIES['sp500'],
        'copper': SERIES['copper'],
        'audjpy': SERIES.get('audjpy'),
        'eem':    SERIES.get('eem'),
        'hyg':    SERIES.get('hyg'),
    }

    for name, sid in asset_map.items():
        if sid and sid in data.columns:
            prices = data[sid].ffill()
            above_trend = pct_above_ma(prices, TREND_MA_FAST)
            ca[f'{name}_above_50d'] = above_trend
            trend_signals.append(above_trend)
        else:
            if name in ('sp500', 'copper'):
                logger.error("Core asset %s missing — cross-asset bucket degraded", name)
            else:
                logger.info("Optional asset %s not available for cross-asset bucket", name)

    if trend_signals:
        # Raw breadth count: 0 to 1 (fraction above trend)
        ca['cross_asset_breadth'] = pd.concat(trend_signals, axis=1).mean(axis=1)
        # Z-score the breadth measure
        ca['cross_asset_breadth_z'] = rolling_z(
            ca['cross_asset_breadth'], Z_WINDOW, min_periods=60
        )
        ca['_cross_asset'] = ca['cross_asset_breadth_z']

        n_assets = len(trend_signals)
        logger.info("Cross-asset damage: tracking %d assets", n_assets)
    else:
        ca['_cross_asset'] = np.nan
        logger.error("No assets available for cross-asset bucket")

    return ca


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET 6: EQUITY TREND (15%)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_equity_trend(data: pd.DataFrame) -> pd.DataFrame:
    """
    SPX momentum + structural trend position.

    Two sub-signals, equal-weighted:
        1. SPX 63-day return z-scored (continuous momentum)
        2. SPX distance from 200d MA, z-scored (structural position)

    Direction: negative z = equity weakness = risk-off.
    """
    et = pd.DataFrame(index=data.index)

    if SERIES['sp500'] not in data.columns:
        logger.error("S&P 500 data missing — equity trend bucket is NaN")
        et['_equity_trend'] = np.nan
        return et

    spx = data[SERIES['sp500']].ffill()

    # 1. 63-day return z-score (same as v1 equity_mom_z)
    ret_63d = spx.pct_change(63)
    et['spx_mom_63d_z'] = rolling_z(ret_63d, Z_WINDOW)

    # 2. Distance from 200d MA as % deviation, z-scored
    ma200 = sma(spx, TREND_MA_SLOW)
    pct_from_200d = (spx - ma200) / ma200
    et['spx_dist_200d_z'] = rolling_z(pct_from_200d, Z_WINDOW)

    # Binary state variables for VCP filter (not part of composite score)
    et['spx_above_200d'] = (spx > ma200).astype(float)
    et['spx_above_50d'] = (spx > sma(spx, TREND_MA_FAST)).astype(float)

    et['_equity_trend'] = (et['spx_mom_63d_z'] + et['spx_dist_200d_z']) / 2

    return et


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET 7: VOL STRUCTURE (20%)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vol_structure(data: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility structure — gets the heaviest weight (20%) because it's the
    fastest-moving and most directly actionable for VCP filtering.

    Three sub-signals, weighted:
        1. VIX level z-score (40% of bucket)
        2. VIX/VIX3M ratio z-score — term structure (30% of bucket)
        3. VIX/VIX3M 20d ROC z-score — deterioration rate (30% of bucket)

    The ROC component captures the "grinding toward inversion" that
    precedes outright vol explosions — exactly the early warning signal.

    Direction: negative z = vol rising / term structure inverting = risk-off.
    """
    vs = pd.DataFrame(index=data.index)

    if SERIES['vix'] not in data.columns:
        logger.error("VIX data missing — vol structure bucket is NaN")
        vs['_vol_structure'] = np.nan
        return vs

    vix = data[SERIES['vix']].ffill()

    # 1. VIX level z-score (inverted: high VIX = negative z = risk-off)
    vs['vix_z'] = rolling_z(-vix, Z_WINDOW)

    # 2. VIX/VIX3M term structure ratio
    if SERIES['vix3m'] in data.columns:
        vix3m = data[SERIES['vix3m']].ffill()
        vix_ratio = vix / vix3m.replace(0, np.nan)
        # High ratio (inverted term structure) = risk-off = negative z
        vs['vix_term_z'] = rolling_z(-vix_ratio, Z_WINDOW)

        # 3. Term structure ROC — the "deterioration before explosion" signal
        vix_ratio_roc = vix_ratio.diff(VOL_TERM_ROC_WINDOW)
        # Rising ratio ROC = term structure deteriorating = negative z
        vs['vix_term_roc_z'] = rolling_z(-vix_ratio_roc, Z_WINDOW)
    else:
        logger.warning("VIX3M not available — vol term structure signals degraded")
        vs['vix_term_z'] = np.nan
        vs['vix_term_roc_z'] = np.nan

    # Weighted combination within bucket
    vix_wt = 0.40
    term_wt = 0.30
    term_roc_wt = 0.30

    vs['_vol_structure'] = (
        vix_wt * vs['vix_z'].fillna(0) +
        term_wt * vs['vix_term_z'].fillna(0) +
        term_roc_wt * vs['vix_term_roc_z'].fillna(0)
    )

    # Store raw VIX for VCP hard cutoffs
    vs['vix_raw'] = vix

    return vs


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE + REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_roro_composite(buckets: dict[str, pd.Series]) -> pd.Series:
    """
    Weighted sum of bucket scores.
    Positive composite = risk-on. Negative = risk-off.
    """
    composite = pd.Series(0.0, index=buckets[list(buckets.keys())[0]].index)

    for bucket_name, weight in BUCKET_WEIGHTS.items():
        if bucket_name in buckets and buckets[bucket_name] is not None:
            score = buckets[bucket_name].fillna(0)
            composite += weight * score
        else:
            logger.warning("Bucket %s missing — treated as 0", bucket_name)

    return composite


def classify_regime_schmitt(
    score: pd.Series,
    enter_threshold: float = REGIME_ENTER_THRESHOLD,
    exit_threshold: float = REGIME_EXIT_THRESHOLD,
    min_days: int = REGIME_MIN_DAYS,
) -> tuple[pd.Series, pd.Series]:
    """
    Schmitt trigger with hysteresis for regime classification.
    Positive score = risk-on, negative = risk-off.

    Returns:
        regime: Series of 'risk_on' | 'risk_off' | 'neutral'
        regime_days: consecutive days in current regime
    """
    regimes = []
    days = []
    current_regime = 'neutral'
    current_days = 0

    for val in score:
        if pd.isna(val):
            regimes.append(np.nan)
            days.append(np.nan)
            continue

        current_days += 1
        can_switch = current_days >= min_days

        if current_regime == 'neutral' and can_switch:
            if val > enter_threshold:
                current_regime = 'risk_on'
                current_days = 1
            elif val < -enter_threshold:
                current_regime = 'risk_off'
                current_days = 1
        elif current_regime == 'risk_on' and can_switch:
            if val < exit_threshold:
                current_regime = 'neutral'
                current_days = 1
        elif current_regime == 'risk_off' and can_switch:
            if val > -exit_threshold:
                current_regime = 'neutral'
                current_days = 1

        regimes.append(current_regime)
        days.append(current_days)

    return (
        pd.Series(regimes, index=score.index, name='roro_v2_regime'),
        pd.Series(days, index=score.index, name='roro_v2_regime_days', dtype='Int64'),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VCP GRADIENT FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vcp_filter(
    roro_score: pd.Series,
    vix_raw: pd.Series,
    spx_above_200d: pd.Series,
    spx_above_50d: pd.Series,
    breadth_pct_200d: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Gradient VCP/breakout filter.

    Returns a DataFrame with:
        vcp_sizing:     float 0.0–1.0 position sizing multiplier
        vcp_zone:       'green' | 'yellow' | 'red'
        vcp_eligible:   bool — meets all prerequisites
        vcp_reason:     str — why ineligible (if applicable)
    """
    vf = pd.DataFrame(index=roro_score.index)
    n = len(roro_score)

    sizing = pd.Series(1.0, index=roro_score.index)
    zone = pd.Series('green', index=roro_score.index, dtype=str)
    eligible = pd.Series(True, index=roro_score.index)
    reason = pd.Series('', index=roro_score.index, dtype=str)

    for i in range(n):
        score_val = roro_score.iloc[i]
        vix_val = vix_raw.iloc[i] if not pd.isna(vix_raw.iloc[i]) else 0.0
        above_200 = spx_above_200d.iloc[i] if not pd.isna(spx_above_200d.iloc[i]) else 1.0
        above_50 = spx_above_50d.iloc[i] if not pd.isna(spx_above_50d.iloc[i]) else 1.0

        sz = 1.0
        zn = 'green'
        elig = True
        rsn = ''

        # ── RORO composite gradient ────────────────────────────────
        if pd.isna(score_val):
            sz = 0.5
            zn = 'yellow'
            rsn = 'insufficient_data'
        elif score_val < VCP_YELLOW_THRESHOLD:
            sz = 0.0
            zn = 'red'
            rsn = 'roro_risk_off'
        elif score_val < VCP_GREEN_THRESHOLD:
            # Linear interpolation between 0.25 and 0.75
            frac = (score_val - VCP_YELLOW_THRESHOLD) / (VCP_GREEN_THRESHOLD - VCP_YELLOW_THRESHOLD)
            sz = 0.25 + frac * 0.50
            zn = 'yellow'
            rsn = 'roro_caution'
        else:
            sz = 1.0
            zn = 'green'

        # ── VIX hard cutoffs (override gradient) ────────────────────
        if vix_val >= VCP_VIX_FULL_STOP:
            sz = 0.0
            zn = 'red'
            rsn = 'vix_full_stop'
            elig = False
        elif vix_val >= VCP_VIX_HARD_CUTOFF:
            sz = min(sz, 0.50)
            if zn == 'green':
                zn = 'yellow'
            rsn = rsn or 'vix_elevated'

        # ── SPX trend prerequisites ────────────────────────────────
        if VCP_SPX_ABOVE_200D_REQUIRED and above_200 < 1.0:
            elig = False
            rsn = 'spx_below_200d'
            sz = 0.0
            zn = 'red'
        elif VCP_SPX_ABOVE_50D_REQUIRED and above_50 < 1.0:
            sz = min(sz, 0.50)
            if zn == 'green':
                zn = 'yellow'
            rsn = rsn or 'spx_below_50d'

        # ── Breadth prerequisite ───────────────────────────────────
        if breadth_pct_200d is not None and not pd.isna(breadth_pct_200d.iloc[i]):
            b = breadth_pct_200d.iloc[i]
            if b < VCP_BREADTH_BEAR_ZONE:
                elig = False
                sz = 0.0
                zn = 'red'
                rsn = 'breadth_bear_zone'
            elif b < VCP_BREADTH_MIN_200D:
                sz = min(sz, 0.50)
                if zn == 'green':
                    zn = 'yellow'
                rsn = rsn or 'breadth_weak'

        sizing.iloc[i] = sz
        zone.iloc[i] = zn
        eligible.iloc[i] = elig
        reason.iloc[i] = rsn

    vf['vcp_sizing'] = sizing
    vf['vcp_zone'] = zone
    vf['vcp_eligible'] = eligible
    vf['vcp_reason'] = reason

    return vf


# ═══════════════════════════════════════════════════════════════════════════════
# FX PAIR REGIME MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

# Maps RORO regime × Dollar regime → FX positioning guidance
# Used by the EdgeFlow scoring engine as a filter/overlay

FX_REGIME_MATRIX = {
    # (roro_regime, dollar_quadrant): { positioning guidance }
    ('risk_on', 'Q3_bear'): {
        'label': 'Risk-on + Weak USD',
        'description': 'Sweet spot for carry. Maximum pro-cyclical exposure.',
        'favor': ['AUDUSD', 'NZDUSD', 'AUDCHF', 'NZDCHF', 'AUDJPY', 'NZDJPY'],
        'avoid': ['USDJPY', 'USDCHF'],
        'sizing_mult': 1.0,
    },
    ('risk_on', 'Q1_moderate_bull'): {
        'label': 'Risk-on + Strong USD',
        'description': 'US outperformance. Carry via non-USD crosses.',
        'favor': ['AUDCHF', 'NZDCHF', 'AUDJPY', 'NZDJPY', 'CADJPY'],
        'avoid': ['EURUSD', 'GBPUSD'],
        'sizing_mult': 0.85,
    },
    ('risk_on', 'Q2_strong_bull'): {
        'label': 'Risk-on + Strong USD (yield-driven)',
        'description': 'USD dominance. Non-USD carry only.',
        'favor': ['AUDCHF', 'NZDCHF', 'AUDJPY'],
        'avoid': ['AUDUSD', 'NZDUSD', 'EURUSD', 'GBPUSD'],
        'sizing_mult': 0.70,
    },
    ('risk_off', 'Q2_strong_bull'): {
        'label': 'Risk-off + Strong USD (global panic)',
        'description': 'Left side of dollar smile. Zero carry. Long USD.',
        'favor': ['USDCHF', 'USDJPY'],  # USD is the haven here
        'avoid': ['AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY', 'AUDCHF', 'NZDCHF'],
        'sizing_mult': 0.50,
    },
    ('risk_off', 'Q4_ambiguous'): {
        'label': 'Risk-off + Weak USD (US-specific stress)',
        'description': 'Rare: US recession while others hold up. CHF/gold.',
        'favor': ['EURCHF', 'GBPCHF'],  # actually want long CHF vs everything
        'avoid': ['AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY'],
        'sizing_mult': 0.30,
    },
    ('risk_off', 'Q3_bear'): {
        'label': 'Risk-off + Weak USD',
        'description': 'Global slowdown with USD weakness. Safe havens only.',
        'favor': [],  # CHF and gold, not FX carry
        'avoid': ['AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY'],
        'sizing_mult': 0.25,
    },
    ('neutral', 'Q1_moderate_bull'): {
        'label': 'Neutral + Moderate USD',
        'description': 'Standard environment. Use macro model signals.',
        'favor': [],
        'avoid': [],
        'sizing_mult': 0.85,
    },
}

# Default for any regime combo not explicitly mapped
FX_REGIME_DEFAULT = {
    'label': 'Unmapped regime combination',
    'description': 'Use macro model signals with standard sizing.',
    'favor': [],
    'avoid': [],
    'sizing_mult': 0.75,
}


def get_fx_regime_guidance(roro_regime: str, dollar_quadrant: str) -> dict:
    """Look up FX positioning guidance for current regime combination."""
    return FX_REGIME_MATRIX.get(
        (roro_regime, dollar_quadrant),
        FX_REGIME_DEFAULT
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_roro_v2(
    data: pd.DataFrame,
    fc_scores: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute the full RORO v2 composite with all 7 buckets.

    Args:
        data:       Wide DataFrame from macro.series_data with series_id columns.
                    Must include at minimum: VIX, HY OAS, SPX.
                    Additional series improve signal quality gracefully.
        fc_scores:  FC composite score series from Layer 1 (for FC momentum bucket).
                    If None, FC momentum bucket is zero-weighted.

    Returns:
        DataFrame with all bucket scores, composite, regime, and VCP filter.
    """
    logger.info("═══════════════════════════════════════════")
    logger.info("  RORO v2 — 7-Bucket Regime Detection")
    logger.info("═══════════════════════════════════════════")

    # ── Verify minimum required data ──────────────────────────────────────
    required = [SERIES['vix'], SERIES['hy_spread'], SERIES['sp500']]
    missing = [s for s in required if s not in data.columns]
    if missing:
        raise ValueError(f"Missing required series: {missing}")

    # ── Log data availability ─────────────────────────────────────────────
    available = [k for k, v in SERIES.items() if v in data.columns]
    total = len(SERIES)
    logger.info(
        "Data availability: %d/%d series (%d missing)",
        len(available), total, total - len(available)
    )
    missing_series = [k for k, v in SERIES.items() if v not in data.columns]
    if missing_series:
        logger.info("Missing series: %s", missing_series)

    # ── Compute each bucket ──────────────────────────────────────────────
    logger.info("Computing Bucket 1: Credit Trend...")
    ct = compute_credit_trend(data)

    logger.info("Computing Bucket 2: Breadth...")
    br = compute_breadth(data)

    logger.info("Computing Bucket 3: Sector Rotation...")
    sr = compute_sector_rotation(data)

    logger.info("Computing Bucket 4: FC Momentum...")
    if fc_scores is not None:
        fm = compute_fc_momentum(fc_scores)
    else:
        logger.warning("FC scores not provided — skipping FC momentum bucket")
        fm = pd.DataFrame({'_fc_momentum': np.nan}, index=data.index)

    logger.info("Computing Bucket 5: Cross-Asset Damage...")
    ca = compute_cross_asset_damage(data)

    logger.info("Computing Bucket 6: Equity Trend...")
    et = compute_equity_trend(data)

    logger.info("Computing Bucket 7: Vol Structure...")
    vs = compute_vol_structure(data)

    # ── Composite score ──────────────────────────────────────────────────
    buckets = {
        'credit_trend':    ct['_credit_trend'],
        'breadth':         br['_breadth'],
        'sector_rotation': sr['_sector_rotation'],
        'fc_momentum':     fm['_fc_momentum'],
        'cross_asset':     ca['_cross_asset'],
        'equity_trend':    et['_equity_trend'],
        'vol_structure':   vs['_vol_structure'],
    }

    logger.info("Computing composite score...")
    composite = compute_roro_composite(buckets)
    composite_ema = ema(composite, EMA_SPAN)

    # ── Regime classification ────────────────────────────────────────────
    logger.info("Classifying regime (Schmitt trigger)...")
    regime, regime_days = classify_regime_schmitt(composite_ema)

    # ── VCP gradient filter ──────────────────────────────────────────────
    logger.info("Computing VCP filter...")
    breadth_pct_200d = (
        data[SERIES['spx_pct_above_200d']]
        if SERIES['spx_pct_above_200d'] in data.columns
        else None
    )
    vcp = compute_vcp_filter(
        roro_score=composite_ema,
        vix_raw=vs.get('vix_raw', data[SERIES['vix']]),
        spx_above_200d=et.get('spx_above_200d', pd.Series(1.0, index=data.index)),
        spx_above_50d=et.get('spx_above_50d', pd.Series(1.0, index=data.index)),
        breadth_pct_200d=breadth_pct_200d,
    )

    # ── Assemble output ──────────────────────────────────────────────────
    out = pd.DataFrame(index=data.index)
    out.index.name = 'time'

    # Bucket scores
    out['roro2_credit_trend'] = ct['_credit_trend']
    out['roro2_breadth'] = br['_breadth']
    out['roro2_sector_rotation'] = sr['_sector_rotation']
    out['roro2_fc_momentum'] = fm['_fc_momentum']
    out['roro2_cross_asset'] = ca['_cross_asset']
    out['roro2_equity_trend'] = et['_equity_trend']
    out['roro2_vol_structure'] = vs['_vol_structure']

    # Component z-scores (for dashboard decomposition)
    for col in ct.columns:
        if col.startswith('_'):
            continue
        out[f'roro2_{col}'] = ct[col]
    for col in vs.columns:
        if col.startswith('_'):
            continue
        out[f'roro2_{col}'] = vs[col]
    for col in et.columns:
        if col.startswith('_'):
            continue
        out[f'roro2_{col}'] = et[col]
    for col in sr.columns:
        if col.startswith('_'):
            continue
        out[f'roro2_{col}'] = sr[col]
    for col in ca.columns:
        if col.startswith('_'):
            continue
        out[f'roro2_{col}'] = ca[col]

    # Composite
    out['roro2_score'] = composite
    out['roro2_score_ema10'] = composite_ema

    # Regime
    out['roro2_regime'] = regime
    out['roro2_regime_days'] = regime_days

    # VCP filter
    out['vcp_sizing'] = vcp['vcp_sizing']
    out['vcp_zone'] = vcp['vcp_zone']
    out['vcp_eligible'] = vcp['vcp_eligible']
    out['vcp_reason'] = vcp['vcp_reason']

    # ── Log summary ──────────────────────────────────────────────────────
    last_valid = out.dropna(subset=['roro2_score']).index
    if len(last_valid) > 0:
        last = last_valid[-1]
        logger.info("═══════════════════════════════════════════")
        logger.info("  RORO v2 Latest State (%s)", last.strftime('%Y-%m-%d'))
        logger.info("═══════════════════════════════════════════")
        for bucket_name in BUCKET_WEIGHTS:
            col = f'roro2_{bucket_name}'
            val = out[col].loc[last]
            wt = BUCKET_WEIGHTS[bucket_name]
            contribution = wt * (val if not pd.isna(val) else 0)
            logger.info(
                "  %-20s (%.0f%%): %+.3f → contribution: %+.3f",
                bucket_name, wt * 100,
                val if not pd.isna(val) else 0,
                contribution
            )
        logger.info("  ─────────────────────────────────────")
        logger.info("  Composite:  %+.3f (EMA10: %+.3f)",
                     out['roro2_score'].loc[last], out['roro2_score_ema10'].loc[last])
        logger.info("  Regime:     %s (day %s)",
                     out['roro2_regime'].loc[last], out['roro2_regime_days'].loc[last])
        logger.info("  VCP zone:   %s (sizing: %.0f%%, eligible: %s)",
                     out['vcp_zone'].loc[last],
                     out['vcp_sizing'].loc[last] * 100,
                     out['vcp_eligible'].loc[last])

    # Regime distribution diagnostics
    regime_dist = regime.value_counts()
    logger.info("Regime distribution: %s", regime_dist.to_dict())
    transitions = (regime != regime.shift(1)).sum() - 1
    years = len(regime.dropna()) / 252
    if years > 0:
        logger.info(
            "Regime transitions: %d total (%.1f/year over %.1f years)",
            transitions, transitions / years, years
        )

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY: v1 → v2 bridge
# ═══════════════════════════════════════════════════════════════════════════════

def compute_roro_v1_columns(roro_v2: pd.DataFrame) -> pd.DataFrame:
    """
    Generate v1-compatible columns from v2 output for API backward compat.
    Maps v2 columns to the column names the existing API/frontend expects.
    """
    v1 = pd.DataFrame(index=roro_v2.index)

    # v1 used these column names in the financial_conditions output
    v1['roro_score'] = roro_v2['roro2_score']
    v1['roro_score_ema10'] = roro_v2['roro2_score_ema10']
    v1['roro_regime'] = roro_v2['roro2_regime']
    v1['roro_regime_days'] = roro_v2['roro2_regime_days']

    # v1 component z-scores (map from v2 equivalents)
    v1['vix_z'] = roro_v2.get('roro2_vix_z', np.nan)
    v1['vix_term_z'] = roro_v2.get('roro2_vix_term_z', np.nan)
    v1['equity_mom_z'] = roro_v2.get('roro2_spx_mom_63d_z', np.nan)

    # These v1 columns came from FC layer, not RORO — carry through unchanged
    # (hy_z, ig_z, copper_gold_z are now in different buckets)

    return v1