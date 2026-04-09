"""
COT Signals — 3-Factor Positioning Model (z-score scored)
============================================================

Implements the COT positioning model with three factors:

    F1 — Crowding / Skew (multiplicative modifier on F2)
        AM net as % of OI   → z-scored
        Lev net as % of OI  → z-scored
        Average → crowding_z
        Applied as: modified_flow × (1 + α × crowding_modifier)
        where crowding_modifier = -sign(flow) × crowding_z
        → Dampens flow that agrees with crowding (piling in)
        → Amplifies flow that opposes crowding (unwinding)

    F2 — Flow (the active directional signal)
        F2a — Weekly spike: delta per leg → z-score
        F2b — Momentum blend: (0.6 × 4w + 0.4 × 13w raw) → z-score
        Average of F2a and F2b
        Scored per leg: AM longs, AM shorts, lev longs, lev shorts
        Short legs inverted: shorts adding = bearish

    F3 — Bearish Divergence (conditional, bearish only)
        Fires when: price near 13w highs AND AM+lev long 13w
        momentum declining AND crowded long
        Graduated bearish adjustment proportional to divergence strength
        Requires price data from macro.series_data — skipped if unavailable

    Combined: (F2 × F1 modifier) + F3 adjustment → cot_score

Sign conventions:
    +1 = higher value is bullish for the currency
    Net long AM/lev = bullish for currency in that contract
    Longs adding momentum = bullish → +1
    Shorts adding momentum = bearish → -1

Reads from:  macro.cot_derived, macro.series_data (for F3)
Writes to:   macro.cot_signals

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.cot_signals (
        currency    TEXT          NOT NULL,
        series_id   TEXT          NOT NULL,
        time        DATE          NOT NULL,
        value       FLOAT,
        updated_at  TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_cot_signals_time
        ON macro.cot_signals(time);
    CREATE INDEX IF NOT EXISTS idx_cot_signals_currency
        ON macro.cot_signals(currency);
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

SOURCE_TABLE = "cot_derived"
TARGET_TABLE = "cot_signals"

CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]

# ── F2a: Weekly spike — raw weekly delta per leg ─────────────────────────────
F2A_CONSTITUENTS = ["am_long_delta", "am_short_delta",
                    "lev_long_delta", "lev_short_delta"]
F2A_SIGNS        = [+1, -1, +1, -1]

# ── F2b: Momentum blend — 0.6 × 4w + 0.4 × 13w raw, then z-score ───────────
MOMENTUM_LEGS = {
    "am_long":   ("am_long_mom_4w",   "am_long_mom_13w",   +1),
    "am_short":  ("am_short_mom_4w",  "am_short_mom_13w",  -1),
    "lev_long":  ("lev_long_mom_4w",  "lev_long_mom_13w",  +1),
    "lev_short": ("lev_short_mom_4w", "lev_short_mom_13w", -1),
}
MOM_BLEND_4W  = 0.6
MOM_BLEND_13W = 0.4

# ── F1: Crowding / Skew — multiplicative modifier on F2 ─────────────────────
CROWDING_CONSTITUENTS = ["am_net_pct_oi", "lev_net_pct_oi"]
CROWDING_SIGNS        = [+1, +1]
CROWDING_ALPHA        = 0.15
CROWDING_MAX_MOD      = 0.50

# ── F3: Bearish Divergence ───────────────────────────────────────────────────
# Price data from prices.price_candles (Oanda daily candles).
# Maps currency → instrument name in that table.
PRICE_INSTRUMENTS = {
    "eur": "EUR_USD",
    "gbp": "GBP_USD",
    "aud": "AUD_USD",
    "cad": "USD_CAD",
    "jpy": "USD_JPY",
    "nzd": "NZD_USD",
    "chf": "USD_CHF",
}
# Inverted pairs: price rising = USD strengthening = bearish for mapped ccy
INVERTED_PRICE = {"cad", "jpy", "chf"}

F3_PRICE_LOOKBACK   = 65
F3_PRICE_THRESHOLD  = 0.90
F3_CROWDING_THRESH  = 0.5
F3_MAX_ADJUSTMENT   = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_panel(
    metric_suffix: str,
    conn,
    schema: str,
    sign: int = 1,
) -> pd.DataFrame:
    """Load a single cot_derived metric into wide format with PIT gating."""
    series_ids = [f"{ccy}_{metric_suffix}" for ccy in CURRENCIES]
    placeholders = ",".join(["%s"] * len(series_ids))

    sql = f"""
        SELECT currency, time, value, estimated_release_date
        FROM {schema}.{SOURCE_TABLE}
        WHERE series_id IN ({placeholders})
        ORDER BY currency, time
    """

    df = pd.read_sql(sql, conn, params=series_ids)
    if df.empty:
        logger.warning("No data for metric suffix: %s", metric_suffix)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "estimated_release_date" in df.columns:
        df["estimated_release_date"] = pd.to_datetime(
            df["estimated_release_date"], errors="coerce"
        )
        has_rd = df["estimated_release_date"].notna()
        if has_rd.sum() == 0:
            logger.warning("  PIT: %s — no release dates, using observation date", metric_suffix)
            df["pivot_date"] = df["time"]
        else:
            n_dropped = (~has_rd).sum()
            if n_dropped > 0:
                logger.info("  PIT: %s — %d rows missing release date, dropped", metric_suffix, n_dropped)
            df = df[has_rd].copy()
            df["pivot_date"] = df["estimated_release_date"]
    else:
        df["pivot_date"] = df["time"]

    panel = df.pivot_table(
        index="pivot_date", columns="currency", values="value", aggfunc="last"
    )
    panel = panel.sort_index()
    available = [c for c in CURRENCIES if c in panel.columns]
    panel = panel[available]
    panel = ffill_to_daily(panel)

    if sign == -1:
        panel = panel * -1

    logger.info(
        "  Loaded %s: %d dates × %d currencies, range %s to %s",
        metric_suffix, len(panel), len(panel.columns),
        panel.index[0].date(), panel.index[-1].date(),
    )
    return panel


def _load_price_data(conn, schema: str) -> pd.DataFrame:
    """Load daily close prices from prices.price_candles for F3 divergence."""
    all_instruments = list(PRICE_INSTRUMENTS.values())
    placeholders = ",".join(["%s"] * len(all_instruments))

    sql = f"""
        SELECT instrument, time, close
        FROM prices.price_candles
        WHERE instrument IN ({placeholders})
          AND granularity = 'D'
        ORDER BY instrument, time
    """

    df = pd.read_sql(sql, conn, params=all_instruments)
    if df.empty:
        logger.warning("  F3: No price data found")
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Map instrument back to currency
    inst_to_ccy = {v: k for k, v in PRICE_INSTRUMENTS.items()}
    df["currency"] = df["instrument"].map(inst_to_ccy)
    df = df.dropna(subset=["currency"])

    panel = df.pivot_table(
        index="time", columns="currency", values="close", aggfunc="last"
    )
    panel = panel.sort_index()

    # Invert pairs where price rising = bearish for the currency
    for ccy in INVERTED_PRICE:
        if ccy in panel.columns:
            panel[ccy] = -panel[ccy]

    available = [c for c in CURRENCIES if c in panel.columns]
    panel = panel[available]

    logger.info("  F3: Loaded price data for %d currencies, %d dates",
                len(panel.columns), len(panel))
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# F2 — FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def _score_f2a(conn, schema: str) -> tuple[pd.DataFrame, dict]:
    """F2a — Weekly spike: z-score of raw weekly delta per leg."""
    logger.info("  F2a — Weekly spike")
    constituent_zn = {}

    for suffix, sign in zip(F2A_CONSTITUENTS, F2A_SIGNS):
        panel = _load_panel(suffix, conn, schema, sign=sign)
        if panel.empty:
            continue
        zn = make_zn_scores(panel, neutral="zero", pan_weight=0.8)
        constituent_zn[suffix] = zn
        logger.info("    Z-scored %s: mean|z|=%.3f", suffix, np.nanmean(np.abs(zn.values)))

    if not constituent_zn:
        return pd.DataFrame(), {}

    f2a = linear_composite(constituent_zn) if len(constituent_zn) > 1 else list(constituent_zn.values())[0]
    logger.info("  F2a composite: mean|z|=%.3f", np.nanmean(np.abs(f2a.values)))
    return f2a, constituent_zn


def _score_f2b(conn, schema: str) -> tuple[pd.DataFrame, dict]:
    """F2b — Momentum blend: (0.6 × 4w + 0.4 × 13w) raw → z-score per leg."""
    logger.info("  F2b — Momentum blend (%.0f/%.0f)", MOM_BLEND_4W * 100, MOM_BLEND_13W * 100)
    constituent_zn = {}

    for leg_name, (metric_4w, metric_13w, sign) in MOMENTUM_LEGS.items():
        panel_4w = _load_panel(metric_4w, conn, schema, sign=1)
        panel_13w = _load_panel(metric_13w, conn, schema, sign=1)

        if panel_4w.empty or panel_13w.empty:
            logger.warning("    Skipping leg %s — missing data", leg_name)
            continue

        common_idx = panel_4w.index.intersection(panel_13w.index)
        common_cols = [c for c in panel_4w.columns if c in panel_13w.columns]
        if not len(common_idx) or not common_cols:
            continue

        blended = (MOM_BLEND_4W * panel_4w.loc[common_idx, common_cols]
                   + MOM_BLEND_13W * panel_13w.loc[common_idx, common_cols])

        if sign == -1:
            blended = blended * -1

        zn = make_zn_scores(blended, neutral="zero", pan_weight=0.8)
        blend_key = f"mom_blend_{leg_name}"
        constituent_zn[blend_key] = zn
        logger.info("    Z-scored %s (sign=%+d): mean|z|=%.3f", blend_key, sign, np.nanmean(np.abs(zn.values)))

    if not constituent_zn:
        return pd.DataFrame(), {}

    f2b = linear_composite(constituent_zn) if len(constituent_zn) > 1 else list(constituent_zn.values())[0]
    logger.info("  F2b composite: mean|z|=%.3f", np.nanmean(np.abs(f2b.values)))
    return f2b, constituent_zn


def _score_flow(conn, schema: str) -> tuple[pd.DataFrame, dict]:
    """F2 — Flow: average of F2a (spike) and F2b (momentum blend)."""
    logger.info("── F2 — Flow ──")
    all_zn = {}

    f2a, f2a_zn = _score_f2a(conn, schema)
    all_zn.update(f2a_zn)

    f2b, f2b_zn = _score_f2b(conn, schema)
    all_zn.update(f2b_zn)

    sub_factors = {}
    if not f2a.empty:
        sub_factors["f2a_spike"] = f2a
        all_zn["f2a_spike"] = f2a
    if not f2b.empty:
        sub_factors["f2b_momentum"] = f2b
        all_zn["f2b_momentum"] = f2b

    if not sub_factors:
        logger.error("  F2: no valid sub-factors")
        return pd.DataFrame(), all_zn

    flow = linear_composite(sub_factors) if len(sub_factors) > 1 else list(sub_factors.values())[0]
    logger.info("  F2 flow: mean|z|=%.3f, range [%.2f, %.2f]",
                np.nanmean(np.abs(flow.values)), np.nanmin(flow.values), np.nanmax(flow.values))
    return flow, all_zn


# ═══════════════════════════════════════════════════════════════════════════════
# F1 — CROWDING / SKEW (multiplicative modifier)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_crowding(conn, schema: str) -> tuple[pd.DataFrame, dict]:
    """F1 — Crowding: z-score AM + lev net % of OI → modifier (not additive)."""
    logger.info("── F1 — Crowding / Skew ──")
    constituent_zn = {}

    for suffix, sign in zip(CROWDING_CONSTITUENTS, CROWDING_SIGNS):
        panel = _load_panel(suffix, conn, schema, sign=sign)
        if panel.empty:
            continue
        zn = make_zn_scores(panel, neutral="mean", pan_weight=0.8)
        constituent_zn[suffix] = zn
        logger.info("    Z-scored %s: mean|z|=%.3f", suffix, np.nanmean(np.abs(zn.values)))

    if not constituent_zn:
        logger.warning("  F1: no data — crowding modifier disabled")
        return pd.DataFrame(), {}

    crowding_z = linear_composite(constituent_zn) if len(constituent_zn) > 1 else list(constituent_zn.values())[0]
    logger.info("  F1 crowding_z: mean|z|=%.3f, range [%.2f, %.2f]",
                np.nanmean(np.abs(crowding_z.values)), np.nanmin(crowding_z.values), np.nanmax(crowding_z.values))
    return crowding_z, constituent_zn


def _apply_crowding_modifier(flow: pd.DataFrame, crowding_z: pd.DataFrame) -> pd.DataFrame:
    """
    F1 × F2 interaction: multiplicative modifier.

    crowding_modifier = -sign(flow) × crowding_z
    multiplier = clip(1 + α × modifier)

    Crowded long + longs adding:        dampened     ✓
    Crowded long + shorts kicking in:   amplified    ✓
    Crowded short + shorts adding:      dampened     ✓
    Crowded short + longs kicking in:   amplified    ✓
    """
    common_idx = flow.index.intersection(crowding_z.index)
    common_cols = [c for c in flow.columns if c in crowding_z.columns]

    if not len(common_idx) or not common_cols:
        logger.warning("  Crowding modifier: no overlap — skipping")
        return flow

    flow_a = flow.loc[common_idx, common_cols]
    crowd_a = crowding_z.loc[common_idx, common_cols]

    modifier = -np.sign(flow_a.values) * crowd_a.values
    multiplier = np.clip(1.0 + CROWDING_ALPHA * modifier,
                         1.0 - CROWDING_MAX_MOD,
                         1.0 + CROWDING_MAX_MOD)

    result = flow.copy()
    result.loc[common_idx, common_cols] = flow_a.values * multiplier

    logger.info("  Crowding modifier (α=%.2f): mean mult=%.3f, range [%.2f, %.2f]",
                CROWDING_ALPHA, np.nanmean(multiplier), np.nanmin(multiplier), np.nanmax(multiplier))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# F3 — BEARISH DIVERGENCE (conditional, bearish only)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_divergence(conn, schema: str, crowding_z: pd.DataFrame) -> pd.DataFrame:
    """
    F3 — Bearish divergence: price near 13w highs while AM+lev long
    momentum declining AND crowded long. Graduated bearish adjustment.
    """
    logger.info("── F3 — Bearish Divergence ──")

    price_panel = _load_price_data(conn, schema)
    if price_panel.empty:
        logger.warning("  F3: no price data — disabled")
        return pd.DataFrame()

    am_long_13w = _load_panel("am_long_mom_13w", conn, schema, sign=1)
    lev_long_13w = _load_panel("lev_long_mom_13w", conn, schema, sign=1)

    if am_long_13w.empty and lev_long_13w.empty:
        logger.warning("  F3: no long momentum data — disabled")
        return pd.DataFrame()

    mom_panels = {}
    if not am_long_13w.empty:
        mom_panels["am"] = am_long_13w
    if not lev_long_13w.empty:
        mom_panels["lev"] = lev_long_13w
    long_mom = linear_composite(mom_panels) if len(mom_panels) > 1 else list(mom_panels.values())[0]

    # Align all inputs
    common_idx = price_panel.index.intersection(long_mom.index)
    if not crowding_z.empty:
        common_idx = common_idx.intersection(crowding_z.index)
    common_cols = [c for c in CURRENCIES
                   if c in price_panel.columns and c in long_mom.columns]

    if not len(common_idx) or not common_cols:
        logger.warning("  F3: insufficient overlap — disabled")
        return pd.DataFrame()

    price = price_panel.loc[common_idx, common_cols]
    mom = long_mom.loc[common_idx, common_cols]

    # Condition 1: price near 13w highs
    rolling_max = price.rolling(F3_PRICE_LOOKBACK, min_periods=1).max()
    rolling_min = price.rolling(F3_PRICE_LOOKBACK, min_periods=1).min()
    price_range = (rolling_max - rolling_min).replace(0, np.nan)
    price_pctile = (price - rolling_min) / price_range
    near_high = price_pctile >= F3_PRICE_THRESHOLD

    # Condition 2: long momentum declining (4w change in 13w momentum < 0)
    mom_declining = mom.diff(20) < 0

    # Condition 3: crowded long
    if not crowding_z.empty:
        crowd_a = crowding_z.reindex(index=common_idx, columns=common_cols)
        crowded_long = crowd_a > F3_CROWDING_THRESH
    else:
        crowded_long = pd.DataFrame(False, index=common_idx, columns=common_cols)

    # All three must hold
    active = near_high & mom_declining & crowded_long

    # Graduated: proportional to momentum decline
    mom_change = mom.diff(20)
    mom_std = mom_change.rolling(261, min_periods=63).std().replace(0, np.nan)
    strength = -(mom_change / mom_std).clip(-3, 0)  # positive when declining
    adjustment = strength.where(active, 0.0).clip(-F3_MAX_ADJUSTMENT, 0.0)

    n_active = active.sum().sum()
    logger.info("  F3: %d active cells (%.1f%%), max adj=%.2f",
                n_active, n_active / max(active.size, 1) * 100,
                adjustment.min().min() if n_active > 0 else 0)
    return adjustment


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cot_signals(
    conn,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Full COT block: (F2 × F1 modifier) + F3 → cot_score
    """
    logger.info("═══ COT Signals (3-factor) ═══")
    all_panels = {}

    # ── F2: Flow ──
    flow, flow_zn = _score_flow(conn, schema)
    all_panels.update(flow_zn)
    if flow.empty:
        logger.error("F2 flow empty — cannot compute COT score")
        return all_panels
    all_panels["f2_flow"] = flow

    # ── F1: Crowding modifier ──
    crowding_z, crowding_zn = _score_crowding(conn, schema)
    for suffix, zn in crowding_zn.items():
        all_panels[f"zn_{suffix}"] = zn

    if not crowding_z.empty:
        all_panels["crowding_z"] = crowding_z
        modified_flow = _apply_crowding_modifier(flow, crowding_z)
    else:
        modified_flow = flow
    all_panels["f2_modified"] = modified_flow

    # ── F3: Bearish Divergence ──
    try:
        f3 = _score_divergence(conn, schema, crowding_z)
    except Exception as e:
        logger.warning("  F3 failed (non-fatal): %s", e)
        f3 = pd.DataFrame()

    if not f3.empty:
        all_panels["f3_divergence"] = f3
        common_idx = modified_flow.index.intersection(f3.index)
        common_cols = [c for c in modified_flow.columns if c in f3.columns]
        cot_score = modified_flow.copy()
        if len(common_idx) and common_cols:
            cot_score.loc[common_idx, common_cols] += f3.loc[common_idx, common_cols]
    else:
        cot_score = modified_flow

    all_panels["cot_score"] = cot_score

    logger.info("═══ COT score: mean|z|=%.3f, range [%.2f, %.2f] ═══",
                np.nanmean(np.abs(cot_score.values)),
                np.nanmin(cot_score.values), np.nanmax(cot_score.values))
    return all_panels


# ═══════════════════════════════════════════════════════════════════════════════
# DB STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _panels_to_long(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        logger.error("Table %s.%s does not exist — run setup SQL first.", schema, TARGET_TABLE)
        raise
    except Exception:
        conn.rollback()
        logger.exception("Upsert failed: %s.%s", schema, TARGET_TABLE)
        raise


def compute_and_store_cot_signals(
    conn,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """Compute COT block signals and write to DB."""
    try:
        panels = compute_cot_signals(conn, schema)
        if panels:
            long_df = _panels_to_long(panels)
            _upsert(long_df, conn, schema)
            logger.info("✓ COT signals complete: %d panels, %d total rows",
                        len(panels), len(long_df))
        return panels
    except Exception as e:
        logger.error("COT signals failed: %s", e, exc_info=True)
        raise