"""
Composite Signals v2 — Hierarchical Scoring Architecture
==========================================================

Combines v2 block scores into a single country-level composite,
then constructs pair signals.

Pipeline:
    1. Load block scores from v2 signal tables (or accept pre-computed)
    2. FFill all blocks to daily business days (the only ffill)
    3. Inverse-vol weight blocks → country composite → rescore
    4. RORO additive modifier on country scores (risk regime shift)
    5. Pair signal = base_score - quote_score
    6. Store country composites + pair signals

Normalisation hierarchy:
    Pass 1: make_zn_scores on each constituent (in *_signals_v2.py)
            with cs_demean=True — removes structural bias at each level
    Pass 2: rescore after each aggregation (constituent → factor → block)
            with cs_demean=True — re-normalises after averaging
    Pass 3: rescore here after combining blocks into composite
            with cs_demean=True — final hierarchical re-normalisation

No separate cross_sectional_zscore step is needed — the v2 z-score
engine applies cross-sectional demeaning at every level, which is
the correct JPMaQS approach. Applying an additional cross-sectional
z-score on top would over-normalise and wash out magnitude information.

The pair signal is the final trading signal:
    positive = bullish for the pair (buy base / sell quote)
    negative = bearish for the pair

Reads from:  macro.growth_signals_v2, macro.labour_signals_v2,
             macro.monetary_signals_v2, macro.rates_signals_v2,
             macro.tot_signals_v2, macro.cot_signals
Writes to:   macro.composite_signals_v2

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.composite_signals_v2 (
        currency    TEXT          NOT NULL,
        series_id   TEXT          NOT NULL,
        time        DATE          NOT NULL,
        value       FLOAT,
        updated_at  TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_composite_signals_v2_time
        ON macro.composite_signals_v2(time);
    CREATE INDEX IF NOT EXISTS idx_composite_signals_v2_currency
        ON macro.composite_signals_v2(currency);
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
from pipeline.signals.zn_scores_v2 import linear_composite, rescore, ffill_to_daily

logger = get_logger(__name__)


CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]

TARGET_TABLE = "composite_signals_v2"

# All 28 G10 pairs in standard FX convention (base/quote).
# Signal > 0 → buy base, sell quote.
PAIRS = [
    # Majors
    ("eur", "usd"), ("gbp", "usd"), ("aud", "usd"), ("nzd", "usd"),
    ("usd", "cad"), ("usd", "jpy"), ("usd", "chf"),
    # EUR crosses
    ("eur", "gbp"), ("eur", "jpy"), ("eur", "aud"),
    ("eur", "cad"), ("eur", "nzd"), ("eur", "chf"),
    # GBP crosses
    ("gbp", "jpy"), ("gbp", "aud"), ("gbp", "cad"),
    ("gbp", "nzd"), ("gbp", "chf"),
    # AUD crosses
    ("aud", "jpy"), ("aud", "cad"), ("aud", "nzd"), ("aud", "chf"),
    # CAD crosses
    ("cad", "jpy"), ("cad", "chf"),
    # NZD crosses
    ("nzd", "jpy"), ("nzd", "cad"), ("nzd", "chf"),
    # CHF cross
    ("chf", "jpy"),
]

# V2 block score keys and their source tables
BLOCK_MAP = {
    "growth_score_v2":    "growth_signals_v2",
    "labour_score_v2":    "labour_signals_v2",
    "monetary_score_v2":  "monetary_signals_v2",
    "rates_score_v2":     "rates_signals_v2",
    "tot_score_v2":       "tot_signals_v2",
    "cot_score":          "cot_signals",
}


# ───────────────────────────────────────────────────────────────────
# Load block scores from DB
# ───────────────────────────────────────────────────────────────────

def _load_block_score(
    block_name: str,
    source_table: str,
    conn,
    schema: str,
) -> pd.DataFrame:
    """
    Load a block score (e.g. growth_score_v2) into wide format.
    Block scores have all 8 currencies.
    """
    series_ids = [f"{ccy}_{block_name}" for ccy in CURRENCIES]
    placeholders = ",".join(["%s"] * len(series_ids))

    sql = f"""
        SELECT currency, time, value
        FROM {schema}.{source_table}
        WHERE series_id IN ({placeholders})
        ORDER BY currency, time
    """

    df = pd.read_sql(sql, conn, params=series_ids)
    if df.empty:
        logger.warning("No data for %s in %s.%s", block_name, schema, source_table)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    panel = df.pivot_table(
        index="time", columns="currency", values="value", aggfunc="last"
    )
    panel = panel.sort_index()

    available = [c for c in CURRENCIES if c in panel.columns]
    panel = panel[available]

    logger.info(
        "  Loaded %s: %d dates × %d currencies, range %s to %s",
        block_name, len(panel), len(panel.columns),
        panel.index[0].date(), panel.index[-1].date(),
    )
    return panel


# ───────────────────────────────────────────────────────────────────
# Pair signal computation
# ───────────────────────────────────────────────────────────────────
# Inverse-volatility block weighting
# ───────────────────────────────────────────────────────────────────

def _invvol_composite(
    block_scores: dict[str, pd.DataFrame],
    vol_window: int = 261,
    min_periods: int = 63,
) -> pd.DataFrame:
    """
    Combine block scores using inverse-volatility weighting.

    At each date, each block's weight is proportional to 1/σ where σ
    is the trailing rolling std of the block's cross-sectional mean
    absolute score. This ensures all blocks contribute equally to
    composite variance regardless of update frequency.

    Parameters
    ----------
    block_scores : dict
        Block name → wide DataFrame (dates × currencies), all ffilled
        to daily.
    vol_window : int
        Rolling window for volatility estimation (default 261 = 1 year).
    min_periods : int
        Minimum observations before vol estimate is valid (default 63 = 3 months).

    Returns
    -------
    DataFrame: inverse-vol weighted composite (dates × currencies).
    """
    labels = list(block_scores.keys())
    dfs = list(block_scores.values())

    # Align all blocks to the same index
    all_idx = dfs[0].index
    for df in dfs[1:]:
        all_idx = all_idx.union(df.index)
    all_idx = all_idx.sort_values()

    cols = dfs[0].columns
    aligned = {lab: df.reindex(index=all_idx, columns=cols) for lab, df in zip(labels, dfs)}

    # Compute rolling volatility per block: std of score levels
    # across currencies over trailing window.
    #
    # We use the cross-sectional mean of each currency's rolling std.
    # This captures how much the signal *varies* over the window,
    # regardless of update frequency. Rates (daily changes) will have
    # high rolling std; monthly blocks (step functions) will have low
    # rolling std between releases.
    block_vols = {}
    for lab, df in aligned.items():
        # Rolling std per currency, then average across currencies
        rolling_std = df.rolling(vol_window, min_periods=min_periods).std()
        # Expanding fallback for early period
        expanding_std = df.expanding(min_periods=min_periods).std()
        rolling_std = rolling_std.fillna(expanding_std)
        # Cross-sectional mean of per-currency vols
        vol = rolling_std.mean(axis=1)
        block_vols[lab] = vol

    # Stack into DataFrame for weight computation
    vol_df = pd.DataFrame(block_vols, index=all_idx)

    # Inverse-vol weights: w_i = (1/σ_i) / Σ(1/σ_j)
    # Floor vol at 1e-8 to avoid division by zero
    inv_vol = 1.0 / vol_df.clip(lower=1e-8)
    inv_vol_sum = inv_vol.sum(axis=1)
    weights_df = inv_vol.div(inv_vol_sum, axis=0)

    # Where vol isn't available yet, fall back to equal weight
    equal_w = 1.0 / len(labels)
    weights_df = weights_df.fillna(equal_w)

    # Log average weights for diagnostics
    avg_weights = weights_df.tail(261).mean()
    for lab in labels:
        logger.info("    Inv-vol weight %s: %.1f%% (trailing 1Y avg)",
                    lab, avg_weights[lab] * 100)

    # Weighted average: for each date, composite = Σ(w_i × block_i)
    stacked = np.stack([aligned[lab].values for lab in labels], axis=0)  # (N_blocks, T, C)
    w_arr = np.stack([weights_df[lab].values for lab in labels], axis=0)  # (N_blocks, T)

    # Expand weights to (N_blocks, T, C) for broadcasting
    w_3d = w_arr[:, :, np.newaxis] * np.ones((1, 1, len(cols)))

    # Handle NaN in block scores: zero out weight where block is NaN
    mask = ~np.isnan(stacked)
    w_3d = w_3d * mask.astype(float)
    w_sum = w_3d.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            w_sum > 0,
            np.nansum(stacked * w_3d, axis=0) / w_sum,
            np.nan,
        )

    return pd.DataFrame(result, index=all_idx, columns=cols)


# ───────────────────────────────────────────────────────────────────
# Pair signal computation
# ───────────────────────────────────────────────────────────────────

def _compute_pair_signals(
    country_composite: pd.DataFrame,
) -> dict[str, pd.Series]:
    """
    Compute pair signals from country composites.
    pair_signal = base_score - quote_score
    """
    pair_signals = {}

    for base, quote in PAIRS:
        pair_name = f"{base}{quote}"

        if base not in country_composite.columns:
            logger.warning("  %s: missing base %s", pair_name, base)
            continue
        if quote not in country_composite.columns:
            logger.warning("  %s: missing quote %s", pair_name, quote)
            continue

        sig = (country_composite[base] - country_composite[quote]).dropna()
        if not sig.empty:
            pair_signals[pair_name] = sig

        if pair_name in pair_signals:
            sig = pair_signals[pair_name]
            logger.info(
                "  %s: latest=%+.3f, mean=%+.3f, range [%.2f, %.2f]",
                pair_name.upper(),
                sig.iloc[-1],
                sig.mean(),
                sig.min(), sig.max(),
            )

    return pair_signals


# ───────────────────────────────────────────────────────────────────
# RORO additive country-level modifier
# ───────────────────────────────────────────────────────────────────

# Risk sensitivity per currency: +1 = risk-on, -1 = safe haven
# AQR/Macrosynergy convention: AUD, NZD are the most risk-sensitive,
# CAD moderately so, JPY and CHF are safe havens.
RISK_BETA = {
    "aud": +1.0,
    "nzd": +1.0,
    "cad": +0.5,
    "gbp": +0.25,
    "eur":  0.0,
    "usd":  0.0,
    "chf": -0.75,
    "jpy": -1.0,
}

# Modifier strength: how much RORO shifts country scores.
# Additive: at extreme roro_score (~±2) and max beta (±1),
# this adds/subtracts up to ±1.0 z-score units.
# Tunable via backtest — start at 0.5.
RORO_ALPHA = 0.0


def _load_roro_score(conn, schema: str) -> pd.Series | None:
    """
    Load the composite RORO score from macro.roro_v2.

    The roro_v2 table has columns: time, roro2_score, roro2_regime, etc.
    Returns a daily Series (date index, float values) or None if unavailable.
    """
    try:
        df = pd.read_sql(f"""
            SELECT time, roro2_score
            FROM {schema}.roro_v2
            WHERE roro2_score IS NOT NULL
            ORDER BY time
        """, conn)

        if df.empty:
            return None

        df["time"] = pd.to_datetime(df["time"])
        df["roro2_score"] = pd.to_numeric(df["roro2_score"], errors="coerce")
        s = df.set_index("time")["roro2_score"].sort_index().dropna()
        logger.info("  RORO: loaded roro2_score from roro_v2, %d dates, range [%.2f, %.2f]",
                    len(s), s.min(), s.max())
        return s
    except Exception as e:
        logger.warning("  RORO: failed to load from roro_v2: %s", e)
        return None



# ───────────────────────────────────────────────────────────────────
# Main pipeline
# ───────────────────────────────────────────────────────────────────

def compute_composite_signals(
    conn=None,
    schema: str = "macro",
    block_panels: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Full composite scoring pipeline — v2 hierarchical architecture.

    Pipeline:
        1. Load/accept block scores at native frequency
        2. FFill all blocks to daily (the only ffill in the pipeline)
        3. Inv-vol weight blocks → raw composite → rescore (cs_demean=True)
        4. RORO additive modifier on country scores
        5. Pair signal = base - quote

    No separate cross_sectional_zscore — the v2 rescore already
    includes cross-sectional demeaning at every level.

    Parameters
    ----------
    conn : psycopg2 connection or None
        Required if block_panels is None (loads from DB).
    schema : str
        Database schema.
    block_panels : dict or None
        Pre-computed block scores keyed by block name.
        Expected keys: growth_score_v2, labour_score_v2,
                       monetary_score_v2, rates_score_v2

    Returns
    -------
    Dict of all outputs:
        "country_composite"  — wide DataFrame of rescored country
                               scores (8 currencies)
        "{pair_name}_signal" — Series per pair (e.g. "eurusd_signal")
    """
    logger.info("═══ Composite Signals v2 ═══")

    # ── Load or accept block scores ───────────────────────────────
    block_scores = {}

    if block_panels is not None:
        for key in BLOCK_MAP:
            if key in block_panels and not block_panels[key].empty:
                block_scores[key] = block_panels[key]
                logger.info("  Using pre-computed %s", key)
            else:
                logger.warning("  Missing pre-computed %s", key)
    else:
        if conn is None:
            raise ValueError("Must provide either conn or block_panels")

        for block_name, source_table in BLOCK_MAP.items():
            panel = _load_block_score(block_name, source_table, conn, schema)
            if not panel.empty:
                block_scores[block_name] = panel

    if not block_scores:
        logger.error("No valid block scores — cannot compute composite")
        return {}

    logger.info("  Compositing %d blocks: %s", len(block_scores), list(block_scores.keys()))

    # ── Inverse-volatility block weighting ─────────────────────
    # Weight each block by 1/σ of its signal so that all blocks
    # contribute equally to composite variance. This prevents
    # daily-updating blocks (rates) from dominating over monthly
    # blocks (labour, growth). Volatility is the most predictable
    # characteristic of signals, so this is the one safe improvement
    # over equal weighting (Rob Carver / JPMaQS guidance).
    #
    # Uses trailing 1-year rolling std, expanding at start.
    # Weights are recomputed daily and normalised to sum to 1.

    # ── FFill all blocks to daily business days ───────────────────
    for key in block_scores:
        block_scores[key] = ffill_to_daily(block_scores[key])
        logger.info("    FFilled %s to daily: %d rows", key, len(block_scores[key]))


    if len(block_scores) == 1:
        raw_composite = list(block_scores.values())[0]
    else:
        raw_composite = linear_composite(block_scores)

    logger.info(
        "  Raw composite (pre-rescore): mean|val|=%.3f, range [%.2f, %.2f]",
        np.nanmean(np.abs(raw_composite.values)),
        np.nanmin(raw_composite.values),
        np.nanmax(raw_composite.values),
    )

    # Final hierarchical re-normalisation with cs_demean
    country_composite = rescore(raw_composite)

    logger.info(
        "  Country composite (post-rescore): mean|z|=%.3f, range [%.2f, %.2f]",
        np.nanmean(np.abs(country_composite.values)),
        np.nanmin(country_composite.values),
        np.nanmax(country_composite.values),
    )

    # ── Cross-sectional snapshot (pre-RORO) ─────────────────────────
    latest = country_composite.iloc[-1].sort_values(ascending=False)
    logger.info("  Latest country rankings (pre-RORO):")
    for ccy, val in latest.items():
        logger.info("    %s: %+.3f", ccy.upper(), val)

    # ── RORO additive country-level modifier ──────────────────────
    # Shifts country scores based on risk regime BEFORE pair signals
    # are computed. In risk-off: safe havens (JPY, CHF) get boosted,
    # risk-on currencies (AUD, NZD) get dampened. This is additive
    # so it can push AGAINST the macro signal direction — unlike the
    # old multiplicative approach which could only scale magnitude.

    # ── Pair signals: base - quote ────────────────────────────────
    pair_signals = _compute_pair_signals(country_composite)

    # ── Assemble output ───────────────────────────────────────────
    outputs = {"country_composite": country_composite}
    for pair_name, sig in pair_signals.items():
        outputs[f"{pair_name}_signal"] = sig

    return outputs


# ───────────────────────────────────────────────────────────────────
# Wide → long conversion for DB storage
# ───────────────────────────────────────────────────────────────────

def _outputs_to_long(outputs: dict) -> pd.DataFrame:
    """
    Convert composite outputs to long format for DB upsert.

    Country composites: series_id = "{ccy}_composite", currency = ccy
    Pair signals:       series_id = "{pair}_signal", currency = "pair"
    """
    frames = []

    # Country composites
    cc = outputs.get("country_composite")
    if cc is not None:
        for ccy in cc.columns:
            s = cc[ccy].dropna()
            if s.empty:
                continue
            frames.append(pd.DataFrame({
                "currency":  ccy,
                "series_id": f"{ccy}_composite",
                "time":      s.index,
                "value":     s.values.round(4),
            }))

    # Pair signals
    for key, val in outputs.items():
        if not key.endswith("_signal"):
            continue
        if isinstance(val, pd.Series):
            s = val.dropna()
            if s.empty:
                continue
            pair_name = key.replace("_signal", "")
            frames.append(pd.DataFrame({
                "currency":  "pair",
                "series_id": key,
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

def compute_and_store_composite_signals(
    conn,
    schema: str = "macro",
    block_panels: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """
    Compute composite + pair signals and write to DB.

    Parameters
    ----------
    block_panels : dict or None
        If provided, uses pre-computed block scores instead of loading
        from DB. Expected keys: growth_score_v2, labour_score_v2,
        monetary_score_v2, rates_score_v2.
    """

    outputs = compute_composite_signals(
        conn=conn, schema=schema, block_panels=block_panels,
    )

    if outputs:
        long_df = _outputs_to_long(outputs)
        _upsert(long_df, conn, schema)

        n_pairs = sum(1 for k in outputs if k.endswith("_signal"))
        logger.info(
            "✓ Composite signals complete: 8 country scores, %d pair signals, %d total rows",
            n_pairs, len(long_df),
        )

    return outputs