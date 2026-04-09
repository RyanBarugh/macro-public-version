"""
Composite Signals — Flat Scoring Architecture
================================================

Combines block scores into a single country-level composite, applies
cross-sectional normalisation (Pete's Step 2), then constructs pair signals.

Pipeline:
    1. Load block scores (monthly/daily native frequency)
    2. FFill all blocks to daily business days (this is the only ffill)
    3. Equal-weight blocks → country composite (no rescore)
    4. Cross-sectional z-score across 8 currencies at each date, clip ±2.5
    5. Pair signal = base_score - quote_score
    6. Store country composites + pair signals

Two normalizations total:
    Pass 1: make_zn_scores on each constituent (in *_signals.py, time-series)
    Pass 2: cross_sectional_zscore here (across currencies at each date)

The pair signal is the final trading signal:
    positive = bullish for the pair (buy base / sell quote)
    negative = bearish for the pair

Reads from:  macro.growth_signals, macro.labour_signals, macro.monetary_signals, macro.rates_signals
Writes to:   macro.composite_signals
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import warnings
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")


from ..engine.logger import get_logger
from .zn_scores import linear_composite, cross_sectional_zscore, ffill_to_daily

logger = get_logger(__name__)


BENCHMARK = "usd"

CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]

TARGET_TABLE = "composite_signals"

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
    Load a block score (e.g. growth_score) into wide format.

    Path A: block scores have all 8 currencies.
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
# Pair signal computation (Path B)
# ───────────────────────────────────────────────────────────────────

def _compute_pair_signals(
    country_composite: pd.DataFrame,
) -> dict[str, pd.Series]:
    """
    Compute pair signals from country composites (Path A).

    Path A: all 8 currencies in panel, simple diff for every pair.
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
# Main pipeline
# ───────────────────────────────────────────────────────────────────

def compute_composite_signals(
    conn=None,
    schema: str = "macro",
    block_panels: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Full composite scoring pipeline — flat architecture.

    Pipeline:
        1. Load/accept block scores at native frequency
        2. FFill all blocks to daily (the only ffill in the pipeline)
        3. Equal-weight blocks → raw country composite (no rescore)
        4. Cross-sectional z-score across 8 currencies, clip ±2.5
        5. Pair signal = base - quote

    Parameters
    ----------
    conn : psycopg2 connection or None
        Required if block_panels is None (loads from DB).
    schema : str
        Database schema.
    block_panels : dict or None
        Pre-computed block scores keyed by block name.

    Returns
    -------
    Dict of all outputs:
        "country_composite"  — wide DataFrame of cross-sectionally normalised
                               country scores (8 currencies, ±2.5 scale)
        "{pair_name}_signal" — Series per pair (e.g. "eurusd_signal")
    """
    logger.info("═══ Composite Signals (Flat Architecture) ═══")

    # ── Load or accept block scores ───────────────────────────────
    block_scores = {}

    if block_panels is not None:
        for key in ["monetary_score", "growth_score", "labour_score", "rates_score"]:
            if key in block_panels and not block_panels[key].empty:
                block_scores[key] = block_panels[key]
                logger.info("  Using pre-computed %s", key)
            else:
                logger.warning("  Missing pre-computed %s", key)
    else:
        if conn is None:
            raise ValueError("Must provide either conn or block_panels")

        block_map = {
            "monetary_score": "monetary_signals",
            "growth_score":    "growth_signals",
            "labour_score":    "labour_signals",
            "rates_score":     "rates_signals",
        }
        for block_name, source_table in block_map.items():
            panel = _load_block_score(block_name, source_table, conn, schema)
            if not panel.empty:
                block_scores[block_name] = panel

    if not block_scores:
        logger.error("No valid block scores — cannot compute composite")
        return {}

    logger.info("  Compositing %d blocks: %s", len(block_scores), list(block_scores.keys()))

    # ── FFill all blocks to daily business days ───────────────────
    # This is the ONLY ffill in the entire pipeline.  Block scores
    # arrive at native frequency (monthly for macro, daily for rates).
    # We ffill to a common daily grid so they can be averaged.
    for key in block_scores:
        block_scores[key] = ffill_to_daily(block_scores[key])
        logger.info("    FFilled %s to daily: %d rows",
                    key, len(block_scores[key]))

    # ── Country composite: equal-weight blocks, NO rescore ────────
    if len(block_scores) == 1:
        raw_composite = list(block_scores.values())[0]
    else:
        raw_composite = linear_composite(block_scores)

    logger.info(
        "  Raw composite (pre-XS): mean|val|=%.3f, range [%.2f, %.2f]",
        np.nanmean(np.abs(raw_composite.values)),
        np.nanmin(raw_composite.values),
        np.nanmax(raw_composite.values),
    )

    # ── Cross-sectional z-score: Pete's Step 2 ───────────────────
    # Z-score across 8 currencies at each date, clip ±2.5.
    # This converts "how strong is this country" into "where does
    # this country rank right now."
    country_composite = cross_sectional_zscore(raw_composite, thresh=2.5)

    logger.info(
        "  Country composite (post-XS): mean|z|=%.3f, range [%.2f, %.2f]",
        np.nanmean(np.abs(country_composite.values)),
        np.nanmin(country_composite.values),
        np.nanmax(country_composite.values),
    )

    # ── Cross-sectional snapshot ──────────────────────────────────
    latest = country_composite.iloc[-1].sort_values(ascending=False)
    logger.info("  Latest country rankings:")
    for ccy, val in latest.items():
        logger.info("    %s: %+.3f", ccy.upper(), val)

    # ── Pair signals: base - quote, no further normalisation ──────
    pair_signals = _compute_pair_signals(country_composite)

    # ── Assemble output ───────────────────────────────────────────
    outputs = {"country_composite": country_composite}
    for pair_name, sig in pair_signals.items():
        outputs[f"{pair_name}_signal"] = sig

    return outputs


# ───────────────────────────────────────────────────────────────────
# Wide → long conversion for DB storage
# ───────────────────────────────────────────────────────────────────

def _outputs_to_long(outputs: dict, lookback_days: int = 90) -> pd.DataFrame:
    """
    Convert composite outputs to long format for DB upsert.

    Country composites: series_id = "{ccy}_composite", currency = ccy
    Pair signals:       series_id = "{pair}_signal", currency = "pair"
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days) if lookback_days > 0 else None
    frames = []

    # Country composites
    cc = outputs.get("country_composite")
    if cc is not None:
        for ccy in cc.columns:
            s = cc[ccy].dropna()
            if cutoff is not None:
                s = s[s.index >= cutoff]
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
            if cutoff is not None:
                s = s[s.index >= cutoff]
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
    lookback_days: int = 90,
    block_panels: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """
    Compute composite + pair signals and write to DB.

    Parameters
    ----------
    block_panels : dict or None
        If provided, uses pre-computed block scores instead of loading
        from DB. Pass the block_score panels from each block's compute
        function to avoid a DB round-trip.
    """

    outputs = compute_composite_signals(
        conn=conn, schema=schema, block_panels=block_panels,
    )

    if outputs:
        long_df = _outputs_to_long(outputs, lookback_days=lookback_days)
        _upsert(long_df, conn, schema)

        n_pairs = sum(1 for k in outputs if k.endswith("_signal"))
        logger.info(
            "✓ Composite signals complete: 8 country scores, %d pair signals, %d total rows",
            n_pairs, len(long_df),
        )

    return outputs