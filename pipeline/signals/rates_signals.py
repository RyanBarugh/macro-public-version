"""
Rates Signals — JPMaQS-Aligned Block Scoring
==============================================

Reads metrics from macro.rates_derived, scores them using
panel-pooled MAD z-scores.

Factor structure (1 factor):
    F1 (carry):   yield_2y   (spliced 2Y nominal yield level, neutral="mean")

Higher yield = tighter monetary stance = bullish for the currency.
neutral="mean" removes structural level differences across currencies
(e.g. JPY structurally lower than AUD).

Yields are daily market prices — estimated_release_date = observation date
(no publication lag). Data enters the scoring panel on the day it is observed.

Reads from:  macro.rates_derived
Writes to:   macro.rates_signals

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.rates_signals (
        currency    TEXT          NOT NULL,
        series_id   TEXT          NOT NULL,
        time        DATE          NOT NULL,
        value       FLOAT,
        updated_at  TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_rates_signals_time
        ON macro.rates_signals(time);
    CREATE INDEX IF NOT EXISTS idx_rates_signals_currency
        ON macro.rates_signals(currency);
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
from .zn_scores import make_zn_scores, linear_composite, ffill_to_daily

logger = get_logger(__name__)


# ───────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────

SOURCE_TABLE = "rates_derived"
TARGET_TABLE = "rates_signals"

# Higher yield = tighter = bullish for currency
FACTORS = {
    "f_carry": {
        "constituents": ["yield_2y"],
        "signs":        [1],
    },
}

CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]


# ───────────────────────────────────────────────────────────────────
# DB helpers
# ───────────────────────────────────────────────────────────────────

def _load_panel(
    metric_suffix: str,
    conn,
    schema: str,
    sign: int = 1,
) -> pd.DataFrame:
    """
    Load a single derived metric into wide format (dates × currencies).

    PIT gating: pivots on estimated_release_date where available,
    falling back to observation date (time) for rows without release
    dates. Yields have release_date = observation date (no lag).
    """
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

    # PIT gating: strict — drop rows without estimated_release_date
    if "estimated_release_date" in df.columns:
        df["estimated_release_date"] = pd.to_datetime(
            df["estimated_release_date"], errors="coerce"
        )
        has_rd = df["estimated_release_date"].notna()
        if has_rd.sum() == 0:
            logger.warning(
                "  PIT: %s — no release dates at all, falling back to observation date",
                metric_suffix,
            )
            df["pivot_date"] = df["time"]
        else:
            n_dropped = (~has_rd).sum()
            if n_dropped > 0:
                logger.info(
                    "  PIT: %s — %d rows missing release date, dropped",
                    metric_suffix, n_dropped,
                )
            df = df[has_rd].copy()
            df["pivot_date"] = df["estimated_release_date"]
    else:
        df["pivot_date"] = df["time"]
        logger.info("  PIT: %s — no release date column, using observation date", metric_suffix)

    panel = df.pivot_table(
        index="pivot_date", columns="currency", values="value", aggfunc="last"
    )
    panel = panel.sort_index()

    available = [c for c in CURRENCIES if c in panel.columns]
    panel = panel[available]

    # FFill to daily business days (fills weekends/holidays for yield data)
    panel = ffill_to_daily(panel)

    if sign == -1:
        panel = panel * -1

    logger.info(
        "  Loaded %s: %d dates × %d currencies, range %s to %s",
        metric_suffix, len(panel), len(panel.columns),
        panel.index[0].date(), panel.index[-1].date(),
    )
    return panel


# ───────────────────────────────────────────────────────────────────
# Scoring pipeline
# ───────────────────────────────────────────────────────────────────

def _score_factor(
    factor_name: str,
    factor_def: dict,
    conn,
    schema: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Score a single factor: load constituents → z-score → average.
    No rescore — cross-sectional normalisation happens in composite.py.

    Neutral routing:
        - excess/change/momentum metrics → "zero" (already centered)
        - raw levels (yield_2y) → "mean" (panel mean removes structural levels)
    """
    constituent_zn = {}

    for suffix, sign in zip(factor_def["constituents"], factor_def["signs"]):
        panel = _load_panel(suffix, conn, schema, sign=sign)
        if panel.empty:
            logger.warning("  Skipping constituent %s — no data", suffix)
            continue

        # Route neutral: excess/change/momentum already centered → zero
        # Raw levels need panel mean to remove structural differences → mean
        if any(kw in suffix for kw in ("excess", "_momentum", "_chg", "_mom", "_slope")):
            zn = make_zn_scores(panel, neutral="zero")
        else:
            zn = make_zn_scores(panel, neutral="mean")
        constituent_zn[suffix] = zn
        logger.info(
            "  Z-scored %s: mean|z|=%.3f, range [%.2f, %.2f]",
            suffix,
            np.nanmean(np.abs(zn.values)),
            np.nanmin(zn.values), np.nanmax(zn.values),
        )

    if not constituent_zn:
        logger.warning("  Factor %s: no valid constituents", factor_name)
        return pd.DataFrame(), {}

    # Single constituent: pass through directly
    if len(constituent_zn) == 1:
        factor_score = list(constituent_zn.values())[0]
    else:
        # Just average — no rescore.
        factor_score = linear_composite(constituent_zn)

    logger.info(
        "  Factor %s: mean|z|=%.3f, range [%.2f, %.2f]",
        factor_name,
        np.nanmean(np.abs(factor_score.values)),
        np.nanmin(factor_score.values), np.nanmax(factor_score.values),
    )

    return factor_score, constituent_zn


def compute_rates_signals(
    conn,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Full rates block scoring pipeline.

    Returns dict of all scored panels:
        "zn_{suffix}"    — z-scored constituents (8 currencies)
        "{factor_name}"  — factor scores
        "rates_score"    — final block composite
    """
    logger.info("═══ Rates Signals ═══")

    all_panels = {}
    factor_scores = {}

    for factor_name, factor_def in FACTORS.items():
        logger.info("Factor: %s", factor_name)
        factor_score, constituent_zn = _score_factor(
            factor_name, factor_def, conn, schema,
        )

        # Store constituent z-scores
        for suffix, zn in constituent_zn.items():
            all_panels[f"zn_{suffix}"] = zn

        # Store factor score
        if not factor_score.empty:
            factor_scores[factor_name] = factor_score
            all_panels[factor_name] = factor_score

    if not factor_scores:
        logger.error("No valid factors — cannot compute rates block score")
        return all_panels

    # Block composite: equal-weight across factors, no rescore.
    if len(factor_scores) == 1:
        block_score = list(factor_scores.values())[0]
    else:
        block_score = linear_composite(factor_scores)

    all_panels["rates_score"] = block_score

    logger.info(
        "═══ Rates block score: mean|z|=%.3f, range [%.2f, %.2f] ═══",
        np.nanmean(np.abs(block_score.values)),
        np.nanmin(block_score.values), np.nanmax(block_score.values),
    )

    return all_panels


# ───────────────────────────────────────────────────────────────────
# Wide → long conversion for DB storage
# ───────────────────────────────────────────────────────────────────

def _panels_to_long(panels: dict[str, pd.DataFrame], lookback_days: int = 90) -> pd.DataFrame:
    """Convert dict of wide panels to long format for DB upsert.
    Only includes the last lookback_days of data to avoid rewriting
    the entire history on every run. Set lookback_days=0 for full history.
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days) if lookback_days > 0 else None
    frames = []
    for key, panel in panels.items():
        for ccy in panel.columns:
            s = panel[ccy].dropna()
            if cutoff is not None:
                s = s[s.index >= cutoff]
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


# ───────────────────────────────────────────────────────────────────
# Table creation + Upsert
# ───────────────────────────────────────────────────────────────────

def _ensure_table(conn, schema: str) -> None:
    """Create rates_signals table if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.{TARGET_TABLE} (
                    currency    TEXT          NOT NULL,
                    series_id   TEXT          NOT NULL,
                    time        DATE          NOT NULL,
                    value       FLOAT,
                    updated_at  TIMESTAMPTZ   DEFAULT NOW(),
                    PRIMARY KEY (currency, series_id, time)
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_time
                    ON {schema}.{TARGET_TABLE}(time)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_currency
                    ON {schema}.{TARGET_TABLE}(currency)
            """)
        conn.commit()
        logger.info("Ensured table %s.%s exists", schema, TARGET_TABLE)
    except Exception:
        conn.rollback()
        raise


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

def compute_and_store_rates_signals(
    conn,
    schema: str = "macro",
    lookback_days: int = 90,
) -> dict[str, pd.DataFrame]:
    """
    Compute rates block signals and write to DB.

    Returns the dict of all scored panels (useful for backtesting
    without needing to re-read from DB).
    """

    try:
        _ensure_table(conn, schema)

        panels = compute_rates_signals(conn, schema)

        if panels:
            long_df = _panels_to_long(panels, lookback_days=lookback_days)
            _upsert(long_df, conn, schema)
            logger.info(
                "✓ Rates signals complete: %d panels, %d total rows",
                len(panels), len(long_df),
            )

            return panels

    except Exception as e:
        logger.error("Rates signals failed: %s", e, exc_info=True)
        raise