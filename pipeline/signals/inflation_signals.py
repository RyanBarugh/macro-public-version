"""
Inflation / Monetary Tightening Signals — JPMaQS-Aligned Block Scoring
=======================================================================

Reads metrics from macro.inflation_derived, scores them using
panel-pooled MAD z-scores (neutral="zero"), composites into factors
and a block score with re-scoring at every aggregation step.

Merges inflation and real yields into a single block so that
inflation context is interpreted alongside the policy response.
High inflation + high real yield = genuinely hawkish (bullish).
High inflation + low real yield = behind the curve (bearish).

Constituents (positive = hawkish/tight = bullish):
    excess_core_cpi    — core CPI 6M ann minus inflation target
    ppi_excess         — PPI YoY minus inflation target
    deflator_excess    — GDP deflator YoY minus inflation target
    infe_excess        — formulaic inflation expectations minus effective target
    real_yield_2y      — 2Y nominal yield minus INFE_2Y (real yield level)

Factor structure:
    F1 (excess_cpi):       excess_core_cpi
    F2 (excess_producer):  ppi_excess, deflator_excess
    F3 (expectations):     infe_excess
    F4 (real_rates):       real_yield_2y

Pipeline:
    1. Load each constituent into wide panel (dates × 8 currencies)
    2. Z-score each constituent: make_zn_scores(panel)
    3. Equal-weight within factors → rescore
    4. Equal-weight across factors → rescore → inflation_score

Reads from:  macro.inflation_derived
Writes to:   macro.inflation_signals

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.inflation_signals (
        currency    TEXT          NOT NULL,
        series_id   TEXT          NOT NULL,
        time        DATE          NOT NULL,
        value       FLOAT,
        updated_at  TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_inflation_signals_time
        ON macro.inflation_signals(time);
    CREATE INDEX IF NOT EXISTS idx_inflation_signals_currency
        ON macro.inflation_signals(currency);
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text

from ..engine.config import DbConfig
from ..engine.logger import get_logger
from .zn_scores import make_zn_scores, linear_composite, rescore, ffill_to_daily

logger = get_logger(__name__)


# ───────────────────────────────────────────────────────────────────
# Factor definitions — monetary tightening block
# ───────────────────────────────────────────────────────────────────
# F1: Relative excess CPI inflation — core CPI trend minus target
# F2: Relative excess producer prices — avg of PPI and deflator vs target
# F3: Relative inflation expectations — formulaic INFE blend minus eff target
# F4: Real yield level — 2Y nominal yield minus INFE_2Y
# All positive = hawkish/tight = bullish for the currency.

FACTORS = {
    "f_excess_cpi": {
        "constituents": ["excess_core_cpi"],
        "signs":        [1],
    },
    "f_excess_producer": {
        "constituents": ["ppi_excess", "deflator_excess"],
        "signs":        [1, 1],
    },
    "f_expectations": {
        "constituents": ["infe_excess"],
        "signs":        [1],
    },
    "f_real_rates": {
        "constituents": ["real_yield_2y"],
        "signs":        [1],
    },
}

CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]


# ───────────────────────────────────────────────────────────────────
# DB helpers
# ───────────────────────────────────────────────────────────────────

def _load_panel(
    metric_suffix: str,
    engine,
    schema: str,
    sign: int = 1,
) -> pd.DataFrame:
    """
    Load a single derived metric into wide format (dates × currencies).

    Queries macro.inflation_derived for all currencies matching the
    metric suffix, pivots to wide, applies sign convention.
    """
    # Build list of series_ids: usd_excess_core_cpi, eur_excess_core_cpi, ...
    series_ids = [f"{ccy}_{metric_suffix}" for ccy in CURRENCIES]
    placeholders = ",".join([f":s{i}" for i in range(len(series_ids))])
    params = {f"s{i}": sid for i, sid in enumerate(series_ids)}

    sql = text(f"""
        SELECT currency, time, value
        FROM {schema}.inflation_derived
        WHERE series_id IN ({placeholders})
        ORDER BY currency, time
    """)

    df = pd.read_sql(sql, engine, params=params)
    if df.empty:
        logger.warning("No data for metric suffix: %s", metric_suffix)
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Pivot to wide: index=time, columns=currency
    panel = df.pivot_table(
        index="time", columns="currency", values="value", aggfunc="last"
    )
    panel = panel.sort_index()

    # Reindex to ensure consistent column order, only include available
    available = [c for c in CURRENCIES if c in panel.columns]
    panel = panel[available]

    # FFill to daily business days (JPMaQS convention)
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
    engine,
    schema: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Score a single factor: load constituents → z-score → composite → rescore.

    Returns
    -------
    (factor_score, constituent_zn_dict)
        factor_score: wide DataFrame of re-scored factor scores
        constituent_zn_dict: mapping of suffix → z-scored panel (for storage)
    """
    constituent_zn = {}

    for suffix, sign in zip(factor_def["constituents"], factor_def["signs"]):
        panel = _load_panel(suffix, engine, schema, sign=sign)
        if panel.empty:
            logger.warning("  Skipping constituent %s — no data", suffix)
            continue

        zn = make_zn_scores(panel)
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

    # Composite: equal-weight average of z-scored constituents
    if len(constituent_zn) == 1:
        # Single constituent — factor IS the z-score, rescore is ~identity
        factor_score = rescore(list(constituent_zn.values())[0])
    else:
        composite = linear_composite(constituent_zn)
        factor_score = rescore(composite)

    logger.info(
        "  Factor %s: mean|z|=%.3f, range [%.2f, %.2f]",
        factor_name,
        np.nanmean(np.abs(factor_score.values)),
        np.nanmin(factor_score.values), np.nanmax(factor_score.values),
    )

    return factor_score, constituent_zn


def compute_inflation_signals(
    engine,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Full inflation block scoring pipeline.

    Returns dict of all scored panels:
        "zn_{suffix}"       — z-scored constituents
        "{factor_name}"     — re-scored factor scores
        "inflation_score"   — final block composite (re-scored)
    """
    logger.info("═══ Inflation Signals ═══")

    all_panels = {}
    factor_scores = {}

    for factor_name, factor_def in FACTORS.items():
        logger.info("Factor: %s", factor_name)
        factor_score, constituent_zn = _score_factor(
            factor_name, factor_def, engine, schema,
        )

        # Store constituent z-scores
        for suffix, zn in constituent_zn.items():
            all_panels[f"zn_{suffix}"] = zn

        # Store factor score
        if not factor_score.empty:
            factor_scores[factor_name] = factor_score
            all_panels[factor_name] = factor_score

    if not factor_scores:
        logger.error("No valid factors — cannot compute inflation block score")
        return all_panels

    # Block composite: equal-weight across factors → rescore
    if len(factor_scores) == 1:
        block_score = rescore(list(factor_scores.values())[0])
    else:
        block_composite = linear_composite(factor_scores)
        block_score = rescore(block_composite)

    all_panels["inflation_score"] = block_score

    logger.info(
        "═══ Inflation block score: mean|z|=%.3f, range [%.2f, %.2f] ═══",
        np.nanmean(np.abs(block_score.values)),
        np.nanmin(block_score.values), np.nanmax(block_score.values),
    )

    return all_panels


# ───────────────────────────────────────────────────────────────────
# Wide → long conversion for DB storage
# ───────────────────────────────────────────────────────────────────

def _panels_to_long(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert dict of wide panels to long format for DB upsert.

    Each panel key becomes part of the series_id:
        key="zn_excess_core_cpi", currency="usd" → series_id="usd_zn_excess_core_cpi"
        key="inflation_score", currency="eur"    → series_id="eur_inflation_score"
    """
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


# ───────────────────────────────────────────────────────────────────
# Upsert
# ───────────────────────────────────────────────────────────────────

def _upsert(df: pd.DataFrame, db_config: DbConfig, schema: str) -> None:
    if df.empty:
        return

    now = datetime.now(timezone.utc)
    cols = ["currency", "series_id", "time", "value"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]

    logger.info("Upserting %d rows into %s.inflation_signals", len(rows), schema)

    sql = f"""
        INSERT INTO {schema}.inflation_signals (
            currency, series_id, time, value, updated_at
        )
        VALUES %s
        ON CONFLICT (currency, series_id, time)
        DO UPDATE SET
            value      = EXCLUDED.value,
            updated_at = EXCLUDED.updated_at
    """

    conn = psycopg2.connect(
        host=db_config.host, port=db_config.port,
        dbname=db_config.dbname, user=db_config.user,
        password=db_config.password, sslmode=db_config.sslmode,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, page_size=1000)
        logger.info("Upsert complete: %s.inflation_signals rows=%d", schema, len(rows))
    except psycopg2.errors.UndefinedTable:
        logger.error(
            "Table %s.inflation_signals does not exist — run setup SQL first.",
            schema,
        )
        raise
    except Exception:
        logger.exception("Upsert failed: %s.inflation_signals", schema)
        raise
    finally:
        conn.close()


# ───────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────

def compute_and_store_inflation_signals(
    db_config: DbConfig,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Compute inflation block signals and write to DB.

    Returns the dict of all scored panels (useful for backtesting
    without needing to re-read from DB).
    """
    engine = create_engine(
        f"postgresql://{db_config.user}:{db_config.password}@"
        f"{db_config.host}:{db_config.port}/{db_config.dbname}"
    )

    try:
        panels = compute_inflation_signals(engine, schema)

        if panels:
            long_df = _panels_to_long(panels)
            _upsert(long_df, db_config, schema)
            logger.info(
                "✓ Inflation signals complete: %d panels, %d total rows",
                len(panels), len(long_df),
            )

        return panels

    except Exception as e:
        logger.error("Inflation signals failed: %s", e, exc_info=True)
        raise
    finally:
        engine.dispose()