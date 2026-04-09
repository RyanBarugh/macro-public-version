"""
Growth Signals — Flat Scoring Architecture
============================================

Reads raw metrics from macro.growth_derived (and labour_derived for
wages), scores them using MAD z-scores at native frequency (monthly),
then averages into factors and a block score WITHOUT rescoring.

Two normalizations total (across the full pipeline):
    Pass 1: make_zn_scores on each constituent (here, time-series)
    Pass 2: cross_sectional_zscore on the country composite (in composite.py)

Factor structure (2 factors, equal weight):
    F1 (output):    gdp_yoy_pct, ip_yoy_pct, ip_6m_ann   [from growth_derived]
    F2 (wages):     excess_wages                           [from labour_derived]

Reads from:  macro.growth_derived, macro.labour_derived
Writes to:   macro.growth_signals
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
from .zn_scores import make_zn_scores, linear_composite

logger = get_logger(__name__)


# ───────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────

BENCHMARK = "usd"

# Sign: +1 = higher value is bullish for the currency.
# All growth metrics: stronger = bullish → all +1.
FACTORS = {
    "f_output": {
        "constituents": ["gdp_yoy_pct", "ip_yoy_pct", "ip_6m_ann"],
        "signs":        [1, 1, 1],
        # neutral="mean" for all (raw levels)
    },
    "f_wages": {
        "constituents": ["excess_wages"],
        "signs":        [1],
        "source_table": "labour_derived",   # cross-block read
        # neutral="zero" (already excess metric)
    },
}

CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]

SOURCE_TABLE = "growth_derived"
TARGET_TABLE = "growth_signals"


# ───────────────────────────────────────────────────────────────────
# Relative value — subtract benchmark before z-scoring
# ───────────────────────────────────────────────────────────────────

def _make_relative_value(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract USD from all currencies, drop USD column.
    Returns 7-column panel of differentials.
    """
    if BENCHMARK not in panel.columns:
        return panel
    return panel.sub(panel[BENCHMARK], axis=0).drop(columns=[BENCHMARK])


# ───────────────────────────────────────────────────────────────────
# DB helpers
# ───────────────────────────────────────────────────────────────────

def _load_panel(
    metric_suffix: str,
    conn,
    schema: str,
    sign: int = 1,
    source_table: str | None = None,
) -> pd.DataFrame:
    """
    Load a single derived metric into wide format (dates × currencies).

    PIT gating: pivots on estimated_release_date where available,
    falling back to observation date (time) for rows without release
    dates. This ensures data only enters the scoring panel when it
    was publicly available.

    Parameters
    ----------
    source_table : str or None
        Override the module-level SOURCE_TABLE (for cross-block reads,
        e.g. f_wages reading from labour_derived).
    """
    tbl = source_table or SOURCE_TABLE
    series_ids = [f"{ccy}_{metric_suffix}" for ccy in CURRENCIES]
    placeholders = ",".join(["%s"] * len(series_ids))

    sql = f"""
        SELECT currency, time, value, estimated_release_date
        FROM {schema}.{tbl}
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

    # Stay at native frequency — no ffill to daily.
    # Scoring at monthly frequency avoids expanding-window drift
    # on forward-filled values.  FFill happens once at the end
    # in composite.py.

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
    """
    constituent_zn = {}
    factor_source = factor_def.get("source_table")  # None → default SOURCE_TABLE

    for suffix, sign in zip(factor_def["constituents"], factor_def["signs"]):
        panel = _load_panel(suffix, conn, schema, sign=sign, source_table=factor_source)
        if panel.empty:
            logger.warning("  Skipping constituent %s — no data", suffix)
            continue

        # Route neutral: excess/change metrics already centered → zero
        # Raw levels need panel mean to remove structural differences → mean
        # min_obs=36: ~3 years of monthly data (native frequency)
        if "excess" in suffix or "_momentum" in suffix or "_chg" in suffix:
            zn = make_zn_scores(panel, neutral="zero", pan_weight=0.3, min_obs=36)
        else:
            zn = make_zn_scores(panel, neutral="mean", pan_weight=0.3, min_obs=36)
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
        # Just average — no rescore.  The composite inherits its
        # scale from the input z-scores.
        factor_score = linear_composite(constituent_zn)

    logger.info(
        "  Factor %s: mean|z|=%.3f, range [%.2f, %.2f]",
        factor_name,
        np.nanmean(np.abs(factor_score.values)),
        np.nanmin(factor_score.values), np.nanmax(factor_score.values),
    )

    return factor_score, constituent_zn


def compute_growth_signals(
    conn,
    schema: str = "macro",
) -> dict[str, pd.DataFrame]:
    """
    Full growth block scoring pipeline.

    Returns dict of all scored panels:
        "zn_{suffix}"       — z-scored constituents (8 currencies, native freq)
        "{factor_name}"     — factor scores (averaged, not rescored)
        "growth_score"      — final block composite (averaged, not rescored)
    """
    logger.info("═══ Growth Signals (Path B) ═══")

    all_panels = {}
    factor_scores = {}

    for factor_name, factor_def in FACTORS.items():
        logger.info("Factor: %s", factor_name)
        factor_score, constituent_zn = _score_factor(
            factor_name, factor_def, conn, schema,
        )

        for suffix, zn in constituent_zn.items():
            all_panels[f"zn_{suffix}"] = zn

        if not factor_score.empty:
            factor_scores[factor_name] = factor_score
            all_panels[factor_name] = factor_score

    if not factor_scores:
        logger.error("No valid factors — cannot compute growth block score")
        return all_panels

    # Block composite: equal-weight across factors, no rescore.
    # Cross-sectional normalisation happens once in composite.py.
    if len(factor_scores) == 1:
        block_score = list(factor_scores.values())[0]
    else:
        block_score = linear_composite(factor_scores)

    all_panels["growth_score"] = block_score

    logger.info(
        "═══ Growth block score: mean|z|=%.3f, range [%.2f, %.2f] ═══",
        np.nanmean(np.abs(block_score.values)),
        np.nanmin(block_score.values), np.nanmax(block_score.values),
    )

    return all_panels


# ───────────────────────────────────────────────────────────────────
# Wide → long conversion for DB storage
# ───────────────────────────────────────────────────────────────────

def _panels_to_long(panels: dict[str, pd.DataFrame], lookback_days: int = 90) -> pd.DataFrame:
    """
    Convert dict of wide panels to long format for DB upsert.
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

def compute_and_store_growth_signals(
    conn,
    schema: str = "macro",
    lookback_days: int = 90,
) -> dict[str, pd.DataFrame]:
    """
    Compute growth block signals and write to DB.

    Returns the dict of all scored panels (useful for backtesting
    without needing to re-read from DB).
    """

    try:
        panels = compute_growth_signals(conn, schema)

        if panels:
            long_df = _panels_to_long(panels, lookback_days=lookback_days)
            _upsert(long_df, conn, schema)
            logger.info(
                "✓ Growth signals complete: %d panels, %d total rows",
                len(panels), len(long_df),
            )

        return panels

    except Exception as e:
        logger.error("Growth signals failed: %s", e, exc_info=True)
        raise