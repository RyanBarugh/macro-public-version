"""
COT Derived — Positioning Metrics per Currency
================================================

Maps CME/ICE FX futures markets to the 8-currency universe, extracts
positioning metrics from TFF and Legacy COT tables, and writes them
to macro.cot_derived in the standard (currency, series_id, time, value)
format used by every other derived module.

NO scoring, NO z-scores, NO points. Pure data reshaping and re-keying.
Scoring belongs exclusively in cot_signals.py.

Methodology:
    Each CME currency future maps 1:1 to a currency. USD uses DXY futures.
    TFF report provides the primary signal groups (asset manager, leveraged
    funds). Legacy report provides retail (non-reportable) positioning.

    COT data is released every Friday (3:30pm ET) for the prior Tuesday's
    positions. estimated_release_date is set to the Friday following each
    report_date (Tuesday).

Output metrics per currency:
    Levels (% of OI):
        {ccy}_am_net_pct_oi         — Asset manager net as % of open interest
        {ccy}_am_long_pct_oi        — Asset manager longs as % of OI
        {ccy}_am_short_pct_oi       — Asset manager shorts as % of OI
        {ccy}_lev_net_pct_oi        — Leveraged funds net as % of OI
        {ccy}_lev_long_pct_oi       — Leveraged funds longs as % of OI
        {ccy}_lev_short_pct_oi      — Leveraged funds shorts as % of OI
        {ccy}_retail_net_pct_oi     — Retail (non-reportable) net as % of OI

    Weekly changes (raw contracts):
        {ccy}_am_long_delta         — Week-on-week change in AM longs
        {ccy}_am_short_delta        — Week-on-week change in AM shorts
        {ccy}_lev_long_delta        — Week-on-week change in leveraged longs
        {ccy}_lev_short_delta       — Week-on-week change in leveraged shorts

    Rolling momentum (cumulative weekly changes):
        {ccy}_am_long_mom_4w        — 4-week rolling sum of AM long changes
        {ccy}_am_short_mom_4w       — 4-week rolling sum of AM short changes
        {ccy}_lev_long_mom_4w       — 4-week rolling sum of leveraged long changes
        {ccy}_lev_short_mom_4w      — 4-week rolling sum of leveraged short changes
        {ccy}_am_long_mom_13w       — 13-week versions
        {ccy}_am_short_mom_13w
        {ccy}_lev_long_mom_13w
        {ccy}_lev_short_mom_13w

    = 19 metrics per currency × 8 currencies = 152 series

Reads from:  pct_of_oi_tff, weekly_changes_tff, rolling_momentum_tff, pct_of_oi
Writes to:   macro.cot_derived

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.cot_derived (
        currency                TEXT          NOT NULL,
        series_id               TEXT          NOT NULL,
        time                    DATE          NOT NULL,
        value                   FLOAT,
        estimated_release_date  DATE,
        updated_at              TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_cot_derived_time
        ON macro.cot_derived(time);
    CREATE INDEX IF NOT EXISTS idx_cot_derived_currency
        ON macro.cot_derived(currency);
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import warnings
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

from ...engine.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_TABLE = "cot_derived"

CURRENCIES = ["usd", "eur", "gbp", "aud", "cad", "jpy", "nzd", "chf"]

# ── Market → currency mapping ─────────────────────────────────────────────────
# CME/ICE FX futures market names as they appear in cot_tff.market column.
# IMPORTANT: Verify these against your actual DB:
#   SELECT DISTINCT market FROM cot.cot_tff ORDER BY market;
# Adjust strings below to match exactly.

MARKET_TO_CURRENCY = {
    "EURO FX - CHICAGO MERCANTILE EXCHANGE":                "eur",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE":          "gbp",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE":      "aud",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE":        "cad",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE":           "jpy",
    "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE":              "nzd",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE":            "chf",
    "USD INDEX - ICE FUTURES U.S.":                         "usd",
}

# Reverse lookup: currency → market name
CURRENCY_TO_MARKET = {v: k for k, v in MARKET_TO_CURRENCY.items()}

# Legacy COT uses identical market names for the same contracts
LEGACY_MARKET_TO_CURRENCY = MARKET_TO_CURRENCY.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# RELEASE DATE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _report_date_to_release_date(report_dates: pd.DatetimeIndex) -> pd.Series:
    """
    COT reports are for Tuesday positions, released the following Friday.
    report_date (Tuesday) + 3 days = Friday release.

    If report_date is not a Tuesday (can happen with holidays), we still
    add enough days to reach the next Friday.
    """
    release_dates = []
    for dt in report_dates:
        # Days until Friday: Monday=0..Sunday=6, Friday=4
        days_ahead = (4 - dt.weekday()) % 7
        if days_ahead <= 0:
            days_ahead += 7
        release_dates.append(dt + timedelta(days=days_ahead))
    return pd.Series(release_dates, index=report_dates)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _rows(
    currency: str,
    metric_id: str,
    s: pd.Series,
    release_dates: pd.Series,
) -> pd.DataFrame:
    """Convert a named Series into long-format rows for DB upsert."""
    s = s.dropna()
    if s.empty:
        return pd.DataFrame()

    rd_aligned = release_dates.reindex(s.index)

    return pd.DataFrame({
        "currency":               currency,
        "series_id":              metric_id,
        "time":                   s.index,
        "value":                  np.round(s.values.astype(float), 4),
        "estimated_release_date": rd_aligned.values,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_tff_pct_of_oi(conn, market: str) -> pd.DataFrame:
    """
    Load % of OI for AM and leveraged funds from pct_of_oi_tff.
    """
    sql = """
        SELECT report_date,
               pct_oi_long_asset_mgr, pct_oi_short_asset_mgr, pct_oi_net_asset_mgr,
               pct_oi_long_lev_funds, pct_oi_short_lev_funds, pct_oi_net_lev_funds
        FROM cot.pct_of_oi_tff
        WHERE market = %s
        ORDER BY report_date
    """
    df = pd.read_sql(sql, conn, params=[market], parse_dates=["report_date"])
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("report_date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_legacy_pct_of_oi(conn, market: str) -> pd.DataFrame:
    """
    Load retail (non-reportable) % of OI from Legacy pct_of_oi.
    """
    sql = """
        SELECT report_date,
               pct_oi_long_retail, pct_oi_short_retail, pct_oi_net_retail
        FROM cot.pct_of_oi
        WHERE market = %s
        ORDER BY report_date
    """
    df = pd.read_sql(sql, conn, params=[market], parse_dates=["report_date"])
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("report_date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_tff_weekly_changes(conn, market: str) -> pd.DataFrame:
    """
    Load weekly position changes for AM and leveraged funds (long + short).
    """
    sql = """
        SELECT report_date,
               d_asset_mgr_long, d_asset_mgr_short,
               d_lev_funds_long, d_lev_funds_short
        FROM cot.weekly_changes_tff
        WHERE market = %s
        ORDER BY report_date
    """
    df = pd.read_sql(sql, conn, params=[market], parse_dates=["report_date"])
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("report_date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_tff_rolling_momentum(conn, market: str) -> pd.DataFrame:
    """
    Load 4w and 13w rolling momentum for AM and leveraged funds (long + short).
    """
    cols = [
        "mom_4w_asset_mgr_long", "mom_4w_asset_mgr_short",
        "mom_4w_lev_funds_long", "mom_4w_lev_funds_short",
        "mom_13w_asset_mgr_long", "mom_13w_asset_mgr_short",
        "mom_13w_lev_funds_long", "mom_13w_lev_funds_short",
    ]
    cols_str = ", ".join(cols)
    sql = f"""
        SELECT report_date, {cols_str}
        FROM cot.rolling_momentum_tff
        WHERE market = %s
        ORDER BY report_date
    """
    df = pd.read_sql(sql, conn, params=[market], parse_dates=["report_date"])
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("report_date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PER-CURRENCY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_currency(
    currency: str,
    conn,
) -> pd.DataFrame:
    """Compute all COT derived metrics for one currency."""
    logger.info("Building COT derived for %s...", currency.upper())

    market = CURRENCY_TO_MARKET.get(currency)
    if market is None:
        logger.warning("  %s: No market mapping configured — skipping", currency.upper())
        return pd.DataFrame()

    frames = []

    # ── Load TFF data ─────────────────────────────────────────────────
    tff_oi = _load_tff_pct_of_oi(conn, market)
    tff_changes = _load_tff_weekly_changes(conn, market)
    tff_momentum = _load_tff_rolling_momentum(conn, market)
    legacy_oi = _load_legacy_pct_of_oi(conn, market)

    if tff_oi.empty:
        logger.warning("  %s: No TFF % of OI data — skipping", currency.upper())
        return pd.DataFrame()

    # ── Release dates ─────────────────────────────────────────────────
    release_dates = _report_date_to_release_date(tff_oi.index)

    # ══════════════════════════════════════════════════════════════════
    # LEVELS: % of OI (for F1 — Crowding / Skew)
    # ══════════════════════════════════════════════════════════════════

    # Asset manager
    if "pct_oi_net_asset_mgr" in tff_oi.columns:
        frames.append(_rows(currency, f"{currency}_am_net_pct_oi",
                            tff_oi["pct_oi_net_asset_mgr"], release_dates))
    if "pct_oi_long_asset_mgr" in tff_oi.columns:
        frames.append(_rows(currency, f"{currency}_am_long_pct_oi",
                            tff_oi["pct_oi_long_asset_mgr"], release_dates))
    if "pct_oi_short_asset_mgr" in tff_oi.columns:
        frames.append(_rows(currency, f"{currency}_am_short_pct_oi",
                            tff_oi["pct_oi_short_asset_mgr"], release_dates))

    # Leveraged funds
    if "pct_oi_net_lev_funds" in tff_oi.columns:
        frames.append(_rows(currency, f"{currency}_lev_net_pct_oi",
                            tff_oi["pct_oi_net_lev_funds"], release_dates))
    if "pct_oi_long_lev_funds" in tff_oi.columns:
        frames.append(_rows(currency, f"{currency}_lev_long_pct_oi",
                            tff_oi["pct_oi_long_lev_funds"], release_dates))
    if "pct_oi_short_lev_funds" in tff_oi.columns:
        frames.append(_rows(currency, f"{currency}_lev_short_pct_oi",
                            tff_oi["pct_oi_short_lev_funds"], release_dates))

    # Retail (from Legacy report)
    if not legacy_oi.empty and "pct_oi_net_retail" in legacy_oi.columns:
        legacy_rd = _report_date_to_release_date(legacy_oi.index)
        frames.append(_rows(currency, f"{currency}_retail_net_pct_oi",
                            legacy_oi["pct_oi_net_retail"], legacy_rd))

    # ══════════════════════════════════════════════════════════════════
    # WEEKLY CHANGES: raw contracts (for F2a — Weekly Spike)
    # ══════════════════════════════════════════════════════════════════

    if not tff_changes.empty:
        changes_rd = _report_date_to_release_date(tff_changes.index)

        change_mapping = {
            "d_asset_mgr_long":  f"{currency}_am_long_delta",
            "d_asset_mgr_short": f"{currency}_am_short_delta",
            "d_lev_funds_long":  f"{currency}_lev_long_delta",
            "d_lev_funds_short": f"{currency}_lev_short_delta",
        }
        for src_col, series_id in change_mapping.items():
            if src_col in tff_changes.columns:
                frames.append(_rows(currency, series_id,
                                    tff_changes[src_col], changes_rd))
    else:
        logger.warning("  %s: No TFF weekly changes data", currency.upper())

    # ══════════════════════════════════════════════════════════════════
    # ROLLING MOMENTUM: cumulative changes (for F2b — Momentum Blend)
    # ══════════════════════════════════════════════════════════════════

    if not tff_momentum.empty:
        mom_rd = _report_date_to_release_date(tff_momentum.index)

        mom_mapping = {
            "mom_4w_asset_mgr_long":   f"{currency}_am_long_mom_4w",
            "mom_4w_asset_mgr_short":  f"{currency}_am_short_mom_4w",
            "mom_4w_lev_funds_long":   f"{currency}_lev_long_mom_4w",
            "mom_4w_lev_funds_short":  f"{currency}_lev_short_mom_4w",
            "mom_13w_asset_mgr_long":  f"{currency}_am_long_mom_13w",
            "mom_13w_asset_mgr_short": f"{currency}_am_short_mom_13w",
            "mom_13w_lev_funds_long":  f"{currency}_lev_long_mom_13w",
            "mom_13w_lev_funds_short": f"{currency}_lev_short_mom_13w",
        }
        for src_col, series_id in mom_mapping.items():
            if src_col in tff_momentum.columns:
                frames.append(_rows(currency, series_id,
                                    tff_momentum[src_col], mom_rd))
    else:
        logger.warning("  %s: No TFF rolling momentum data", currency.upper())

    # ── Log summary ───────────────────────────────────────────────────
    valid = [f for f in frames if not f.empty]
    if not valid:
        logger.warning("  %s: No metrics produced", currency.upper())
        return pd.DataFrame()

    result = pd.concat(valid, ignore_index=True)

    n_metrics = result["series_id"].nunique()
    n_rows = len(result)
    date_range = f"{result['time'].min()} → {result['time'].max()}"

    logger.info(
        "  ✓ %s: %d metrics, %d rows, %s",
        currency.upper(), n_metrics, n_rows, date_range,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# UPSERT
# ═══════════════════════════════════════════════════════════════════════════════

def _upsert(df: pd.DataFrame, conn, schema: str) -> None:
    if df.empty:
        return

    now = datetime.now(timezone.utc)
    cols = ["currency", "series_id", "time", "value", "estimated_release_date"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]

    logger.info("Upserting %d rows into %s.%s", len(rows), schema, TARGET_TABLE)

    sql = f"""
        INSERT INTO {schema}.{TARGET_TABLE} (
            currency, series_id, time, value, estimated_release_date, updated_at
        )
        VALUES %s
        ON CONFLICT (currency, series_id, time)
        DO UPDATE SET
            value                  = EXCLUDED.value,
            estimated_release_date = EXCLUDED.estimated_release_date,
            updated_at             = EXCLUDED.updated_at
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


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_store_cot_derived(
    conn,
    schema: str = "macro",
    currencies: list[str] | None = None,
) -> None:
    """
    Compute and store COT derived metrics for all configured currencies.

    Parameters
    ----------
    conn : psycopg2 connection
        Open connection to the database. Must have access to both the
        TFF/Legacy COT tables (public schema) and the macro schema.
    schema : str
        Target schema for cot_derived table (default "macro").
    currencies : list[str] or None
        Subset of currencies to compute. None = all 8.
    """
    logger.info("═══ COT Derived: mapping CME markets to currencies ═══")

    target_currencies = currencies if currencies else CURRENCIES

    # Verify we have market mappings for all requested currencies
    missing = [c for c in target_currencies if c not in CURRENCY_TO_MARKET]
    if missing:
        logger.error("No market mapping for currencies: %s", missing)
        target_currencies = [c for c in target_currencies if c in CURRENCY_TO_MARKET]

    all_frames = []
    for currency in target_currencies:
        df = _build_currency(currency, conn)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No COT derived data produced")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Ensure proper types before upsert
    combined["time"] = pd.to_datetime(combined["time"]).dt.date
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce").round(4)
    if "estimated_release_date" in combined.columns:
        combined["estimated_release_date"] = pd.to_datetime(
            combined["estimated_release_date"], errors="coerce"
        ).dt.date

    try:
        _upsert(combined, conn, schema)
        logger.info(
            "✓ COT derived complete: %d currencies, %d metrics, %d total rows",
            len(all_frames),
            combined["series_id"].nunique(),
            len(combined),
        )
    except psycopg2.errors.UndefinedTable:
        logger.error(
            "Table %s.%s does not exist — run setup SQL first.",
            schema, TARGET_TABLE,
        )
        raise