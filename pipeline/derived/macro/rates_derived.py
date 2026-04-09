"""
Rates Derived — Unified Multi-Currency
=======================================

Computes yield-derived metrics for the rates block across all 8 currencies.
Writes to macro.rates_derived.

NO scoring, NO z-scores. Pure math transforms only.
Scoring belongs exclusively in rates_signals.py.

Rates block factors served by this layer:
  F1: Real yield momentum blend  → real_yield_2y_mom_blend (20d/63d/126d weighted)
  F2: De-meaned carry            → yield_2y_excess_5y (2Y minus 5Y rolling mean)
  F3: Long-end momentum blend    → yield_10y_mom_blend (20d/63d/126d weighted)

  Legacy (kept for backward compat):
  yield_2y_mom_blend (3W+6W avg), yield_2y_momentum (21d), ma21, slope

Data splicing:
  Each currency has a legacy provider (FRED, ECB, RBA, BoC, etc.) with deep history
  and an EODHD provider (2019-02+) for fresh daily updates. The splice loads both,
  uses legacy for pre-EODHD dates and EODHD where available (EODHD wins on overlap).
  Both 2Y and 10Y yields are spliced.

Output metrics per currency:
  {ccy}_yield_2y              — Spliced raw 2Y nominal yield (daily)
  {ccy}_yield_2y_ma21         — 21-day simple moving average
  {ccy}_yield_2y_ma21_slope   — Daily change of 21-day MA (trend direction)
  {ccy}_yield_2y_mom_3w       — 15 business day change
  {ccy}_yield_2y_mom_6w       — 30 business day change
  {ccy}_yield_2y_mom_blend    — Average of 3W + 6W momentum
  {ccy}_yield_2y_momentum     — 21 business day change (legacy compatibility)
  {ccy}_real_yield_2y         — 2Y nominal minus INFE blend (from monetary_derived)
  {ccy}_real_yield_2y_mom_blend — 20d/63d/126d (30/50/20) momentum of real yield
  {ccy}_yield_2y_excess_5y    — 2Y yield minus 5Y rolling mean (de-meaned carry)
  {ccy}_yield_10y             — Spliced 10Y nominal yield (legacy + EODHD)
  {ccy}_yield_10y_mom_blend   — 20d/63d/126d (30/50/20) momentum of 10Y yield

Cross-table dependencies:
  - INFE blend from macro.monetary_derived (monthly, forward-filled to daily)

Reads from:  macro.series_data, macro.monetary_derived
Writes to:   macro.rates_derived

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.rates_derived (
        currency                TEXT          NOT NULL,
        series_id               TEXT          NOT NULL,
        time                    DATE          NOT NULL,
        value                   FLOAT,
        estimated_release_date  DATE,
        updated_at              TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_rates_derived_time
        ON macro.rates_derived(time);
    CREATE INDEX IF NOT EXISTS idx_rates_derived_currency
        ON macro.rates_derived(currency);
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ...engine.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Currency configs
# ---------------------------------------------------------------------------
# yield_2y_id       = legacy provider (deep history)
# yield_2y_eod_id   = EODHD provider (2019-02+, takes priority on overlap)
# yield_10y_id      = legacy provider for 10Y (deep history)
# yield_10y_eod_id  = EODHD 10Y yield (2019+, takes priority on overlap)

CURRENCY_CONFIGS = {
    "usd": {
        "yield_2y_id":      "usd_yield_2y",
        "yield_2y_eod_id":  "usd_yield_2y_eod",
        "yield_10y_id":     "usd_yield_10y",
        "yield_10y_eod_id": "usd_yield_10y_eod",
    },
    "eur": {
        "yield_2y_id":      "eur_yield_2y",
        "yield_2y_eod_id":  "eur_yield_2y_eod",
        "yield_10y_id":     "eur_yield_10y",
        "yield_10y_eod_id": "eur_yield_10y_eod",
    },
    "gbp": {
        "yield_2y_id":      "gbp_yield_2y",
        "yield_2y_eod_id":  "gbp_yield_2y_eod",
        "yield_10y_id":     "gbp_yield_10y",
        "yield_10y_eod_id": "gbp_yield_10y_eod",
    },
    "aud": {
        "yield_2y_id":      "aud_yield_2y",
        "yield_2y_eod_id":  "aud_yield_2y_eod",
        "yield_10y_id":     "aud_yield_10y",
        "yield_10y_eod_id": "aud_yield_10y_eod",
    },
    "cad": {
        "yield_2y_id":      "cad_yield_2y",
        "yield_2y_eod_id":  "cad_yield_2y_eod",
        "yield_10y_id":     "cad_yield_10y",
        "yield_10y_eod_id": "cad_yield_10y_eod",
    },
    "jpy": {
        "yield_2y_id":      "jpy_yield_2y",
        "yield_2y_eod_id":  "jpy_yield_2y_eod",
        "yield_10y_id":     "jpy_yield_10y",
        "yield_10y_eod_id": "jpy_yield_10y_eod",
    },
    "nzd": {
        "yield_2y_id":      "nzd_yield_2y",
        "yield_2y_eod_id":  "nzd_yield_2y_eod",
        "yield_10y_id":     "nzd_yield_10y",
        "yield_10y_eod_id": "nzd_yield_10y_eod",
    },
    "chf": {
        "yield_2y_id":      "chf_yield_2y",
        "yield_2y_eod_id":  "chf_yield_2y_eod",
        "yield_10y_id":     "chf_yield_10y",
        "yield_10y_eod_id": "chf_yield_10y_eod",
    },
}

# ---------------------------------------------------------------------------
# Momentum blend constants (same as financial_conditions.py)
# ---------------------------------------------------------------------------

MOM_WINDOWS = (20, 63, 126)       # 1M, 3M, 6M in business days
MOM_WEIGHTS = (0.30, 0.50, 0.20)  # Weight on short/medium/long


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_raw(series_id: str, conn, schema: str) -> pd.Series:
    sql = f"""
        SELECT time, value
        FROM {schema}.series_data
        WHERE series_id = %s
        ORDER BY time
    """
    df = pd.read_sql(sql, conn, params=[series_id])
    if df.empty:
        logger.warning("No data for series_id=%s", series_id)
        return pd.Series(dtype=float)
    df["time"]  = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value"]).sort_values("time")
    return df.set_index("time")["value"]


def _load_cross_table(
    conn, schema: str, table: str, currency: str, series_suffix: str,
) -> pd.Series:
    """
    Load a single metric from another derived table.
    Returns series indexed by time (preserves native frequency).
    """
    series_id = f"{currency}_{series_suffix}"
    sql = f"""
        SELECT time, value
        FROM {schema}.{table}
        WHERE currency = %s AND series_id = %s
        ORDER BY time
    """
    df = pd.read_sql(sql, conn, params=[currency, series_id])
    if df.empty:
        logger.warning("  Cross-table: no data for %s.%s", table, series_id)
        return pd.Series(dtype=float)
    df["time"]  = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value"]).sort_values("time")
    return df.set_index("time")["value"]


# ---------------------------------------------------------------------------
# Splice helper
# ---------------------------------------------------------------------------

def _splice_yields(
    legacy: pd.Series,
    eodhd: pd.Series,
) -> pd.Series:
    """
    Splice legacy provider (deep history) with EODHD (2019+).
    EODHD takes priority where both have data.
    Returns daily series sorted by time.
    """
    if legacy.empty and eodhd.empty:
        return pd.Series(dtype=float)
    if eodhd.empty:
        return legacy.sort_index()
    if legacy.empty:
        return eodhd.sort_index()

    # Use legacy for dates before EODHD starts, EODHD for everything else
    eodhd_start = eodhd.dropna().index.min()
    before = legacy[legacy.index < eodhd_start]
    spliced = pd.concat([before, eodhd]).sort_index()
    # Safety: drop any remaining duplicates (keep EODHD = last)
    spliced = spliced[~spliced.index.duplicated(keep="last")]
    return spliced


# ---------------------------------------------------------------------------
# Momentum blend helper
# ---------------------------------------------------------------------------

def _momentum_blend(s: pd.Series) -> pd.Series:
    """
    Multi-window momentum blend: 20d/63d/126d weighted 30%/50%/20%.
    Same pattern as financial_conditions.py momentum_composite.
    Returns weighted sum of changes (not z-scored — that happens in signals layer).
    """
    components = []
    for lookback, weight in zip(MOM_WINDOWS, MOM_WEIGHTS):
        chg = s - s.shift(lookback)
        components.append(chg * weight)

    blend = components[0]
    for c in components[1:]:
        blend = blend + c
    return blend.round(4)


# ---------------------------------------------------------------------------
# Row collector
# ---------------------------------------------------------------------------

def _rows(
    currency: str,
    metric_id: str,
    s: pd.Series,
    rd: pd.Series | None = None,
) -> pd.DataFrame:
    s = s.dropna()
    if s.empty:
        return pd.DataFrame()
    d = {
        "currency":  currency,
        "series_id": metric_id,
        "time":      s.index,
        "value":     s.values,
    }
    if rd is not None:
        aligned = rd.reindex(s.index)
        d["estimated_release_date"] = aligned.values
    else:
        d["estimated_release_date"] = s.index  # yields: release = observation
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# Per-currency computation
# ---------------------------------------------------------------------------

def _build_currency(
    currency: str,
    cfg: dict,
    conn,
    schema: str,
    infe_blend: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute all rates derived metrics for one currency."""

    logger.info("─── %s rates derived ───", currency.upper())
    frames: list[pd.DataFrame] = []

    # ==================================================================
    # LOAD + SPLICE 2Y YIELD
    # ==================================================================

    legacy = _load_raw(cfg["yield_2y_id"], conn, schema)
    eodhd  = _load_raw(cfg["yield_2y_eod_id"], conn, schema)

    y2 = _splice_yields(legacy, eodhd)

    if y2.empty:
        logger.warning("  %s: no yield data at all — skipping", currency.upper())
        return pd.DataFrame()

    # Log splice info
    if not legacy.empty and not eodhd.empty:
        eodhd_start = eodhd.dropna().index.min()
        logger.info("  %s: spliced %s (%s→%s) + %s (%s→%s), cutoff=%s",
                     currency.upper(),
                     cfg["yield_2y_id"],
                     legacy.index.min().strftime("%Y-%m-%d"),
                     legacy.index.max().strftime("%Y-%m-%d"),
                     cfg["yield_2y_eod_id"],
                     eodhd.index.min().strftime("%Y-%m-%d"),
                     eodhd.index.max().strftime("%Y-%m-%d"),
                     eodhd_start.strftime("%Y-%m-%d"))
    logger.info("  %s: yield_2y range %s → %s (%d obs)",
                currency.upper(),
                y2.index.min().strftime("%Y-%m-%d"),
                y2.index.max().strftime("%Y-%m-%d"),
                len(y2))

    # Yields are public on observation date
    rd_y2 = pd.Series(y2.index, index=y2.index)

    # ==================================================================
    # METRIC 1: Raw spliced 2Y yield level
    # ==================================================================

    frames.append(_rows(currency, f"{currency}_yield_2y", y2.round(4), rd_y2))
    logger.info("  ✓ yield_2y: %.2f%%",
                y2.dropna().iloc[-1] if not y2.dropna().empty else float("nan"))

    # ==================================================================
    # METRIC 2: 21-day simple moving average
    # ==================================================================

    if len(y2) > 21:
        ma21 = y2.rolling(window=21, min_periods=15).mean().round(4)
        frames.append(_rows(currency, f"{currency}_yield_2y_ma21", ma21, rd_y2))
        logger.info("  ✓ yield_2y_ma21: %.2f%%",
                     ma21.dropna().iloc[-1] if not ma21.dropna().empty else float("nan"))

    # ==================================================================
    # METRIC 3: Slope of 21-day MA (daily change of MA = trend direction)
    # ==================================================================

    if len(y2) > 22:
        ma21 = y2.rolling(window=21, min_periods=15).mean()
        slope = (ma21 - ma21.shift(1)).round(4)
        frames.append(_rows(currency, f"{currency}_yield_2y_ma21_slope", slope, rd_y2))
        logger.info("  ✓ yield_2y_ma21_slope: %+.4f",
                     slope.dropna().iloc[-1] if not slope.dropna().empty else float("nan"))

    # ==================================================================
    # METRIC 4: 3-week momentum (15 business day change)
    # ==================================================================

    if len(y2) > 15:
        mom_3w = (y2 - y2.shift(15)).round(4)
        frames.append(_rows(currency, f"{currency}_yield_2y_mom_3w", mom_3w, rd_y2))
        logger.info("  ✓ yield_2y_mom_3w: %+.2fpp",
                     mom_3w.dropna().iloc[-1] if not mom_3w.dropna().empty else float("nan"))

    # ==================================================================
    # METRIC 5: 6-week momentum (30 business day change)
    # ==================================================================

    if len(y2) > 30:
        mom_6w = (y2 - y2.shift(30)).round(4)
        frames.append(_rows(currency, f"{currency}_yield_2y_mom_6w", mom_6w, rd_y2))
        logger.info("  ✓ yield_2y_mom_6w: %+.2fpp",
                     mom_6w.dropna().iloc[-1] if not mom_6w.dropna().empty else float("nan"))

    # ==================================================================
    # METRIC 6: Momentum blend (average of 3W + 6W) — legacy
    # ==================================================================

    if len(y2) > 30:
        mom_3w = (y2 - y2.shift(15))
        mom_6w = (y2 - y2.shift(30))
        blend = ((mom_3w + mom_6w) / 2.0).round(4)
        frames.append(_rows(currency, f"{currency}_yield_2y_mom_blend", blend, rd_y2))
        logger.info("  ✓ yield_2y_mom_blend: %+.2fpp",
                     blend.dropna().iloc[-1] if not blend.dropna().empty else float("nan"))

    # ==================================================================
    # METRIC 7: 21-day momentum (legacy compatibility with monetary_derived)
    # ==================================================================

    if len(y2) > 21:
        mom_21d = (y2 - y2.shift(21)).round(4)
        frames.append(_rows(currency, f"{currency}_yield_2y_momentum", mom_21d, rd_y2))
        logger.info("  ✓ yield_2y_momentum: %+.2fpp",
                     mom_21d.dropna().iloc[-1] if not mom_21d.dropna().empty else float("nan"))

    # ==================================================================
    # METRIC 8: Real yield (2Y nominal minus INFE blend)
    # ==================================================================

    real_yield = pd.Series(dtype=float)

    if infe_blend is not None and not infe_blend.empty and not y2.empty:
        # INFE is monthly — forward-fill to daily for subtraction
        infe_daily = infe_blend.reindex(y2.index, method="ffill")
        overlap = y2.index.intersection(infe_daily.dropna().index)
        if len(overlap) > 0:
            real_yield = (y2.loc[overlap] - infe_daily.loc[overlap]).round(4)
            frames.append(_rows(currency, f"{currency}_real_yield_2y", real_yield, rd_y2))
            logger.info("  ✓ real_yield_2y: %.2f%%",
                         real_yield.dropna().iloc[-1] if not real_yield.dropna().empty else float("nan"))
        else:
            logger.warning("  %s: real yield — no overlapping dates", currency.upper())
    else:
        logger.warning("  %s: real yield — INFE blend not available", currency.upper())

    # ==================================================================
    # METRIC 9: Real yield 2Y momentum blend (20d/63d/126d, 30/50/20)
    # ==================================================================
    # F1 in the v2 rates block. Captures direction of real monetary
    # tightening across multiple horizons. Strips out inflation noise
    # that contaminates nominal yield momentum.

    if not real_yield.empty and len(real_yield) > max(MOM_WINDOWS):
        ry_mom = _momentum_blend(real_yield)
        frames.append(_rows(currency, f"{currency}_real_yield_2y_mom_blend", ry_mom, rd_y2))
        logger.info("  ✓ real_yield_2y_mom_blend: %+.4f",
                     ry_mom.dropna().iloc[-1] if not ry_mom.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: real_yield_2y_mom_blend — insufficient data", currency.upper())

    # ==================================================================
    # METRIC 10: 2Y yield excess over 5Y rolling mean (de-meaned carry)
    # ==================================================================
    # F2 in the v2 rates block. Where the yield level sits relative to
    # its own recent history. Removes structural bias (JPY low, AUD high).
    # Centered around zero by construction.

    rolling_window = 1305  # 5 years of business days
    min_periods = 504      # ~2 years before valid

    if len(y2) > min_periods:
        rolling_mean = y2.rolling(window=rolling_window, min_periods=min_periods).mean()
        excess = (y2 - rolling_mean).round(4)
        frames.append(_rows(currency, f"{currency}_yield_2y_excess_5y", excess, rd_y2))
        logger.info("  ✓ yield_2y_excess_5y: %+.2fpp",
                     excess.dropna().iloc[-1] if not excess.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: yield_2y_excess_5y — insufficient data (%d obs)", currency.upper(), len(y2))

    # ==================================================================
    # LOAD + SPLICE 10Y YIELD + METRIC 11: 10Y yield level
    # ==================================================================
    # F3 in the v2 rates block. Long-end captures term premium, fiscal
    # outlook, long-run growth expectations — distinct from 2Y which
    # is mostly near-term policy.
    #
    # Spliced the same way as 2Y: legacy (deep history) + EODHD (2019+).

    y10_legacy_id = cfg.get("yield_10y_id")
    y10_eod_id = cfg.get("yield_10y_eod_id")
    y10 = pd.Series(dtype=float)

    if y10_legacy_id or y10_eod_id:
        legacy_10y = _load_raw(y10_legacy_id, conn, schema) if y10_legacy_id else pd.Series(dtype=float)
        eodhd_10y = _load_raw(y10_eod_id, conn, schema) if y10_eod_id else pd.Series(dtype=float)

        y10 = _splice_yields(legacy_10y, eodhd_10y)

        if not y10.empty:
            y10 = y10.sort_index()
            y10 = y10[~y10.index.duplicated(keep="last")]
            rd_y10 = pd.Series(y10.index, index=y10.index)

            # Log splice info
            if not legacy_10y.empty and not eodhd_10y.empty:
                eodhd_start = eodhd_10y.dropna().index.min()
                logger.info("  %s: 10Y spliced %s (%s→%s) + %s (%s→%s), cutoff=%s",
                             currency.upper(),
                             y10_legacy_id,
                             legacy_10y.index.min().strftime("%Y-%m-%d"),
                             legacy_10y.index.max().strftime("%Y-%m-%d"),
                             y10_eod_id,
                             eodhd_10y.index.min().strftime("%Y-%m-%d"),
                             eodhd_10y.index.max().strftime("%Y-%m-%d"),
                             eodhd_start.strftime("%Y-%m-%d"))
            elif not legacy_10y.empty:
                logger.info("  %s: 10Y from legacy only: %s (%d obs)", currency.upper(), y10_legacy_id, len(legacy_10y))
            elif not eodhd_10y.empty:
                logger.info("  %s: 10Y from EODHD only: %s (%d obs)", currency.upper(), y10_eod_id, len(eodhd_10y))

            frames.append(_rows(currency, f"{currency}_yield_10y", y10.round(4), rd_y10))
            logger.info("  ✓ yield_10y: %.2f%% (%d obs, %s → %s)",
                         y10.dropna().iloc[-1] if not y10.dropna().empty else float("nan"),
                         len(y10),
                         y10.index.min().strftime("%Y-%m-%d"),
                         y10.index.max().strftime("%Y-%m-%d"))

            # ==================================================================
            # METRIC 12: 10Y yield momentum blend (20d/63d/126d, 30/50/20)
            # ==================================================================

            if len(y10) > max(MOM_WINDOWS):
                y10_mom = _momentum_blend(y10)
                frames.append(_rows(currency, f"{currency}_yield_10y_mom_blend", y10_mom, rd_y10))
                logger.info("  ✓ yield_10y_mom_blend: %+.4f",
                             y10_mom.dropna().iloc[-1] if not y10_mom.dropna().empty else float("nan"))
            else:
                logger.warning("  %s: yield_10y_mom_blend — insufficient data (%d obs)", currency.upper(), len(y10))
        else:
            logger.warning("  %s: no 10Y yield data from either source", currency.upper())
    else:
        logger.warning("  %s: no yield_10y_id or yield_10y_eod_id configured", currency.upper())

    # ==================================================================
    # ASSEMBLE
    # ==================================================================

    valid = [f for f in frames if not f.empty]
    if not valid:
        return pd.DataFrame()

    result = pd.concat(valid, ignore_index=True)
    result["time"]  = pd.to_datetime(result["time"]).dt.date
    result["value"] = pd.to_numeric(result["value"], errors="coerce").round(4)
    if "estimated_release_date" in result.columns:
        result["estimated_release_date"] = pd.to_datetime(
            result["estimated_release_date"], errors="coerce",
        ).dt.date
    else:
        result["estimated_release_date"] = None
    logger.info("  %s: %d rows, %d metrics",
                currency.upper(), len(result), result["series_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Schema migration (idempotent)
# ---------------------------------------------------------------------------

def _ensure_table(conn, schema: str) -> None:
    """Create table if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.rates_derived (
                    currency                TEXT          NOT NULL,
                    series_id               TEXT          NOT NULL,
                    time                    DATE          NOT NULL,
                    value                   FLOAT,
                    estimated_release_date  DATE,
                    updated_at              TIMESTAMPTZ   DEFAULT NOW(),
                    PRIMARY KEY (currency, series_id, time)
                )
            """)
        conn.commit()
        logger.info("Ensured table %s.rates_derived", schema)
    except Exception:
        conn.rollback()
        raise

# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(df: pd.DataFrame, conn, schema: str) -> None:
    if df.empty:
        return

    now  = datetime.now(timezone.utc)
    cols = ["currency", "series_id", "time", "value", "estimated_release_date"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]

    logger.info("Upserting %d rows into %s.rates_derived", len(rows), schema)

    sql = f"""
        INSERT INTO {schema}.rates_derived (
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
        logger.info("Upsert complete: %s.rates_derived rows=%d", schema, len(rows))
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        logger.error(
            "Table %s.rates_derived does not exist — run setup SQL first.", schema
        )
        raise
    except Exception:
        conn.rollback()
        logger.exception("Upsert failed: %s.rates_derived", schema)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_and_store_rates_derived(
    conn,
    schema: str = "macro",
    currencies: list[str] | None = None,
    lookback_days: int = 90,
) -> None:
    """
    Compute and store rates derived metrics for all configured currencies.

    Args:
        conn: psycopg2 connection.
        schema: Database schema (default "macro").
        currencies: Optional subset e.g. ['gbp', 'aud'].
                    Defaults to all currencies in CURRENCY_CONFIGS.
        lookback_days: Only upsert rows from the last N days.
                       Set to 0 to write full history (backfill).
                       Default 0 ensures new metrics get full history
                       on first run. Use 90 for daily incremental runs.
    """
    configs = CURRENCY_CONFIGS
    if currencies:
        configs = {k: v for k, v in CURRENCY_CONFIGS.items() if k in currencies}

    _ensure_table(conn, schema)

    all_frames = []
    for currency, cfg in configs.items():
        # Load INFE blend from monetary_derived (monthly)
        infe_blend = _load_cross_table(
            conn, schema, "monetary_derived", currency, "infe_blend",
        )
        if infe_blend.empty:
            logger.warning("  %s: INFE blend not available — real yield will be skipped",
                           currency.upper())

        df = _build_currency(
            currency, cfg, conn, schema,
            infe_blend=infe_blend if not infe_blend.empty else None,
        )
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No rates derived data produced")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    # Filter to recent data on incremental runs
    if lookback_days > 0:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        combined["time"] = pd.to_datetime(combined["time"])
        before = len(combined)
        combined = combined[combined["time"] >= cutoff]
        logger.info("Lookback filter: %d → %d rows (last %d days)", before, len(combined), lookback_days)
    _upsert(combined, conn, schema)
    logger.info(
        "✓ Rates derived complete: %d currencies, %d total rows",
        len(all_frames),
        len(combined),
    )