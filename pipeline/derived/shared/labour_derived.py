"""
Labour Derived — Unified Multi-Currency
========================================

Computes labour derived metrics for all 8 currencies.
Writes to macro.labour_derived (shared table with currency column).

NO scoring, NO z-scores, NO std_scale. Pure math transforms only.

JPMaQS labour factors served by this layer:
  Regression signal: UR changes at 3 windows (3M, 6M, 12M)
  Scorecard F1: Excess employment growth — emp YoY vs 5Y trend
  Scorecard F2: Unemployment gap — UR minus 5Y rolling mean
  Scorecard F3: Excess wage growth — wages YoY minus (target + productivity YoY)

Output metrics per currency:
  {ccy}_unemp_rate          — Unemployment rate level
  {ccy}_unemp_3m_chg        — UR 3M change (regression signal)
  {ccy}_unemp_6m_chg        — UR 6M change (regression signal)
  {ccy}_unemp_12m_chg       — UR 12M change (regression signal)
  {ccy}_unemp_gap           — UR minus 5Y rolling mean (ex-COVID)
  {ccy}_emp_yoy_pct         — Employment level YoY %
  {ccy}_emp_excess          — Emp YoY minus 5Y rolling median (ex-COVID)
  {ccy}_wages_yoy           — Wage growth YoY % (computed or passthrough)
  {ccy}_productivity_yoy    — (Real GDP / employment) YoY % (quarterly)
  {ccy}_excess_wages        — Wages YoY minus (target + productivity YoY)

Reads from:  macro.series_data
Writes to:   macro.labour_derived

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.labour_derived (
        currency    TEXT          NOT NULL,
        series_id   TEXT          NOT NULL,
        time        DATE          NOT NULL,
        value       FLOAT,
        updated_at  TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_labour_derived_time
        ON macro.labour_derived(time);
    CREATE INDEX IF NOT EXISTS idx_labour_derived_currency
        ON macro.labour_derived(currency);
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text

from ...engine.config import DbConfig
from ...engine.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# COVID exclusion window for rolling baselines
# ---------------------------------------------------------------------------

COVID_START = pd.Timestamp("2020-03-01")
COVID_END   = pd.Timestamp("2020-12-01")


# ---------------------------------------------------------------------------
# Currency configs
# ---------------------------------------------------------------------------

CURRENCY_CONFIGS = {
    "usd": {
        "inflation_target": 2.0,
        "unemp_id":         "usd_unemployment_rate_sa",
        "employment_id":    "usd_nfp_level_sa",
        "employment_freq":  "M",
        "wages_id":         "usd_wage_ahe_sa",
        "wages_freq":       "M",
        "wages_type":       "index",
        "gdp_real_id":      "usd_gdp_real_bea_sa",
    },
    "eur": {
        "inflation_target": 2.0,
        "unemp_id":         "eur_unemployment_rate_sa",
        "employment_id":    "eur_emp_level_lfs_sa",
        "employment_freq":  "Q",
        "wages_id":         "eur_negotiated_wages_qoq",   # published as YoY %
        "wages_freq":       "Q",
        "wages_type":       "pct",
        "gdp_real_id":      "eur_gdp_real",
    },
    "gbp": {
        "inflation_target": 2.0,
        "unemp_id":         "gbp_unemployment_rate_sa",
        "employment_id":    "gbp_employment_level_sa",
        "employment_freq":  "M",
        "wages_id":         "gbp_wages_regular_sa",       # published as YoY %
        "wages_freq":       "M",
        "wages_type":       "pct",
        "gdp_real_id":      "gbp_gdp_real",
    },
    "aud": {
        "inflation_target": 2.5,
        "unemp_id":         "aud_unemployment_rate_sa",
        "employment_id":    "aud_employment_level_sa",
        "employment_freq":  "M",
        "wages_id":         "aud_wages_wpi_index_sa",
        "wages_freq":       "Q",
        "wages_type":       "index",
        "gdp_real_id":      "aud_gdp_real",
    },
    "cad": {
        "inflation_target": 2.0,
        "unemp_id":         "cad_unemployment_rate",
        "employment_id":    "cad_employment_level",
        "employment_freq":  "M",
        "wages_id":         "cad_avg_hourly_wage",
        "wages_freq":       "M",
        "wages_type":       "index",
        "gdp_real_id":      "cad_gdp_real",
    },
    "jpy": {
        "inflation_target": 2.0,
        "unemp_id":         "jpy_unemployment_rate_nsa",
        "employment_id":    "jpy_employment_level_nsa",
        "employment_freq":  "M",
        "wages_id":         "jpy_wages_manufacturing_sa",
        "wages_freq":       "M",
        "wages_type":       "index",
        "gdp_real_id":      "jpy_gdp_real",
    },
    "nzd": {
        "inflation_target": 2.0,
        "unemp_id":         "nzd_unemployment_rate_sa",
        "unemp_freq":       "Q",       # NZD UR is quarterly
        "employment_id":    "nzd_employment_level_sa",
        "employment_freq":  "Q",
        "wages_id":         "nzd_avg_hourly_earnings_nsa",
        "wages_freq":       "Q",
        "wages_type":       "index",
        "gdp_real_id":      "nzd_gdp_real_sa",
    },
    "chf": {
        "inflation_target": 2.0,
        "unemp_id":         "chf_unemployment_rate_nsa",
        "employment_id":    "chf_employment_level_sa",
        "employment_freq":  "Q",
        "wages_id":         "chf_wages_nominal_yoy",      # published as YoY %
        "wages_freq":       "Q",
        "wages_type":       "pct",
        "gdp_real_id":      "chf_gdp_real",
    },
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_raw(series_id: str, engine, schema: str) -> pd.Series:
    sql = text(f"""
        SELECT time, value
        FROM {schema}.series_data
        WHERE series_id = :sid
        ORDER BY time
    """)
    df = pd.read_sql(sql, engine, params={"sid": series_id})
    if df.empty:
        logger.warning("No data for series_id=%s", series_id)
        return pd.Series(dtype=float)
    df["time"]  = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value"]).sort_values("time")
    return df.set_index("time")["value"]


# ---------------------------------------------------------------------------
# Frequency normalisation
# ---------------------------------------------------------------------------

def _to_monthly(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index.to_period("M").to_timestamp()
    return s.groupby(level=0).last().sort_index()


def _quarterly_to_monthly(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index.to_period("Q").to_timestamp()
    s = s.groupby(level=0).last().sort_index()
    monthly_idx = pd.date_range(s.index[0], s.index[-1], freq="MS")
    return s.reindex(monthly_idx, method="ffill")


def _to_quarterly(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index.to_period("Q").to_timestamp()
    return s.groupby(level=0).last().sort_index()


def _normalise_to_monthly(s: pd.Series, freq: str) -> pd.Series:
    if freq == "Q":
        return _quarterly_to_monthly(s)
    return _to_monthly(s)


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def _yoy(s: pd.Series) -> pd.Series:
    """YoY for monthly data: shift(12)."""
    return ((s / s.shift(12)) - 1.0).mul(100.0).round(4)


def _yoy_quarterly(s: pd.Series) -> pd.Series:
    """YoY for quarterly data: shift(4)."""
    return ((s / s.shift(4)) - 1.0).mul(100.0).round(4)


def _chg(s: pd.Series, n: int) -> pd.Series:
    """N-period absolute change."""
    return (s - s.shift(n)).round(4)


def _rolling_mean_ex_covid(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Rolling mean excluding COVID window from baseline."""
    masked = s.copy()
    masked[(masked.index >= COVID_START) & (masked.index <= COVID_END)] = np.nan
    return masked.rolling(window=window, min_periods=min_obs).mean()


def _rolling_median_ex_covid(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Rolling median excluding COVID window from baseline."""
    masked = s.copy()
    masked[(masked.index >= COVID_START) & (masked.index <= COVID_END)] = np.nan
    return masked.rolling(window=window, min_periods=min_obs).median()


# ---------------------------------------------------------------------------
# Row collector
# ---------------------------------------------------------------------------

def _rows(currency: str, metric_id: str, s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    if s.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "currency":  currency,
        "series_id": metric_id,
        "time":      s.index,
        "value":     s.values,
    })


# ---------------------------------------------------------------------------
# Build per currency
# ---------------------------------------------------------------------------

def _build_currency(currency: str, cfg: dict, engine, schema: str) -> pd.DataFrame:
    logger.info("Building labour derived for %s...", currency.upper())
    frames = []
    target = cfg["inflation_target"]

    # ── Unemployment rate + changes ───────────────────────────────────────
    raw_u = _load_raw(cfg["unemp_id"], engine, schema)
    unemp_freq = cfg.get("unemp_freq", "M")
    if not raw_u.empty:
        u = _normalise_to_monthly(raw_u, unemp_freq)
        frames.append(_rows(currency, f"{currency}_unemp_rate", u.round(4)))

        # Regression signal: UR changes at 3 windows
        frames.append(_rows(currency, f"{currency}_unemp_3m_chg",  _chg(u, 3)))
        frames.append(_rows(currency, f"{currency}_unemp_6m_chg",  _chg(u, 6)))
        frames.append(_rows(currency, f"{currency}_unemp_12m_chg", _chg(u, 12)))

        # UR gap: UR minus 5Y rolling mean (60M, ex-COVID)
        ur_baseline = _rolling_mean_ex_covid(u, window=60, min_obs=24)
        unemp_gap = (u - ur_baseline).round(4)
        frames.append(_rows(currency, f"{currency}_unemp_gap", unemp_gap))

        logger.info("  ✓ UR: rate=%.2f%%  3m_chg=%.2f  gap=%.2f",
                    u.dropna().iloc[-1],
                    _chg(u, 3).dropna().iloc[-1] if not _chg(u, 3).dropna().empty else float("nan"),
                    unemp_gap.dropna().iloc[-1] if not unemp_gap.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: unemployment rate missing", currency)

    # ── Employment + excess ───────────────────────────────────────────────
    raw_e = _load_raw(cfg["employment_id"], engine, schema)
    emp_monthly = pd.Series(dtype=float)
    emp_yoy = pd.Series(dtype=float)
    if not raw_e.empty:
        emp_monthly = _normalise_to_monthly(raw_e, cfg["employment_freq"])
        emp_yoy = _yoy(emp_monthly)
        frames.append(_rows(currency, f"{currency}_emp_yoy_pct", emp_yoy))

        # Excess employment: emp YoY minus 5Y rolling median (60M, ex-COVID)
        emp_baseline = _rolling_median_ex_covid(emp_yoy, window=60, min_obs=24)
        emp_excess = (emp_yoy - emp_baseline).round(4)
        frames.append(_rows(currency, f"{currency}_emp_excess", emp_excess))

        logger.info("  ✓ employment: yoy=%.2f%%  excess=%.2f",
                    emp_yoy.dropna().iloc[-1] if not emp_yoy.dropna().empty else float("nan"),
                    emp_excess.dropna().iloc[-1] if not emp_excess.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: employment missing", currency)

    # ── Wages ─────────────────────────────────────────────────────────────
    raw_w = _load_raw(cfg["wages_id"], engine, schema)
    wages_yoy_monthly = pd.Series(dtype=float)
    if not raw_w.empty:
        w = _normalise_to_monthly(raw_w, cfg["wages_freq"])

        if cfg["wages_type"] == "pct":
            # Already published as YoY % — store directly
            wages_yoy_monthly = w.round(4)
        else:
            # Index/level — compute YoY
            wages_yoy_monthly = _yoy(w)

        frames.append(_rows(currency, f"{currency}_wages_yoy", wages_yoy_monthly))
        logger.info("  ✓ wages (%s): yoy=%.2f%%",
                    cfg["wages_type"],
                    wages_yoy_monthly.dropna().iloc[-1] if not wages_yoy_monthly.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: wages missing", currency)

    # ── Productivity: GDP / employment → YoY (quarterly) ─────────────────
    raw_gdp = _load_raw(cfg["gdp_real_id"], engine, schema)
    productivity_yoy_monthly = pd.Series(dtype=float)
    if not raw_gdp.empty and not raw_e.empty:
        gdp_q = _to_quarterly(raw_gdp)
        # Quarterly employment: average of monthly values within quarter
        emp_q = _normalise_to_monthly(raw_e, cfg["employment_freq"])
        emp_q = emp_q.resample("QS").mean().dropna()

        combined = pd.DataFrame({"gdp": gdp_q, "emp": emp_q}).dropna()
        if not combined.empty and (combined["emp"] != 0).all():
            productivity = combined["gdp"] / combined["emp"]
            prod_yoy = _yoy_quarterly(productivity)
            frames.append(_rows(currency, f"{currency}_productivity_yoy", prod_yoy))

            # FFill to monthly for excess wages computation
            monthly_idx = pd.date_range(
                prod_yoy.dropna().index[0],
                prod_yoy.dropna().index[-1],
                freq="MS",
            )
            productivity_yoy_monthly = prod_yoy.reindex(monthly_idx, method="ffill")

            logger.info("  ✓ productivity: yoy=%.2f%%",
                        prod_yoy.dropna().iloc[-1] if not prod_yoy.dropna().empty else float("nan"))
        else:
            logger.warning("  %s: productivity — alignment or zero issue", currency)
    else:
        logger.warning("  %s: GDP or employment missing for productivity", currency)

    # ── Excess wages: wages_yoy − (target + productivity_yoy) ─────────────
    if not wages_yoy_monthly.empty and not productivity_yoy_monthly.empty:
        combined = pd.DataFrame({
            "w": wages_yoy_monthly,
            "p": productivity_yoy_monthly,
        }).dropna()
        if not combined.empty:
            excess_wages = (combined["w"] - target - combined["p"]).round(4)
            frames.append(_rows(currency, f"{currency}_excess_wages", excess_wages))
            logger.info("  ✓ excess_wages: latest=%.2f (wages=%.2f - target=%.1f - prod=%.2f)",
                        excess_wages.dropna().iloc[-1] if not excess_wages.dropna().empty else float("nan"),
                        combined["w"].dropna().iloc[-1] if not combined["w"].dropna().empty else float("nan"),
                        target,
                        combined["p"].dropna().iloc[-1] if not combined["p"].dropna().empty else float("nan"))

    # ── Assemble ──────────────────────────────────────────────────────────
    valid = [f for f in frames if not f.empty]
    if not valid:
        return pd.DataFrame()

    result = pd.concat(valid, ignore_index=True)
    result["time"]  = pd.to_datetime(result["time"]).dt.date
    result["value"] = pd.to_numeric(result["value"], errors="coerce").round(4)
    logger.info("  %s: %d rows, %d metrics",
                currency.upper(), len(result), result["series_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(df: pd.DataFrame, db_config: DbConfig, schema: str) -> None:
    if df.empty:
        return

    now  = datetime.now(timezone.utc)
    cols = ["currency", "series_id", "time", "value"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]

    logger.info("Upserting %d rows into %s.labour_derived", len(rows), schema)

    sql = f"""
        INSERT INTO {schema}.labour_derived (
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
        logger.info("Upsert complete: %s.labour_derived rows=%d", schema, len(rows))
    except psycopg2.errors.UndefinedTable:
        logger.error("Table %s.labour_derived does not exist — run setup SQL first.", schema)
        raise
    except Exception:
        logger.exception("Upsert failed: %s.labour_derived", schema)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_and_store_labour_derived(
    db_config: DbConfig,
    schema: str = "macro",
    currencies: list[str] | None = None,
) -> None:
    """
    Compute and store labour derived metrics for all configured currencies.
    """
    configs = CURRENCY_CONFIGS
    if currencies:
        configs = {k: v for k, v in CURRENCY_CONFIGS.items() if k in currencies}

    engine = create_engine(
        f"postgresql://{db_config.user}:{db_config.password}@"
        f"{db_config.host}:{db_config.port}/{db_config.dbname}"
    )

    try:
        all_frames = []
        for currency, cfg in configs.items():
            df = _build_currency(currency, cfg, engine, schema)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            logger.warning("No labour derived data produced")
            return

        combined = pd.concat(all_frames, ignore_index=True)
        _upsert(combined, db_config, schema)
        logger.info(
            "✓ Labour derived complete: %d currencies, %d total rows",
            len(all_frames), len(combined),
        )
    except Exception as e:
        logger.error("Labour derived failed: %s", e, exc_info=True)
        raise
    finally:
        engine.dispose()