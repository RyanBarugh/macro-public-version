"""
Inflation Derived — Unified Multi-Currency
============================================

Computes inflation derived metrics for all 8 currencies.
Writes to macro.inflation_derived (shared table with currency column).

NO scoring, NO z-scores, NO std_scale. Pure math transforms only.
Scoring belongs exclusively in the signals layer.

JPMaQS inflation factors served by this layer:
  F1: Relative excess CPI — core_6m_ann minus inflation_target
  F2: Relative excess producer prices — deflator_yoy + ppi_yoy, both vs target
  F3: Inflation expectations — formulaic blend (built in signals from inputs here)

Output metrics per currency:
  {ccy}_core_6m_ann       — Core CPI 6M/6M annualised %
  {ccy}_core_yoy          — Core CPI YoY %
  {ccy}_headline_yoy      — Headline CPI YoY %
  {ccy}_headline_6m_ann   — Headline CPI 6M/6M annualised %
  {ccy}_ppi_yoy           — PPI YoY %
  {ccy}_deflator_yoy      — GDP deflator YoY % (quarterly frequency)
  {ccy}_ppi_cpi_spread    — PPI YoY minus Headline CPI YoY
  {ccy}_excess_core_cpi   — core_6m_ann minus inflation target
  {ccy}_deflator_excess   — deflator_yoy minus inflation target
  {ccy}_ppi_excess        — ppi_yoy minus inflation target
  {ccy}_infe_excess       — inflation expectations excess over effective target
  {ccy}_real_yield_2y     — 2Y nominal yield minus INFE_2Y (real yield level)

Reads from:  macro.series_data
Writes to:   macro.inflation_derived

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.inflation_derived (
        currency    TEXT          NOT NULL,
        series_id   TEXT          NOT NULL,
        time        DATE          NOT NULL,
        value       FLOAT,
        updated_at  TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_inflation_derived_time
        ON macro.inflation_derived(time);
    CREATE INDEX IF NOT EXISTS idx_inflation_derived_currency
        ON macro.inflation_derived(currency);
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
# Currency configs
# ---------------------------------------------------------------------------

CURRENCY_CONFIGS = {
    "usd": {
        "inflation_target": 2.0,
        "headline_id":      "us_cpi_all_items_sa",
        "core_id":          "us_cpi_core_sa",
        "ppi_id":           "usd_ppi_final_demand_sa",
        "gdp_nominal_id":   "usd_gdp_nominal_bea_sa",
        "gdp_real_id":      "usd_gdp_real_bea_sa",
        "yield_2y_id":      "usd_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
        "ppi_freq":         "M",
    },
    "eur": {
        "inflation_target": 2.0,
        "headline_id":      "eur_hicp_headline_nsa",
        "core_id":          "eur_hicp_core_nsa",
        "ppi_id":           "eur_ppi_total_sts_nsa",
        "gdp_nominal_id":   "eur_gdp_nominal",
        "gdp_real_id":      "eur_gdp_real",
        "yield_2y_id":      "eur_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
        "ppi_freq":         "M",
    },
    "gbp": {
        "inflation_target": 2.0,
        "headline_id":      "gbp_cpi_headline_nsa",
        "core_id":          "gbp_cpi_core_nsa",
        "ppi_id":           "gbp_ppi_total_nsa",
        "gdp_nominal_id":   "gbp_gdp_nominal",
        "gdp_real_id":      "gbp_gdp_real",
        "yield_2y_id":      "gbp_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
        "ppi_freq":         "M",
    },
    "aud": {
        "inflation_target": 2.5,
        "headline_id":      "aud_cpi_headline_m_sa",
        "headline_backfill_id":  "aud_cpi_headline_q_nsa",
        "headline_backfill_freq": "Q",
        "core_id":          "aud_cpi_trimmed_mean_m_sa",
        "core_backfill_id": "aud_cpi_core_q_nsa",
        "core_backfill_freq": "Q",
        "ppi_id":           "aud_ppi_headline_q",
        "gdp_nominal_id":   "aud_gdp_nominal",
        "gdp_real_id":      "aud_gdp_real",
        "yield_2y_id":      "aud_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
        "ppi_freq":         "Q",
    },
    "cad": {
        "inflation_target": 2.0,
        "headline_id":      "cad_cpi_headline_sa",
        "core_id":          "cad_cpi_core_sa",
        "ppi_id":           "cad_ppi_total_nsa",
        "gdp_nominal_id":   "cad_gdp_nominal",
        "gdp_real_id":      "cad_gdp_real",
        "yield_2y_id":      "cad_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
        "ppi_freq":         "M",
    },
    "jpy": {
        "inflation_target": 2.0,
        "headline_id":      "jpy_cpi_headline_sa",
        "core_id":          "jpy_cpi_core_sa",
        "ppi_id":           "jpy_ppi_headline_nsa",
        "gdp_nominal_id":   "jpy_gdp_nominal",
        "gdp_real_id":      "jpy_gdp_real",
        "yield_2y_id":      "jpy_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
        "ppi_freq":         "M",
    },
    "nzd": {
        "inflation_target": 2.0,
        "headline_id":      "nzd_cpi_headline_sa",
        "core_id":          "nzd_cpi_nontradable_sa",
        "ppi_id":           "nzd_ppi_outputs_nsa",
        "gdp_nominal_id":   "nzd_gdp_nominal_sa",
        "gdp_real_id":      "nzd_gdp_real_sa",
        "yield_2y_id":      "nzd_yield_2y",
        "headline_freq":    "Q",
        "core_freq":        "Q",
        "ppi_freq":         "Q",
    },
    "chf": {
        "inflation_target": 2.0,
        "headline_id":      "chf_cpi_headline_nsa",
        "core_id":          "chf_cpi_core_sa",
        "core_type":        "pct",       # FSO publishes core CPI as YoY %, not index
        "ppi_id":           "chf_ppi_headline_nsa",
        "gdp_nominal_id":   "chf_gdp_nominal_eurostat",
        "gdp_real_id":      "chf_gdp_real_eurostat",
        "yield_2y_id":      "chf_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
        "ppi_freq":         "M",
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
    """Normalise to month-start, deduplicate (keep last)."""
    s = s.copy()
    s.index = s.index.to_period("M").to_timestamp()
    return s.groupby(level=0).last().sort_index()


def _quarterly_to_monthly(s: pd.Series) -> pd.Series:
    """Normalise quarterly to month-start then ffill to monthly."""
    s = s.copy()
    s.index = s.index.to_period("Q").to_timestamp()
    s = s.groupby(level=0).last().sort_index()
    monthly_idx = pd.date_range(s.index[0], s.index[-1], freq="MS")
    return s.reindex(monthly_idx, method="ffill")


def _to_quarterly(s: pd.Series) -> pd.Series:
    """Normalise to quarter-start, deduplicate (keep last)."""
    s = s.copy()
    s.index = s.index.to_period("Q").to_timestamp()
    return s.groupby(level=0).last().sort_index()


def _normalise_to_monthly(s: pd.Series, freq: str) -> pd.Series:
    """Route to correct normaliser based on declared frequency."""
    if freq == "Q":
        return _quarterly_to_monthly(s)
    return _to_monthly(s)


def _splice_series(
    primary: pd.Series,
    backfill: pd.Series,
    primary_freq: str,
    backfill_freq: str,
) -> pd.Series:
    """
    Splice two series: use primary where available, backfill for earlier history.

    Both are normalised to monthly. Primary takes precedence in overlap.
    Growth rates are base-invariant so different index bases (2015=100 vs 2020=100)
    don't matter — we never use the spliced levels directly, only YoY/6M ann.
    """
    p = _normalise_to_monthly(primary, primary_freq)
    b = _normalise_to_monthly(backfill, backfill_freq)
    # Use backfill for dates before primary starts, primary for everything else
    cutoff = p.dropna().index[0]
    before = b[b.index < cutoff]
    return pd.concat([before, p]).sort_index()


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def _ann_6m(s: pd.Series) -> pd.Series:
    """6M/6M annualised: ((s_t / s_{t-6})^2 - 1) × 100."""
    ratio = s / s.shift(6)
    return ((ratio ** 2) - 1.0).mul(100.0).round(4)


def _ann_3m(s: pd.Series) -> pd.Series:
    """3M/3M annualised: ((s_t / s_{t-3})^4 - 1) × 100."""
    ratio = s / s.shift(3)
    return ((ratio ** 4) - 1.0).mul(100.0).round(4)


def _yoy(s: pd.Series) -> pd.Series:
    """YoY: (s_t / s_{t-12} - 1) × 100  (monthly data)."""
    return ((s / s.shift(12)) - 1.0).mul(100.0).round(4)


def _yoy_quarterly(s: pd.Series) -> pd.Series:
    """YoY via shift(4) for quarterly data."""
    return ((s / s.shift(4)) - 1.0).mul(100.0).round(4)


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
    logger.info("Building inflation derived for %s...", currency.upper())
    frames = []
    target = cfg["inflation_target"]

    # ── Core CPI ──────────────────────────────────────────────────────────
    raw_c = _load_raw(cfg["core_id"], engine, schema)
    core_6m = pd.Series(dtype=float)
    if not raw_c.empty:
        # Splice with quarterly backfill if configured (e.g. AUD)
        if cfg.get("core_backfill_id"):
            raw_cb = _load_raw(cfg["core_backfill_id"], engine, schema)
            if not raw_cb.empty:
                c = _splice_series(raw_c, raw_cb, cfg["core_freq"], cfg["core_backfill_freq"])
                logger.info("  ✓ Core spliced: backfill %s → primary %s",
                            cfg["core_backfill_id"], cfg["core_id"])
            else:
                c = _normalise_to_monthly(raw_c, cfg["core_freq"])
        else:
            c = _normalise_to_monthly(raw_c, cfg["core_freq"])

        if cfg.get("core_type") == "pct":
            # Already published as YoY % (e.g. CHF FSO) — use directly
            core_yoy = c.round(4)
            core_6m = c.round(4)
            core_3m = c.round(4)
        else:
            # Index level — compute transforms
            core_6m = _ann_6m(c)
            core_3m = _ann_3m(c)
            core_yoy = _yoy(c)

        frames.append(_rows(currency, f"{currency}_core_6m_ann", core_6m))
        frames.append(_rows(currency, f"{currency}_core_3m_ann", core_3m))
        frames.append(_rows(currency, f"{currency}_core_yoy", core_yoy))
        excess_core = (core_6m - target).round(4)
        frames.append(_rows(currency, f"{currency}_excess_core_cpi", excess_core))

        # Core acceleration: 3M ann minus 6M ann
        # Positive = inflation accelerating, negative = decelerating
        core_accel = (core_3m - core_6m).round(4)
        frames.append(_rows(currency, f"{currency}_core_acceleration", core_accel))

        logger.info("  ✓ core: 6m_ann=%.2f%%  3m_ann=%.2f%%  accel=%.2f  excess=%.2f",
                    core_6m.dropna().iloc[-1] if not core_6m.dropna().empty else float("nan"),
                    core_3m.dropna().iloc[-1] if not core_3m.dropna().empty else float("nan"),
                    core_accel.dropna().iloc[-1] if not core_accel.dropna().empty else float("nan"),
                    excess_core.dropna().iloc[-1] if not excess_core.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: core CPI missing", currency)

    # ── Headline CPI ──────────────────────────────────────────────────────
    raw_h = _load_raw(cfg["headline_id"], engine, schema)
    headline_6m = pd.Series(dtype=float)
    headline_yoy = pd.Series(dtype=float)
    if not raw_h.empty:
        # Splice with quarterly backfill if configured (e.g. AUD)
        if cfg.get("headline_backfill_id"):
            raw_hb = _load_raw(cfg["headline_backfill_id"], engine, schema)
            if not raw_hb.empty:
                h = _splice_series(raw_h, raw_hb, cfg["headline_freq"], cfg["headline_backfill_freq"])
                logger.info("  ✓ Headline spliced: backfill %s → primary %s",
                            cfg["headline_backfill_id"], cfg["headline_id"])
            else:
                h = _normalise_to_monthly(raw_h, cfg["headline_freq"])
        else:
            h = _normalise_to_monthly(raw_h, cfg["headline_freq"])
        headline_6m = _ann_6m(h)
        headline_yoy = _yoy(h)
        frames.append(_rows(currency, f"{currency}_headline_yoy", headline_yoy))
        frames.append(_rows(currency, f"{currency}_headline_6m_ann", headline_6m))
        logger.info("  ✓ headline: yoy=%.2f%%",
                    headline_yoy.dropna().iloc[-1] if not headline_yoy.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: headline CPI missing", currency)

    # ── PPI ───────────────────────────────────────────────────────────────
    raw_p = _load_raw(cfg["ppi_id"], engine, schema)
    ppi_yoy = pd.Series(dtype=float)
    if not raw_p.empty:
        p = _normalise_to_monthly(raw_p, cfg["ppi_freq"])
        ppi_yoy = _yoy(p)
        frames.append(_rows(currency, f"{currency}_ppi_yoy", ppi_yoy))
        ppi_excess = (ppi_yoy - target).round(4)
        frames.append(_rows(currency, f"{currency}_ppi_excess", ppi_excess))
        logger.info("  ✓ ppi: yoy=%.2f%%  excess=%.2f",
                    ppi_yoy.dropna().iloc[-1] if not ppi_yoy.dropna().empty else float("nan"),
                    ppi_excess.dropna().iloc[-1] if not ppi_excess.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: PPI missing", currency)

    # ── PPI-CPI spread ────────────────────────────────────────────────────
    if not ppi_yoy.empty and not headline_yoy.empty:
        combined = pd.DataFrame({"ppi": ppi_yoy, "cpi": headline_yoy}).dropna()
        if not combined.empty:
            spread = (combined["ppi"] - combined["cpi"]).round(4)
            frames.append(_rows(currency, f"{currency}_ppi_cpi_spread", spread))
            logger.info("  ✓ ppi_cpi_spread: latest=%.2f pp", spread.dropna().iloc[-1])

    # ── Inflation Expectations (JPMaQS formulaic construction) ────────
    # INFE = blend of CPI trend + effective target at 1Y/2Y/5Y horizons
    # excess = mean(INFE_1Y, INFE_2Y, INFE_5Y) − effective_target
    infe_2y_monthly = pd.Series(dtype=float)
    if not core_6m.empty and not headline_6m.empty and not headline_yoy.empty:
        infe_df = pd.DataFrame({
            "h6m": headline_6m,
            "c6m": core_6m,
            "hyoy": headline_yoy,
        }).dropna(subset=["h6m", "c6m"])

        if not infe_df.empty:
            trend = (infe_df["h6m"] + infe_df["c6m"]) / 2
            cpi_gap = (infe_df["hyoy"] - target).dropna()
            cred_adj = cpi_gap.rolling(36, min_periods=12).mean()
            eff_target = target + cred_adj.reindex(trend.index).ffill().fillna(0.0)

            infe_1y = 0.75 * trend + 0.25 * eff_target
            infe_2y = 0.50 * trend + 0.50 * eff_target
            infe_5y = 0.25 * trend + 0.75 * eff_target

            infe_2y_monthly = infe_2y  # keep for real yield computation

            infe_excess = ((infe_1y + infe_2y + infe_5y) / 3 - eff_target).round(4)

            frames.append(_rows(currency, f"{currency}_infe_excess", infe_excess))
            logger.info("  ✓ infe_excess: latest=%.2f pp",
                        infe_excess.dropna().iloc[-1] if not infe_excess.dropna().empty else float("nan"))
        else:
            logger.warning("  %s: INFE — insufficient CPI data", currency)
    else:
        logger.warning("  %s: INFE — missing core/headline 6M or headline YoY", currency)

    # ── GDP Deflator (quarterly) ──────────────────────────────────────────
    raw_nom = _load_raw(cfg["gdp_nominal_id"], engine, schema)
    raw_real = _load_raw(cfg["gdp_real_id"], engine, schema)
    if not raw_nom.empty and not raw_real.empty:
        nom = _to_quarterly(raw_nom)
        real = _to_quarterly(raw_real)
        combined = pd.DataFrame({"nom": nom, "real": real}).dropna()
        if not combined.empty and (combined["real"] != 0).all():
            deflator = (combined["nom"] / combined["real"] * 100.0).round(4)
            deflator_yoy = _yoy_quarterly(deflator)
            frames.append(_rows(currency, f"{currency}_deflator_yoy", deflator_yoy))
            deflator_excess = (deflator_yoy - target).round(4)
            frames.append(_rows(currency, f"{currency}_deflator_excess", deflator_excess))
            logger.info("  ✓ deflator: yoy=%.2f%%  excess=%.2f",
                        deflator_yoy.dropna().iloc[-1] if not deflator_yoy.dropna().empty else float("nan"),
                        deflator_excess.dropna().iloc[-1] if not deflator_excess.dropna().empty else float("nan"))
        else:
            logger.warning("  %s: GDP deflator — alignment or zero issue", currency)
    else:
        logger.warning("  %s: GDP nominal/real missing for deflator", currency)

    # ── Real 2Y yield: nominal yield minus INFE_2Y ───────────────────────
    # Daily yield sampled at month-end, minus monthly INFE_2Y.
    # Scored on level (neutral="zero"): higher real yield = tighter = bullish.
    yield_id = cfg.get("yield_2y_id")
    if yield_id and not infe_2y_monthly.empty:
        raw_y = _load_raw(yield_id, engine, schema)
        if not raw_y.empty:
            # Sample yield at month-end to align with monthly INFE
            y_monthly = _to_monthly(raw_y)
            combined = pd.DataFrame({
                "yield": y_monthly,
                "infe": infe_2y_monthly,
            }).dropna()
            if not combined.empty:
                real_yield = (combined["yield"] - combined["infe"]).round(4)
                frames.append(_rows(currency, f"{currency}_real_yield_2y", real_yield))
                logger.info("  ✓ real_yield_2y: %.2f%% (nom=%.2f - infe=%.2f)",
                            real_yield.dropna().iloc[-1] if not real_yield.dropna().empty else float("nan"),
                            combined["yield"].dropna().iloc[-1] if not combined["yield"].dropna().empty else float("nan"),
                            combined["infe"].dropna().iloc[-1] if not combined["infe"].dropna().empty else float("nan"))
            else:
                logger.warning("  %s: real yield — no overlapping dates", currency)
        else:
            logger.warning("  %s: 2Y yield missing (%s)", currency, yield_id)
    elif yield_id:
        logger.warning("  %s: real yield — INFE_2Y not available", currency)

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

    logger.info("Upserting %d rows into %s.inflation_derived", len(rows), schema)

    sql = f"""
        INSERT INTO {schema}.inflation_derived (
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
        logger.info("Upsert complete: %s.inflation_derived rows=%d", schema, len(rows))
    except psycopg2.errors.UndefinedTable:
        logger.error(
            "Table %s.inflation_derived does not exist — run setup SQL first.", schema
        )
        raise
    except Exception:
        logger.exception("Upsert failed: %s.inflation_derived", schema)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_and_store_inflation_derived(
    db_config: DbConfig,
    schema: str = "macro",
    currencies: list[str] | None = None,
) -> None:
    """
    Compute and store inflation derived metrics for all configured currencies.

    Args:
        currencies: Optional list to run a subset e.g. ['gbp', 'aud'].
                    Defaults to all currencies in CURRENCY_CONFIGS.
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
            logger.warning("No inflation derived data produced")
            return

        combined = pd.concat(all_frames, ignore_index=True)
        _upsert(combined, db_config, schema)
        logger.info(
            "✓ Inflation derived complete: %d currencies, %d total rows",
            len(all_frames),
            len(combined),
        )
    except Exception as e:
        logger.error("Inflation derived failed: %s", e, exc_info=True)
        raise
    finally:
        engine.dispose()