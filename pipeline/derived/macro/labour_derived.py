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
  {ccy}_wages_yoy_3mma      — Wage growth YoY % 3-month moving average
  {ccy}_excess_wages        — wages_3mma minus LPGT minus INFTEFF (JPMaQS)
  {ccy}_infteff             — Effective inflation target (JPMaQS credibility-adjusted)
  {ccy}_lpgt                — Trend labour productivity growth (GDP trend − workforce trend)

Reads from:  macro.series_data
Writes to:   macro.labour_derived

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.labour_derived (
        currency                TEXT          NOT NULL,
        series_id               TEXT          NOT NULL,
        time                    DATE          NOT NULL,
        value                   FLOAT,
        estimated_release_date  DATE,
        updated_at              TIMESTAMPTZ   DEFAULT NOW(),
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

from ...engine.logger import get_logger
from ...engine.release_dates import get_release_date_mapper

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# COVID exclusion window for rolling baselines
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Currency configs
# ---------------------------------------------------------------------------

CURRENCY_CONFIGS = {
    "usd": {
        "inflation_target":  2.0,
        "target_start_date": "1996-01-01",   # Fed implicit 2% target ~1996
        "unemp_id":          "usd_unemployment_rate_sa",
        "employment_id":     "usd_nfp_level_sa",
        "employment_freq":   "M",
        "wages_id":          "usd_wage_ahe_sa",
        "wages_freq":        "M",
        "wages_type":        "index",
        "gdp_real_id":       "usd_gdp_real_bea_sa",
        "labour_force_id":   "usd_labour_force_level",
        "labour_force_freq": "M",
        "headline_id":       "us_cpi_all_items_sa",
        "core_id":           "us_cpi_core_sa",
        "headline_freq":     "M",
        "core_freq":         "M",
    },
    "eur": {
        "inflation_target":  2.0,
        "target_start_date": "1999-01-01",   # ECB mandate from inception
        "unemp_id":          "eur_unemployment_rate_sa",
        "employment_id":     "eur_emp_level_lfs_sa",
        "employment_freq":   "Q",
        "wages_id":          "eur_negotiated_wages_qoq",   # published as YoY %
        "wages_freq":        "Q",
        "wages_type":        "pct",
        "gdp_real_id":       "eur_gdp_real",
        "labour_force_id":   "eur_labour_force_level_nsa",
        "labour_force_freq": "Q",
        "headline_id":       "eur_hicp_headline_nsa",
        "core_id":           "eur_hicp_core_nsa",
        "headline_freq":     "M",
        "core_freq":         "M",
    },
    "gbp": {
        "inflation_target":  2.0,
        "target_start_date": "1992-10-01",   # BoE inflation targeting began
        "unemp_id":          "gbp_unemployment_rate_sa",
        "employment_id":     "gbp_employment_level_sa",
        "employment_freq":   "M",
        "wages_id":          "gbp_wages_regular_sa",       # published as YoY %
        "wages_freq":        "M",
        "wages_type":        "pct",
        "gdp_real_id":       "gbp_gdp_real",
        "labour_force_id":   "gbp_labour_force_level_sa",
        "labour_force_freq": "Q",
        "headline_id":       "gbp_cpi_headline_nsa",
        "core_id":           "gbp_cpi_core_nsa",
        "headline_freq":     "M",
        "core_freq":         "M",
    },
    "aud": {
        "inflation_target":  2.5,
        "target_start_date": "1993-04-01",   # RBA target band 2-3%, midpoint 2.5
        "unemp_id":          "aud_unemployment_rate_sa",
        "employment_id":     "aud_employment_level_sa",
        "employment_freq":   "M",
        "wages_id":          "aud_wages_wpi_index_sa",
        "wages_freq":        "Q",
        "wages_type":        "index",
        "gdp_real_id":       "aud_gdp_real",
        "labour_force_id":   "aud_labour_force_level_sa",
        "labour_force_freq": "M",
        "headline_id":       "aud_cpi_headline_m_sa",
        "headline_backfill_id":  "aud_cpi_headline_q_nsa",
        "headline_backfill_freq": "Q",
        "core_id":           "aud_cpi_trimmed_mean_m_sa",
        "core_backfill_id":  "aud_cpi_core_q_nsa",
        "core_backfill_freq": "Q",
        "headline_freq":     "M",
        "core_freq":         "M",
    },
    "cad": {
        "inflation_target":  2.0,
        "target_start_date": "1991-02-01",   # BoC/Govt joint target
        "unemp_id":          "cad_unemployment_rate",
        "employment_id":     "cad_employment_level",
        "employment_freq":   "M",
        "wages_id":          "cad_avg_hourly_wage",
        "wages_freq":        "M",
        "wages_type":        "index",
        "gdp_real_id":       "cad_gdp_real",
        "labour_force_id":   "cad_labour_force_level_sa",
        "labour_force_freq": "M",
        "headline_id":       "cad_cpi_headline_sa",
        "core_id":           "cad_cpi_core_sa",
        "headline_freq":     "M",
        "core_freq":         "M",
    },
    "jpy": {
        "inflation_target":  2.0,
        "target_start_date": "2013-01-22",   # BoJ formal 2% adoption
        "unemp_id":          "jpy_unemployment_rate_nsa",
        "employment_id":     "jpy_employment_level_nsa",
        "employment_freq":   "M",
        "wages_id":          "jpy_wages_manufacturing_sa",
        "wages_freq":        "M",
        "wages_type":        "index",
        "gdp_real_id":       "jpy_gdp_real",
        "labour_force_id":   "jpy_labour_force_level_sa",
        "labour_force_freq": "M",
        "headline_id":       "jpy_cpi_headline_sa",
        "core_id":           "jpy_cpi_core_sa",
        "headline_freq":     "M",
        "core_freq":         "M",
    },
    "nzd": {
        "inflation_target":  2.0,
        "target_start_date": "1990-03-01",   # RBNZ — first CB to adopt IT
        "unemp_id":          "nzd_unemployment_rate_sa",
        "unemp_freq":        "Q",       # NZD UR is quarterly
        "employment_id":     "nzd_employment_level_sa",
        "employment_freq":   "Q",
        "wages_id":          "nzd_avg_hourly_earnings_nsa",
        "wages_freq":        "Q",
        "wages_type":        "index",
        "gdp_real_id":       "nzd_gdp_real_sa",
        "labour_force_id":   "nzd_labour_force_level_sa",
        "labour_force_freq": "Q",
        "headline_id":       "nzd_cpi_headline_sa",
        "core_id":           "nzd_cpi_nontradable_sa",
        "headline_freq":     "Q",
        "core_freq":         "Q",
    },
    "chf": {
        "inflation_target":  None,           # SNB "below 2%" — no point target
        "target_start_date": None,           # pro-forma throughout
        "unemp_id":          "chf_unemployment_rate_nsa",
        "employment_id":     "chf_employment_level_sa",
        "employment_freq":   "Q",
        "wages_id":          "chf_wages_nominal_yoy",      # published as YoY %
        "wages_freq":        "Q",
        "wages_type":        "pct",
        "gdp_real_id":       "chf_gdp_real",
        "labour_force_id":   "chf_labour_force_level_sa",
        "labour_force_freq": "Q",
        "headline_id":       "chf_cpi_headline_nsa",
        "core_id":           "chf_cpi_core_sa",
        "core_type":         "pct",          # FSO publishes core as YoY %
        "headline_freq":     "M",
        "core_freq":         "M",
    },
}


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


# ---------------------------------------------------------------------------
# Frequency normalisation
# ---------------------------------------------------------------------------

def _to_monthly(s: pd.Series) -> pd.Series:
    """Normalise to month-end, deduplicate (keep last)."""
    s = s.copy()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s.groupby(level=0).last().sort_index()

def _quarterly_to_monthly(s):
    s = s.copy()
    s.index = s.index.to_period("Q").to_timestamp("Q")   # quarter's month-end
    s = s.groupby(level=0).last().sort_index()
    monthly_idx = pd.date_range(s.index[0], s.index[-1], freq="ME")
    return s.reindex(monthly_idx, method="ffill")

def _to_quarterly(s):
    s = s.copy()
    s.index = s.index.to_period("Q").to_timestamp("Q")   # quarter's month-end
    return s.groupby(level=0).last().sort_index()

def _normalise_to_monthly(s: pd.Series, freq: str) -> pd.Series:
    if freq == "Q":
        return _quarterly_to_monthly(s)
    return _to_monthly(s)


def _splice_series(
    primary: pd.Series,
    backfill: pd.Series,
    primary_freq: str,
    backfill_freq: str,
) -> tuple[pd.Series, pd.Timestamp]:
    """
    Splice two series: use primary where available, backfill for earlier history.
    Both normalised to monthly. Primary takes precedence in overlap.
    Returns (spliced_series, cutoff) where cutoff is the first date of the primary.
    """
    p = _normalise_to_monthly(primary, primary_freq)
    b = _normalise_to_monthly(backfill, backfill_freq)
    cutoff = p.dropna().index[0]
    before = b[b.index < cutoff]
    return pd.concat([before, p]).sort_index(), cutoff


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

# ---------------------------------------------------------------------------
# INFTEFF — effective inflation target (JPMaQS-aligned)
# ---------------------------------------------------------------------------

GLOBAL_BENCHMARK = 2.0   # JPMaQS pro-forma anchor for countries without target


def _compute_infteff(
    headline_yoy: pd.Series,
    core_yoy: pd.Series,
    inflation_target: float | None,
    target_start_date: str | None,
) -> pd.Series:
    """
    Compute the effective inflation target (INFTEFF) per JPMaQS methodology.

    Pro-forma for all currencies:
        INFTEFF = (3Y_median(headline_yoy) + 3Y_median(core_yoy) + 2.0) / 3

    Uses medians for outlier robustness and a 2.0% global anchor for stability.
    """
    headline_3y_median = headline_yoy.rolling(36, min_periods=12).median()
    core_3y_median = core_yoy.rolling(36, min_periods=12).median()
    return ((headline_3y_median + core_3y_median + GLOBAL_BENCHMARK) / 3.0).round(4)


# ---------------------------------------------------------------------------
# Release date helpers
# ---------------------------------------------------------------------------

def _rd(idx, mapper, series_id):
    """Build release-date Series for a single raw input."""
    return pd.Series([mapper(series_id, t) for t in idx], index=idx)


def _rd_multi(rd_series_list):
    """
    Element-wise max of multiple release-date Series (multi-input metric).
    """
    combined = pd.concat(rd_series_list, axis=1)
    return combined.max(axis=1)


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
        d["estimated_release_date"] = rd.reindex(s.index).values
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# Build per currency
# ---------------------------------------------------------------------------

def _build_currency(
    currency: str, cfg: dict, conn, schema: str, mapper,
) -> pd.DataFrame:
    logger.info("Building labour derived for %s...", currency.upper())
    frames = []
    target = cfg["inflation_target"]

    # ── INFTEFF — effective inflation target (JPMaQS) ─────────────────────
    infteff_monthly = pd.Series(dtype=float)
    rd_infteff = pd.Series(dtype="object")

    raw_headline = _load_raw(cfg["headline_id"], conn, schema)
    raw_core = _load_raw(cfg["core_id"], conn, schema)

    if not raw_headline.empty:
        # Splice headline with quarterly backfill if configured (e.g. AUD)
        headline_splice_cutoff = None
        if cfg.get("headline_backfill_id"):
            raw_hb = _load_raw(cfg["headline_backfill_id"], conn, schema)
            if not raw_hb.empty:
                headline, headline_splice_cutoff = _splice_series(
                    raw_headline, raw_hb,
                    cfg["headline_freq"], cfg["headline_backfill_freq"],
                )
                logger.info("  ✓ INFTEFF headline spliced: backfill %s → primary %s",
                            cfg["headline_backfill_id"], cfg["headline_id"])
            else:
                headline = _normalise_to_monthly(raw_headline, cfg["headline_freq"])
        else:
            headline = _normalise_to_monthly(raw_headline, cfg["headline_freq"])

        headline_yoy = _yoy(headline)
        rd_headline_cpi = _rd(headline.index, mapper, cfg["headline_id"])

        # Core CPI: handle splice + pct-type (CHF publishes core as YoY % directly)
        core_yoy = pd.Series(dtype=float)
        rd_core_cpi = pd.Series(dtype="object")
        if not raw_core.empty:
            # Splice core with quarterly backfill if configured (e.g. AUD)
            if cfg.get("core_backfill_id"):
                raw_cb = _load_raw(cfg["core_backfill_id"], conn, schema)
                if not raw_cb.empty:
                    core, _ = _splice_series(
                        raw_core, raw_cb,
                        cfg["core_freq"], cfg["core_backfill_freq"],
                    )
                    logger.info("  ✓ INFTEFF core spliced: backfill %s → primary %s",
                                cfg["core_backfill_id"], cfg["core_id"])
                else:
                    core = _normalise_to_monthly(raw_core, cfg["core_freq"])
            else:
                core = _normalise_to_monthly(raw_core, cfg["core_freq"])

            rd_core_cpi = _rd(core.index, mapper, cfg["core_id"])
            if cfg.get("core_type") == "pct":
                core_yoy = core.round(4)
            else:
                core_yoy = _yoy(core)

        # Compute INFTEFF
        infteff_monthly = _compute_infteff(
            headline_yoy=headline_yoy,
            core_yoy=core_yoy if not core_yoy.empty else headline_yoy,
            inflation_target=target,
            target_start_date=cfg.get("target_start_date"),
        )

        # Release date: gated on headline CPI (+ core for pro-forma currencies)
        if target is None and not rd_core_cpi.empty:
            rd_infteff = _rd_multi([rd_headline_cpi, rd_core_cpi])
        else:
            rd_infteff = rd_headline_cpi

        frames.append(_rows(currency, f"{currency}_infteff", infteff_monthly, rd_infteff))

        latest = infteff_monthly.dropna().iloc[-1] if not infteff_monthly.dropna().empty else float("nan")
        path = "pro-forma" if target is None else "adjusted"
        logger.info("  ✓ infteff (%s): %.2f%%", path, latest)
    else:
        logger.warning("  %s: headline CPI missing — cannot compute INFTEFF", currency)

    # ── LPGT — trend labour productivity growth (JPMaQS) ──────────────────
    # LPGT = 20Q_rolling_median(GDP_yoy) − 20Q_rolling_median(workforce_yoy)
    # Both computed at quarterly frequency, then ffilled to monthly.
    lpgt_monthly = pd.Series(dtype=float)
    wf_trend_monthly = pd.Series(dtype=float)
    rd_lpgt = pd.Series(dtype="object")

    raw_gdp = _load_raw(cfg["gdp_real_id"], conn, schema)
    raw_wf = _load_raw(cfg.get("labour_force_id", ""), conn, schema)

    if not raw_gdp.empty and not raw_wf.empty:
        # GDP: already quarterly
        gdp_q = _to_quarterly(raw_gdp)
        gdp_yoy_q = _yoy_quarterly(gdp_q)
        gdp_trend = gdp_yoy_q.rolling(window=20, min_periods=8).median()

        # Workforce: normalise to quarterly (monthly → end-of-quarter, quarterly → as-is)
        if cfg.get("labour_force_freq") == "M":
            wf_monthly = _to_monthly(raw_wf)
            wf_q = wf_monthly.resample("QE").last().dropna()
        else:
            wf_q = _to_quarterly(raw_wf)
        wf_yoy_q = _yoy_quarterly(wf_q)
        wf_trend = wf_yoy_q.rolling(window=20, min_periods=8).median()

        # LPGT at quarterly frequency
        lpgt_q = (gdp_trend - wf_trend).round(4)

        # FFill to monthly for use in excess_wages and emp_excess
        lpgt_valid = lpgt_q.dropna()
        if not lpgt_valid.empty:
            monthly_idx = pd.date_range(
                lpgt_valid.index[0], lpgt_valid.index[-1], freq="ME",
            )
            lpgt_monthly = lpgt_q.reindex(monthly_idx, method="ffill")

            # Workforce trend ffilled to monthly (for emp_excess anchor)
            wf_trend_monthly = wf_trend.reindex(monthly_idx, method="ffill")

            # Release date: gated on GDP (slower publisher) and workforce
            rd_gdp = _rd(gdp_q.index, mapper, cfg["gdp_real_id"])
            rd_wf = _rd(wf_q.index, mapper, cfg["labour_force_id"])
            rd_lpgt_q = _rd_multi([rd_gdp, rd_wf])
            rd_lpgt = rd_lpgt_q.reindex(monthly_idx, method="ffill")

            frames.append(_rows(currency, f"{currency}_lpgt", lpgt_monthly, rd_lpgt))

            logger.info("  ✓ lpgt: %.2f%% (gdp_trend=%.2f, wf_trend=%.2f)",
                        lpgt_monthly.dropna().iloc[-1],
                        gdp_trend.dropna().iloc[-1] if not gdp_trend.dropna().empty else float("nan"),
                        wf_trend.dropna().iloc[-1] if not wf_trend.dropna().empty else float("nan"))
        else:
            logger.warning("  %s: LPGT — not enough history after rolling median", currency)
    else:
        missing = []
        if raw_gdp.empty:
            missing.append("GDP")
        if raw_wf.empty:
            missing.append("labour force")
        logger.warning("  %s: LPGT skipped — missing %s", currency, " + ".join(missing))

    # ── Unemployment rate + changes ───────────────────────────────────────
    raw_u = _load_raw(cfg["unemp_id"], conn, schema)
    unemp_freq = cfg.get("unemp_freq", "M")
    if not raw_u.empty:
        u = _normalise_to_monthly(raw_u, unemp_freq)
        rd_unemp = _rd(u.index, mapper, cfg["unemp_id"])

        frames.append(_rows(currency, f"{currency}_unemp_rate", u.round(4), rd_unemp))

        # Regression signal: UR changes at 3 windows
        frames.append(_rows(currency, f"{currency}_unemp_3m_chg",  _chg(u, 3), rd_unemp))
        frames.append(_rows(currency, f"{currency}_unemp_6m_chg",  _chg(u, 6), rd_unemp))
        frames.append(_rows(currency, f"{currency}_unemp_12m_chg", _chg(u, 12), rd_unemp))

        # UR gap: 3MMA(UR) minus 5Y rolling mean (60M, ex-COVID) — JPMaQS convention
        u_3mma = u.rolling(3, min_periods=2).mean()
        ur_baseline = u_3mma.rolling(window=60, min_periods=24).median()
        unemp_gap = (u_3mma - ur_baseline).round(4)
        frames.append(_rows(currency, f"{currency}_unemp_gap", unemp_gap, rd_unemp))

        logger.info("  ✓ UR: rate=%.2f%%  3m_chg=%.2f  gap=%.2f",
                    u.dropna().iloc[-1],
                    _chg(u, 3).dropna().iloc[-1] if not _chg(u, 3).dropna().empty else float("nan"),
                    unemp_gap.dropna().iloc[-1] if not unemp_gap.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: unemployment rate missing", currency)

    # ── Employment + excess ───────────────────────────────────────────────
    raw_e = _load_raw(cfg["employment_id"], conn, schema)
    emp_monthly = pd.Series(dtype=float)
    emp_yoy = pd.Series(dtype=float)
    rd_emp = pd.Series(dtype="object")

    if not raw_e.empty:
        emp_monthly = _normalise_to_monthly(raw_e, cfg["employment_freq"])
        rd_emp = _rd(emp_monthly.index, mapper, cfg["employment_id"])

        emp_yoy = _yoy(emp_monthly)
        frames.append(_rows(currency, f"{currency}_emp_yoy_pct", emp_yoy, rd_emp))

        # Excess employment: 3MMA(emp YoY) minus workforce trend (JPMaQS)
        emp_yoy_3mma = emp_yoy.rolling(3, min_periods=2).mean()
        if not wf_trend_monthly.empty:
            combined_emp = pd.DataFrame({
                "emp": emp_yoy_3mma,
                "wf": wf_trend_monthly,
            }).dropna()
            if not combined_emp.empty:
                emp_excess = (combined_emp["emp"] - combined_emp["wf"]).round(4)
                frames.append(_rows(currency, f"{currency}_emp_excess", emp_excess, rd_emp))
                logger.info("  ✓ employment: yoy=%.2f%%  excess=%.2f (wf_trend=%.2f)",
                            emp_yoy.dropna().iloc[-1] if not emp_yoy.dropna().empty else float("nan"),
                            emp_excess.dropna().iloc[-1],
                            combined_emp["wf"].dropna().iloc[-1])
            else:
                logger.warning("  %s: emp_excess — no overlap between emp_yoy and wf_trend", currency)
        else:
            logger.warning("  %s: emp_excess skipped — workforce trend not available", currency)
    else:
        logger.warning("  %s: employment missing", currency)

    # ── Wages ─────────────────────────────────────────────────────────────
    raw_w = _load_raw(cfg["wages_id"], conn, schema)
    wages_yoy_monthly = pd.Series(dtype=float)
    wages_yoy_3mma = pd.Series(dtype=float)
    rd_wages = pd.Series(dtype="object")

    if not raw_w.empty:
        w = _normalise_to_monthly(raw_w, cfg["wages_freq"])
        rd_wages = _rd(w.index, mapper, cfg["wages_id"])

        if cfg["wages_type"] == "pct":
            # Already published as YoY % — store directly
            wages_yoy_monthly = w.round(4)
        else:
            # Index/level — compute YoY
            wages_yoy_monthly = _yoy(w)

        frames.append(_rows(currency, f"{currency}_wages_yoy", wages_yoy_monthly, rd_wages))

        # 3-month moving average for excess wages (JPMaQS convention)
        wages_yoy_3mma = wages_yoy_monthly.rolling(3, min_periods=2).mean().round(4)
        frames.append(_rows(currency, f"{currency}_wages_yoy_3mma", wages_yoy_3mma, rd_wages))

        logger.info("  ✓ wages (%s): yoy=%.2f%%  3mma=%.2f%%",
                    cfg["wages_type"],
                    wages_yoy_monthly.dropna().iloc[-1] if not wages_yoy_monthly.dropna().empty else float("nan"),
                    wages_yoy_3mma.dropna().iloc[-1] if not wages_yoy_3mma.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: wages missing", currency)

    # ── Excess wages (JPMaQS): wages_3mma − LPGT − INFTEFF ──────────────
    if not wages_yoy_3mma.empty and not lpgt_monthly.empty and not infteff_monthly.empty:
        combined = pd.DataFrame({
            "w": wages_yoy_3mma,
            "lpgt": lpgt_monthly,
            "infteff": infteff_monthly,
        }).dropna()
        if not combined.empty:
            excess_wages = (combined["w"] - combined["lpgt"] - combined["infteff"]).round(4)

            # Release date: gated on wages, GDP (via LPGT), and CPI (via INFTEFF)
            rd_list = [rd_wages]
            if not rd_lpgt.empty:
                rd_list.append(rd_lpgt)
            if not rd_infteff.empty:
                rd_list.append(rd_infteff)
            rd_excess_wages = _rd_multi(rd_list)

            frames.append(_rows(currency, f"{currency}_excess_wages", excess_wages, rd_excess_wages))
            logger.info("  ✓ excess_wages (JPMaQS): %.2f%% (wages_3mma=%.2f - lpgt=%.2f - infteff=%.2f)",
                        excess_wages.dropna().iloc[-1],
                        combined["w"].dropna().iloc[-1],
                        combined["lpgt"].dropna().iloc[-1],
                        combined["infteff"].dropna().iloc[-1])
    else:
        missing = []
        if wages_yoy_3mma.empty:
            missing.append("wages_3mma")
        if lpgt_monthly.empty:
            missing.append("LPGT")
        if infteff_monthly.empty:
            missing.append("INFTEFF")
        if missing:
            logger.warning("  %s: excess_wages skipped — missing %s", currency, " + ".join(missing))

    # ── Assemble ──────────────────────────────────────────────────────────
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
        result["estimated_release_date"] = result["estimated_release_date"].where(
            result["estimated_release_date"].notna(), None
        )
    else:
        result["estimated_release_date"] = None
    logger.info("  %s: %d rows, %d metrics",
                currency.upper(), len(result), result["series_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Schema migration (idempotent)
# ---------------------------------------------------------------------------

def _ensure_release_date_column(conn, schema: str) -> None:
    """Add estimated_release_date column if it doesn't exist yet."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                ALTER TABLE {schema}.labour_derived
                ADD COLUMN IF NOT EXISTS estimated_release_date DATE
            """)
        conn.commit()
        logger.info("Ensured estimated_release_date column on %s.labour_derived", schema)
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        logger.debug("Table %s.labour_derived does not exist yet", schema)

# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(df: pd.DataFrame, conn, schema: str) -> None:
    if df.empty:
        return

    now  = datetime.now(timezone.utc)
    cols = ["currency", "series_id", "time", "value", "estimated_release_date"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]

    logger.info("Upserting %d rows into %s.labour_derived", len(rows), schema)

    sql = f"""
        INSERT INTO {schema}.labour_derived (
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
        logger.info("Upsert complete: %s.labour_derived rows=%d", schema, len(rows))
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        logger.error("Table %s.labour_derived does not exist — run setup SQL first.", schema)
        raise
    except Exception:
        conn.rollback()
        logger.exception("Upsert failed: %s.labour_derived", schema)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_and_store_labour_derived(
    conn,
    schema: str = "macro",
    currencies: list[str] | None = None,
    lookback_days: int = 90,
) -> None:
    """
    Compute and store labour derived metrics for all configured currencies.
    """
    configs = CURRENCY_CONFIGS
    if currencies:
        configs = {k: v for k, v in CURRENCY_CONFIGS.items() if k in currencies}

    _ensure_release_date_column(conn, schema)
    mapper = get_release_date_mapper(conn)

    all_frames = []
    for currency, cfg in configs.items():
        df = _build_currency(currency, cfg, conn, schema, mapper)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No labour derived data produced")
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
        "✓ Labour derived complete: %d currencies, %d total rows",
        len(all_frames), len(combined),
    )