"""
Monetary Derived — Unified Multi-Currency
==========================================

Computes all derived metrics for the monetary tightening block.
Absorbs CPI transforms needed for INFE and excess_core_cpi, adds
yield-derived metrics anchored to r* (trend GDP growth).

NO scoring, NO z-scores. Pure math transforms only.
Scoring belongs exclusively in monetary_signals.py.

Monetary tightening block factors served by this layer:
  F1: Excess CPI inflation      → excess_core_cpi (core_6m_ann minus INFTEFF)
  F2: Inflation direction        → nowcast_cpi_chg (base-effects projected YoY change)
  F3: Policy divergence momentum → yield_2y_momentum (21-day change in 2Y)

Cross-table dependencies:
  - INFTEFF from macro.labour_derived (effective inflation target per currency)
  - trend GDP from macro.growth_derived (20Q rolling median = r* proxy)

Output metrics per currency:
  ── CPI intermediates (needed for INFE + Factor 1) ──
  {ccy}_core_6m_ann           — Core CPI 6M/6M annualised %
  {ccy}_headline_yoy          — Headline CPI YoY %
  {ccy}_excess_core_cpi       — core_6m_ann minus INFTEFF (Factor 1)

  ── CPI nowcast (Factor 2: inflation direction) ──
  {ccy}_nowcast_cpi_chg       — Projected YoY change vs current (Factor 2 signal)
  {ccy}_nowcast_yoy_avg       — Projected headline YoY avg 1-3M ahead (monitoring)

  ── Inflation expectations ──
  {ccy}_infe_1y               — 75% trend + 25% INFTEFF
  {ccy}_infe_2y               — 50% trend + 50% INFTEFF
  {ccy}_infe_5y               — 25% trend + 75% INFTEFF
  {ccy}_infe_blend            — mean(1Y, 2Y, 5Y) ≈ 2.6Y effective duration

  ── Yield-derived (Factors 2–3) ──
  {ccy}_real_yield_2y         — 2Y nominal minus INFE_blend
  {ccy}_real_yield_2y_excess  — real_yield minus r* (trend GDP) (level, stored for monitoring)
  {ccy}_ry_excess_3m_chg      — 63-day change in real_yield_excess (Factor 2 constituent)
  {ccy}_ry_excess_6m_chg      — 130-day change in real_yield_excess (Factor 2 constituent)
  {ccy}_ry_excess_12m_chg     — 261-day change in real_yield_excess (Factor 2 constituent)
  {ccy}_yield_2y_momentum     — 21-day change in 2Y yield (Factor 3)

Reads from:  macro.series_data, macro.labour_derived, macro.growth_derived
Writes to:   macro.monetary_derived

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.monetary_derived (
        currency                TEXT          NOT NULL,
        series_id               TEXT          NOT NULL,
        time                    DATE          NOT NULL,
        value                   FLOAT,
        estimated_release_date  DATE,
        updated_at              TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_monetary_derived_time
        ON macro.monetary_derived(time);
    CREATE INDEX IF NOT EXISTS idx_monetary_derived_currency
        ON macro.monetary_derived(currency);
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ...engine.logger import get_logger
from ...engine.release_dates import get_release_date_mapper

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Currency configs
# ---------------------------------------------------------------------------

CURRENCY_CONFIGS = {
    "usd": {
        "headline_id":      "us_cpi_all_items_sa",
        "core_id":          "us_cpi_core_sa",
        "yield_2y_id":      "usd_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
    },
    "eur": {
        "headline_id":      "eur_hicp_headline_nsa",
        "core_id":          "eur_hicp_core_nsa",
        "yield_2y_id":      "eur_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
    },
    "gbp": {
        "headline_id":      "gbp_cpi_headline_nsa",
        "core_id":          "gbp_cpi_core_nsa",
        "yield_2y_id":      "gbp_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
    },
    "aud": {
        "headline_id":      "aud_cpi_headline_m_sa",
        "headline_backfill_id":  "aud_cpi_headline_q_nsa",
        "headline_backfill_freq": "Q",
        "core_id":          "aud_cpi_trimmed_mean_m_sa",
        "core_backfill_id": "aud_cpi_core_q_nsa",
        "core_backfill_freq": "Q",
        "yield_2y_id":      "aud_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
    },
    "cad": {
        "headline_id":      "cad_cpi_headline_sa",
        "core_id":          "cad_cpi_core_sa",
        "yield_2y_id":      "cad_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
    },
    "jpy": {
        "headline_id":      "jpy_cpi_headline_sa",
        "core_id":          "jpy_cpi_core_sa",
        "yield_2y_id":      "jpy_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
    },
    "nzd": {
        "headline_id":      "nzd_cpi_headline_sa",
        "core_id":          "nzd_cpi_nontradable_sa",
        "yield_2y_id":      "nzd_yield_2y",
        "headline_freq":    "Q",
        "core_freq":        "Q",
    },
    "chf": {
        "headline_id":      "chf_cpi_headline_nsa",
        "core_id":          "chf_cpi_core_sa",
        "core_type":        "pct",       # FSO publishes core CPI as YoY %, not index
        "yield_2y_id":      "chf_yield_2y",
        "headline_freq":    "M",
        "core_freq":        "M",
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


def _load_cross_table(
    conn, schema: str, table: str, currency: str, series_suffix: str,
) -> pd.Series:
    """
    Load a single metric from another derived table (labour_derived, growth_derived).
    Returns monthly series indexed by time.
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
    s = df.set_index("time")["value"]
    # Normalise to month-end for consistent joins
    s.index = s.index.to_period("M").to_timestamp("M")
    s = s.groupby(level=0).last().sort_index()
    return s


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

    Both are normalised to monthly. Primary takes precedence in overlap.
    Growth rates are base-invariant so different index bases (2015=100 vs 2020=100)
    don't matter — we never use the spliced levels directly, only YoY/6M ann.

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

def _ann_6m(s: pd.Series) -> pd.Series:
    """6M/6M annualised: ((s_t / s_{t-6})^2 - 1) × 100."""
    ratio = s / s.shift(6)
    return ((ratio ** 2) - 1.0).mul(100.0).round(4)


def _yoy(s: pd.Series) -> pd.Series:
    """YoY: (s_t / s_{t-12} - 1) × 100  (monthly data)."""
    return ((s / s.shift(12)) - 1.0).mul(100.0).round(4)


# ---------------------------------------------------------------------------
# Release date helpers
# ---------------------------------------------------------------------------

def _rd(idx, mapper, series_id, backfill_id=None, cutoff=None):
    """
    Build release-date Series for a single (possibly spliced) raw input.

    For non-spliced: every date uses series_id's lag.
    For spliced: dates before cutoff use backfill_id's lag, at/after use series_id's.
    """
    def _pick(t):
        if backfill_id and cutoff is not None and t < cutoff:
            return mapper(backfill_id, t)
        return mapper(series_id, t)

    return pd.Series([_pick(t) for t in idx], index=idx)


def _rd_multi(rd_series_list):
    """
    Element-wise max of multiple release-date Series (multi-input metric).

    Each input is a pd.Series of dates. Returns the latest release date
    at each observation date — the row can't exist until all inputs publish.
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
    infteff: pd.Series | None = None,
    trend_gdp: pd.Series | None = None,
) -> pd.DataFrame:
    logger.info("Building monetary derived for %s...", currency.upper())
    frames = []

    # INFTEFF is required — no hardcoded fallback
    if infteff is not None:
        eff_target_series = infteff
        logger.info("  Using INFTEFF as effective target (latest=%.2f%%)",
                    infteff.dropna().iloc[-1] if not infteff.dropna().empty else float("nan"))
    else:
        eff_target_series = None
        logger.warning("  %s: INFTEFF unavailable — excess_core_cpi and INFE will be skipped",
                       currency)

    # ==================================================================
    # CPI TRANSFORMS (intermediates for INFE + Factor 1: excess_core_cpi)
    # ==================================================================

    # ── Core CPI ──────────────────────────────────────────────────────
    raw_c = _load_raw(cfg["core_id"], conn, schema)
    core_6m = pd.Series(dtype=float)
    rd_core = pd.Series(dtype="object")
    core_splice_cutoff = None

    if not raw_c.empty:
        # Splice with quarterly backfill if configured (e.g. AUD)
        if cfg.get("core_backfill_id"):
            raw_cb = _load_raw(cfg["core_backfill_id"], conn, schema)
            if not raw_cb.empty:
                c, core_splice_cutoff = _splice_series(
                    raw_c, raw_cb, cfg["core_freq"], cfg["core_backfill_freq"],
                )
                logger.info("  ✓ Core spliced: backfill %s → primary %s",
                            cfg["core_backfill_id"], cfg["core_id"])
            else:
                c = _normalise_to_monthly(raw_c, cfg["core_freq"])
        else:
            c = _normalise_to_monthly(raw_c, cfg["core_freq"])

        # Release dates for core CPI (splice-aware)
        rd_core = _rd(
            c.index, mapper, cfg["core_id"],
            backfill_id=cfg.get("core_backfill_id") if core_splice_cutoff else None,
            cutoff=core_splice_cutoff,
        )

        if cfg.get("core_type") == "pct":
            # Already published as YoY % (e.g. CHF FSO) — use directly
            core_6m = c.round(4)
        else:
            core_6m = _ann_6m(c)

        # Store core_6m_ann (monitoring + input to excess_core_cpi)
        frames.append(_rows(currency, f"{currency}_core_6m_ann", core_6m, rd_core))

        # Store core_yoy — simple YoY % for level scoring with neutral="mean"
        if cfg.get("core_type") == "pct":
            # CHF: already published as YoY % — same as core_6m for this currency
            core_yoy = c.round(4)
        else:
            core_yoy = _yoy(c)
        frames.append(_rows(currency, f"{currency}_core_yoy", core_yoy, rd_core))

        # Factor 1: excess_core_cpi — vs INFTEFF (required)
        if eff_target_series is not None:
            infteff_aligned = eff_target_series.reindex(core_6m.index, method="ffill")
            excess_core = (core_6m - infteff_aligned).round(4)
            frames.append(_rows(currency, f"{currency}_excess_core_cpi", excess_core, rd_core))
        else:
            excess_core = pd.Series(dtype=float)
            logger.warning("  %s: excess_core_cpi skipped — no INFTEFF", currency)

        logger.info("  ✓ core: 6m_ann=%.2f%%  excess=%.2f",
                    core_6m.dropna().iloc[-1] if not core_6m.dropna().empty else float("nan"),
                    excess_core.dropna().iloc[-1] if not excess_core.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: core CPI missing", currency)

    # ── Headline CPI (needed for INFE construction) ────────────────────
    raw_h = _load_raw(cfg["headline_id"], conn, schema)
    headline_6m = pd.Series(dtype=float)
    headline_yoy = pd.Series(dtype=float)
    rd_headline = pd.Series(dtype="object")
    headline_splice_cutoff = None

    if not raw_h.empty:
        # Splice with quarterly backfill if configured (e.g. AUD)
        if cfg.get("headline_backfill_id"):
            raw_hb = _load_raw(cfg["headline_backfill_id"], conn, schema)
            if not raw_hb.empty:
                h, headline_splice_cutoff = _splice_series(
                    raw_h, raw_hb, cfg["headline_freq"], cfg["headline_backfill_freq"],
                )
                logger.info("  ✓ Headline spliced: backfill %s → primary %s",
                            cfg["headline_backfill_id"], cfg["headline_id"])
            else:
                h = _normalise_to_monthly(raw_h, cfg["headline_freq"])
        else:
            h = _normalise_to_monthly(raw_h, cfg["headline_freq"])

        # Release dates for headline CPI (splice-aware)
        rd_headline = _rd(
            h.index, mapper, cfg["headline_id"],
            backfill_id=cfg.get("headline_backfill_id") if headline_splice_cutoff else None,
            cutoff=headline_splice_cutoff,
        )

        headline_6m = _ann_6m(h)
        headline_yoy = _yoy(h)

        # Store headline_yoy (monitoring)
        frames.append(_rows(currency, f"{currency}_headline_yoy", headline_yoy, rd_headline))

        logger.info("  ✓ headline: yoy=%.2f%%",
                    headline_yoy.dropna().iloc[-1] if not headline_yoy.dropna().empty else float("nan"))

        # ==============================================================
        # CPI NOWCAST — base-effects projection (Factor 2: inflation direction)
        # ==============================================================
        # Project headline YoY forward 1/2/3 months using recent MoM trend.
        # The denominators (CPI index from 11/10/9 months ago) are already known.
        # The numerator is projected using a 3-month trailing average MoM.
        #
        # nowcast_cpi_chg = avg(projected_yoy_1m, _2m, _3m) - current_yoy
        #   Positive = inflation accelerating = hawkish = bullish for currency
        #   Negative = inflation decelerating = dovish = bearish for currency
        # Naturally centers around zero, so neutral="zero" works in scoring.
        # ==============================================================

        if len(h.dropna()) >= 15:
            # Monthly % change in CPI index
            mom = h.pct_change()
            # 3-month trailing average MoM (recent momentum assumption)
            mom_trend = mom.rolling(3, min_periods=2).mean()

            # Build nowcast projections at each observation month
            nowcast_yoy_list = []
            for n in [1, 2, 3]:
                # Projected index: current level × (1 + mom_trend)^n
                projected_idx = h * ((1 + mom_trend) ** n)
                # Known denominator: CPI index from (12 - n) months ago
                denominator = h.shift(12 - n)
                # Projected YoY for month t+n
                projected_yoy = ((projected_idx / denominator) - 1.0).mul(100.0)
                nowcast_yoy_list.append(projected_yoy)

            # Average of 1/2/3 month projections (~2 month effective horizon)
            nowcast_avg = (nowcast_yoy_list[0] + nowcast_yoy_list[1] + nowcast_yoy_list[2]) / 3
            # Signal: projected change from current YoY
            nowcast_chg = (nowcast_avg - headline_yoy).round(4)

            # Store the projected change (the F2 signal)
            frames.append(_rows(currency, f"{currency}_nowcast_cpi_chg", nowcast_chg, rd_headline))

            # Also store the projected YoY average for monitoring
            frames.append(_rows(currency, f"{currency}_nowcast_yoy_avg", nowcast_avg.round(4), rd_headline))

            latest_chg = nowcast_chg.dropna().iloc[-1] if not nowcast_chg.dropna().empty else float("nan")
            latest_avg = nowcast_avg.dropna().iloc[-1] if not nowcast_avg.dropna().empty else float("nan")
            logger.info("  ✓ nowcast: projected_yoy=%.2f%%  chg=%+.2fpp (vs current yoy)",
                        latest_avg, latest_chg)
        else:
            logger.warning("  %s: nowcast skipped — insufficient headline history", currency)

    else:
        logger.warning("  %s: headline CPI missing", currency)

    # ==================================================================
    # INFE — Inflation Expectations (3 horizons + blend)
    # Uses INFTEFF as effective target instead of hardcoded + credibility adj.
    # ==================================================================

    infe_blend = pd.Series(dtype=float)
    rd_infe = pd.Series(dtype="object")

    if not core_6m.empty and not headline_6m.empty:
        infe_df = pd.DataFrame({
            "h6m": headline_6m,
            "c6m": core_6m,
        }).dropna()

        if not infe_df.empty:
            trend = (infe_df["h6m"] + infe_df["c6m"]) / 2

            # Effective target: INFTEFF required
            if eff_target_series is not None:
                eff_tgt = eff_target_series.reindex(trend.index, method="ffill")
            else:
                logger.warning("  %s: INFE skipped — no INFTEFF", currency)
                eff_tgt = None

            if eff_tgt is not None:
                # Three horizons: short = mostly CPI now, long = mostly target
                infe_1y = (0.75 * trend + 0.25 * eff_tgt).round(4)
                infe_2y = (0.50 * trend + 0.50 * eff_tgt).round(4)
                infe_5y = (0.25 * trend + 0.75 * eff_tgt).round(4)
                infe_blend = ((infe_1y + infe_2y + infe_5y) / 3).round(4)

                # INFE depends on both core and headline CPI release dates
                rd_infe = _rd_multi([rd_core, rd_headline])

                # Store all horizons + blend
                frames.append(_rows(currency, f"{currency}_infe_1y", infe_1y, rd_infe))
                frames.append(_rows(currency, f"{currency}_infe_2y", infe_2y, rd_infe))
                frames.append(_rows(currency, f"{currency}_infe_5y", infe_5y, rd_infe))
                frames.append(_rows(currency, f"{currency}_infe_blend", infe_blend, rd_infe))

                logger.info("  ✓ infe: 1Y=%.2f%%  2Y=%.2f%%  5Y=%.2f%%  blend=%.2f%%",
                            infe_1y.dropna().iloc[-1] if not infe_1y.dropna().empty else float("nan"),
                            infe_2y.dropna().iloc[-1] if not infe_2y.dropna().empty else float("nan"),
                            infe_5y.dropna().iloc[-1] if not infe_5y.dropna().empty else float("nan"),
                            infe_blend.dropna().iloc[-1] if not infe_blend.dropna().empty else float("nan"))
        else:
            logger.warning("  %s: INFE — insufficient CPI data", currency)
    else:
        logger.warning("  %s: INFE — missing core/headline 6M", currency)

    # ==================================================================
    # YIELD-DERIVED METRICS (Factors 2–3) — DAILY FREQUENCY
    # ==================================================================

    yield_2y_id = cfg.get("yield_2y_id")

    # ── Load raw 2Y yield — keep daily ────────────────────────────────
    y2_daily = pd.Series(dtype=float)

    if yield_2y_id:
        raw_y2 = _load_raw(yield_2y_id, conn, schema)
        if not raw_y2.empty:
            y2_daily = raw_y2.sort_index()
        else:
            logger.warning("  %s: 2Y yield missing (%s)", currency, yield_2y_id)

    # Yields are public on observation date — release date = observation date
    rd_y2 = pd.Series(y2_daily.index, index=y2_daily.index) if not y2_daily.empty else pd.Series(dtype="object")

    # ── Factor 2: Real yield excess vs r* (trend GDP) ────────────────
    if not y2_daily.empty and not infe_blend.empty:
        infe_daily = infe_blend.reindex(y2_daily.index, method="ffill")
        combined = pd.DataFrame({
            "yield": y2_daily,
            "infe": infe_daily,
        }).dropna()
        if not combined.empty:
            real_yield = (combined["yield"] - combined["infe"]).round(4)

            # Release date = latest of (yield date, INFE release date)
            rd_infe_daily = rd_infe.reindex(real_yield.index, method="ffill")
            rd_real_yield = pd.concat([
                pd.Series(real_yield.index, index=real_yield.index),
                rd_infe_daily,
            ], axis=1).max(axis=1)

            frames.append(_rows(currency, f"{currency}_real_yield_2y", real_yield, rd_real_yield))
            logger.info("  ✓ real_yield_2y: %.2f%% (nom=%.2f - infe_blend=%.2f)",
                        real_yield.iloc[-1],
                        combined["yield"].iloc[-1],
                        combined["infe"].iloc[-1])

            # Excess real yield vs r* (trend GDP growth)
            if trend_gdp is not None and not trend_gdp.empty:
                rstar_daily = trend_gdp.reindex(real_yield.index, method="ffill")
                ry_excess = (real_yield - rstar_daily).round(4)
                frames.append(_rows(currency, f"{currency}_real_yield_2y_excess", ry_excess, rd_real_yield))
                logger.info("  ✓ real_yield_2y_excess (vs r*): %+.2f (ry=%.2f - r*=%.2f)",
                            ry_excess.dropna().iloc[-1] if not ry_excess.dropna().empty else float("nan"),
                            real_yield.dropna().iloc[-1] if not real_yield.dropna().empty else float("nan"),
                            rstar_daily.dropna().iloc[-1] if not rstar_daily.dropna().empty else float("nan"))

                # Change windows: is the policy gap closing or widening?
                for label, window in [("3m", 63), ("6m", 130), ("12m", 261)]:
                    if len(ry_excess) > window:
                        chg = (ry_excess - ry_excess.shift(window)).round(4)
                        frames.append(_rows(currency, f"{currency}_ry_excess_{label}_chg", chg, rd_real_yield))
                        logger.info("  ✓ ry_excess_%s_chg: %+.2f",
                                    label,
                                    chg.dropna().iloc[-1] if not chg.dropna().empty else float("nan"))
            else:
                logger.warning("  %s: real_yield_excess — trend GDP (r*) unavailable", currency)
        else:
            logger.warning("  %s: real yield — no overlapping dates", currency)
    elif not y2_daily.empty:
        logger.warning("  %s: real yield — INFE blend not available", currency)

    # ── Factor 3: 2Y yield momentum (21 business day change) ─────────
    if not y2_daily.empty and len(y2_daily) > 21:
        y2_momentum = (y2_daily - y2_daily.shift(21)).round(4)

        # Release date = observation date
        rd_y2_mom = pd.Series(y2_momentum.dropna().index, index=y2_momentum.dropna().index)
        frames.append(_rows(currency, f"{currency}_yield_2y_momentum", y2_momentum, rd_y2_mom))
        logger.info("  ✓ yield_2y_momentum: %+.2fpp",
                    y2_momentum.dropna().iloc[-1] if not y2_momentum.dropna().empty else float("nan"))

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
                CREATE TABLE IF NOT EXISTS {schema}.monetary_derived (
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
        logger.info("Ensured table %s.monetary_derived", schema)
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

    logger.info("Upserting %d rows into %s.monetary_derived", len(rows), schema)

    sql = f"""
        INSERT INTO {schema}.monetary_derived (
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
        logger.info("Upsert complete: %s.monetary_derived rows=%d", schema, len(rows))
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        logger.error(
            "Table %s.monetary_derived does not exist — run setup SQL first.", schema
        )
        raise
    except Exception:
        conn.rollback()
        logger.exception("Upsert failed: %s.monetary_derived", schema)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_and_store_monetary_derived(
    conn,
    schema: str = "macro",
    currencies: list[str] | None = None,
    lookback_days: int = 90,
) -> None:
    """
    Compute and store monetary derived metrics for all configured currencies.

    Args:
        currencies: Optional list to run a subset e.g. ['gbp', 'aud'].
                    Defaults to all currencies in CURRENCY_CONFIGS.
    """
    configs = CURRENCY_CONFIGS
    if currencies:
        configs = {k: v for k, v in CURRENCY_CONFIGS.items() if k in currencies}

    _ensure_table(conn, schema)

    # Build release date mapper once — cached lookup for series → (frequency, lag)
    mapper = get_release_date_mapper(conn)

    all_frames = []
    for currency, cfg in configs.items():
        # Load cross-table dependencies
        infteff = _load_cross_table(
            conn, schema, "labour_derived", currency, "infteff",
        )
        # r* proxy = 20Q rolling median of GDP YoY (trend growth)
        raw_gdp_yoy = _load_cross_table(
            conn, schema, "growth_derived", currency, "gdp_yoy_pct",
        )
        trend_gdp = None
        if not raw_gdp_yoy.empty:
            trend_gdp = raw_gdp_yoy.rolling(window=20, min_periods=8).median()
            logger.info("  %s: r* (trend GDP) loaded: %.2f%%",
                        currency.upper(),
                        trend_gdp.dropna().iloc[-1] if not trend_gdp.dropna().empty else float("nan"))
        else:
            logger.warning("  %s: GDP YoY not available — r* will be missing", currency)

        df = _build_currency(
            currency, cfg, conn, schema, mapper,
            infteff=infteff if not infteff.empty else None,
            trend_gdp=trend_gdp,
        )
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No monetary derived data produced")
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
        "✓ Monetary derived complete: %d currencies, %d total rows",
        len(all_frames),
        len(combined),
    )