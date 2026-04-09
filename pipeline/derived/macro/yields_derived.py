"""
Yields Derived — CB Credibility Metrics
========================================

Computes yield-vs-policy credibility metrics for all 8 currencies.
Writes to macro.yields_derived. Self-contained — no reads from labour_derived.

NO scoring, NO z-scores. Pure math transforms only.

Output metrics per currency (12 for CHF, 13 for formal-target CBs):
  {ccy}_infteff_proforma        — (h_3y + c_3y + 2.0) / 3
  {ccy}_infteff_credadj         — tgt + 0.5 × gap_3y (formal-target CBs only)
  {ccy}_policy_spread           — 2YY − PR
  {ccy}_implied_moves           — policy_spread / 0.25
  {ccy}_implied_moves_ma21      — 21d SMA of implied_moves
  {ccy}_implied_moves_wow       — 5d change of implied_moves
  {ccy}_implied_moves_mom       — 21d change of implied_moves
  {ccy}_implied_moves_mom_blend — 20d/63d/126d (30/50/20) momentum blend
  {ccy}_implied_moves_stale     — level − mom_blend (persistent mispricing)
  {ccy}_implied_moves_wow_std   — rolling 52w std of 5d changes
  {ccy}_implied_moves_wow_z     — wow / wow_std (z-score of weekly repricing)
  {ccy}_implied_moves_wow_z_cum — 4-week rolling sum of wow_z (repricing pressure)
  {ccy}_real_policy_rate        — PR − INFTEFF
  {ccy}_fwd_real_rate           — 2YY − INFTEFF
  {ccy}_target_drift            — INFTEFF − stated_target

Cross-table dependencies:
  - {ccy}_yield_2y    from macro.rates_derived (daily, spliced)
  - {ccy}_policy_rate from macro.series_data   (BIS, daily)
  - headline + core CPI from macro.series_data (monthly index)

Reads from:  macro.rates_derived, macro.series_data
Writes to:   macro.yields_derived

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.yields_derived (
        currency                TEXT          NOT NULL,
        series_id               TEXT          NOT NULL,
        time                    DATE          NOT NULL,
        value                   FLOAT,
        estimated_release_date  DATE,
        updated_at              TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_yields_derived_time
        ON macro.yields_derived(time);
    CREATE INDEX IF NOT EXISTS idx_yields_derived_currency
        ON macro.yields_derived(currency);
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ...engine.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLOBAL_BENCHMARK = 2.0
MOM_WINDOWS = (20, 63, 126)
MOM_WEIGHTS = (0.30, 0.50, 0.20)

CURRENCY_CONFIGS = {
    "usd": {
        "inflation_target":  2.0,
        "target_start_date": "1996-01-01",
        "headline_id":       "us_cpi_all_items_sa",
        "headline_freq":     "M",
        "core_id":           "us_cpi_core_sa",
        "core_freq":         "M",
    },
    "eur": {
        "inflation_target":  2.0,
        "target_start_date": "1999-01-01",
        "headline_id":       "eur_hicp_headline_nsa",
        "headline_freq":     "M",
        "core_id":           "eur_hicp_core_nsa",
        "core_freq":         "M",
    },
    "gbp": {
        "inflation_target":  2.0,
        "target_start_date": "1992-10-01",
        "headline_id":       "gbp_cpi_headline_nsa",
        "headline_freq":     "M",
        "core_id":           "gbp_cpi_core_nsa",
        "core_freq":         "M",
    },
    "aud": {
        "inflation_target":  2.5,
        "target_start_date": "1993-04-01",
        "headline_id":       "aud_cpi_headline_m_sa",
        "headline_freq":     "M",
        "headline_backfill_id":   "aud_cpi_headline_q_nsa",
        "headline_backfill_freq": "Q",
        "core_id":           "aud_cpi_trimmed_mean_m_sa",
        "core_freq":         "M",
        "core_backfill_id":  "aud_cpi_core_q_nsa",
        "core_backfill_freq": "Q",
    },
    "cad": {
        "inflation_target":  2.0,
        "target_start_date": "1991-02-01",
        "headline_id":       "cad_cpi_headline_sa",
        "headline_freq":     "M",
        "core_id":           "cad_cpi_core_sa",
        "core_freq":         "M",
    },
    "jpy": {
        "inflation_target":  2.0,
        "target_start_date": "2013-01-22",
        "headline_id":       "jpy_cpi_headline_sa",
        "headline_freq":     "M",
        "core_id":           "jpy_cpi_core_sa",
        "core_freq":         "M",
    },
    "nzd": {
        "inflation_target":  2.0,
        "target_start_date": "1990-03-01",
        "headline_id":       "nzd_cpi_headline_sa",
        "headline_freq":     "Q",
        "core_id":           "nzd_cpi_nontradable_sa",
        "core_freq":         "Q",
    },
    "chf": {
        "inflation_target":  None,
        "target_start_date": None,
        "headline_id":       "chf_cpi_headline_nsa",
        "headline_freq":     "M",
        "core_id":           "chf_cpi_core_sa",
        "core_freq":         "M",
        "core_type":         "pct",
    },
}

STATED_TARGETS = {
    "usd": 2.0, "eur": 2.0, "gbp": 2.0, "jpy": 2.0,
    "cad": 2.0, "aud": 2.5, "nzd": 2.0, "chf": 2.0,
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
# CPI helpers
# ---------------------------------------------------------------------------

def _normalise_to_monthly(s: pd.Series, freq: str) -> pd.Series:
    if freq == "Q":
        return s.resample("MS").ffill()
    return s


def _yoy(index_series: pd.Series) -> pd.Series:
    return ((index_series / index_series.shift(12)) - 1.0).round(6) * 100


def _load_and_splice(conn, schema, primary_id, primary_freq,
                     backfill_id=None, backfill_freq=None):
    """Load a CPI series, splicing with quarterly backfill if configured."""
    raw = _load_raw(primary_id, conn, schema)
    if raw.empty:
        return pd.Series(dtype=float)

    if backfill_id:
        raw_bf = _load_raw(backfill_id, conn, schema)
        if not raw_bf.empty:
            bf_monthly = _normalise_to_monthly(raw_bf, backfill_freq or "Q")
            primary_monthly = _normalise_to_monthly(raw, primary_freq)
            spliced = bf_monthly.combine_first(primary_monthly)
            spliced.update(primary_monthly)
            logger.info("    Spliced: backfill %s → primary %s", backfill_id, primary_id)
            return spliced.sort_index()

    return _normalise_to_monthly(raw, primary_freq)


# ---------------------------------------------------------------------------
# INFTEFF computations
# ---------------------------------------------------------------------------

def _compute_infteff_proforma(
    headline_yoy: pd.Series,
    core_yoy: pd.Series,
) -> pd.Series:
    h_3y = headline_yoy.rolling(36, min_periods=12).median()
    c_3y = core_yoy.rolling(36, min_periods=12).median()
    return ((h_3y + c_3y + GLOBAL_BENCHMARK) / 3.0).round(4)


def _compute_infteff_credadj(
    headline_yoy: pd.Series,
    inflation_target: float,
    target_start_date: str | None = None,
) -> pd.Series:
    gap = headline_yoy - inflation_target
    gap_3y = gap.rolling(36, min_periods=12).mean()
    result = (inflation_target + 0.5 * gap_3y).round(4)
    if target_start_date:
        result = result[result.index >= pd.Timestamp(target_start_date)]
    return result


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
    else:
        d["estimated_release_date"] = s.index
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# Per-currency computation
# ---------------------------------------------------------------------------

def _build_currency(
    currency: str,
    cfg: dict,
    conn,
    schema: str,
) -> pd.DataFrame:
    ccy = currency
    frames: list[pd.DataFrame] = []
    logger.info("Processing yields_derived for %s", ccy.upper())

    # ── Load inputs ───────────────────────────────────────────────────────

    yield_2y = _load_cross_table(conn, schema, "rates_derived", ccy, "yield_2y")
    if yield_2y.empty:
        logger.warning("  %s: yield_2y not available — skipping", ccy.upper())
        return pd.DataFrame()

    policy_rate = _load_raw(f"{ccy}_policy_rate", conn, schema)
    if policy_rate.empty:
        logger.warning("  %s: policy_rate not available — skipping", ccy.upper())
        return pd.DataFrame()

    target = cfg["inflation_target"]
    stated_target = STATED_TARGETS[ccy]

    # ── Load and splice CPI ───────────────────────────────────────────────

    headline = _load_and_splice(
        conn, schema, cfg["headline_id"], cfg["headline_freq"],
        cfg.get("headline_backfill_id"), cfg.get("headline_backfill_freq"),
    )
    core = _load_and_splice(
        conn, schema, cfg["core_id"], cfg["core_freq"],
        cfg.get("core_backfill_id"), cfg.get("core_backfill_freq"),
    )

    headline_yoy = _yoy(headline) if not headline.empty else pd.Series(dtype=float)
    if not core.empty:
        core_yoy = core.round(4) if cfg.get("core_type") == "pct" else _yoy(core)
    else:
        core_yoy = headline_yoy

    # ── INFTEFF pro-forma ─────────────────────────────────────────────────

    infteff_pf = pd.Series(dtype=float)
    if not headline_yoy.empty:
        infteff_pf = _compute_infteff_proforma(headline_yoy, core_yoy)
        frames.append(_rows(ccy, f"{ccy}_infteff_proforma", infteff_pf))
        if not infteff_pf.dropna().empty:
            logger.info("  ✓ infteff_proforma: %.2f%%", infteff_pf.dropna().iloc[-1])

    # ── INFTEFF credibility-adjusted ──────────────────────────────────────

    infteff_ca = pd.Series(dtype=float)
    if target is not None and not headline_yoy.empty:
        infteff_ca = _compute_infteff_credadj(
            headline_yoy, target, cfg.get("target_start_date"),
        )
        frames.append(_rows(ccy, f"{ccy}_infteff_credadj", infteff_ca))
        if not infteff_ca.dropna().empty:
            logger.info("  ✓ infteff_credadj: %.2f%%", infteff_ca.dropna().iloc[-1])
    elif target is None:
        logger.info("  %s: no formal target — credadj not applicable", ccy.upper())

    # ── Choose main INFTEFF ───────────────────────────────────────────────

    if not infteff_ca.dropna().empty:
        infteff_main = infteff_ca
        infteff_path = "credadj"
    elif not infteff_pf.dropna().empty:
        infteff_main = infteff_pf
        infteff_path = "proforma"
    else:
        infteff_main = pd.Series(dtype=float)
        infteff_path = "none"
    logger.info("  INFTEFF source: %s", infteff_path)

    # ── Align to daily ────────────────────────────────────────────────────

    policy_daily = policy_rate.reindex(yield_2y.index, method="ffill")
    if not infteff_main.empty:
        infteff_daily = infteff_main.reindex(yield_2y.index, method="ffill")
    else:
        infteff_daily = pd.Series(dtype=float)

    # ── policy_spread = 2YY − PR ──────────────────────────────────────────

    policy_spread = (yield_2y - policy_daily).round(4).dropna()
    frames.append(_rows(ccy, f"{ccy}_policy_spread", policy_spread))

    # ── implied_moves = spread / 0.25 ─────────────────────────────────────

    implied = (policy_spread / 0.25).round(4)
    frames.append(_rows(ccy, f"{ccy}_implied_moves", implied))

    # ── implied_moves_ma21 ────────────────────────────────────────────────

    implied_ma21 = implied.rolling(21, min_periods=5).mean().round(4)
    frames.append(_rows(ccy, f"{ccy}_implied_moves_ma21", implied_ma21))

    # ── implied_moves_wow = 5d change ─────────────────────────────────────

    implied_wow = (implied - implied.shift(5)).round(4)
    frames.append(_rows(ccy, f"{ccy}_implied_moves_wow", implied_wow))

    # ── implied_moves_mom = 21d change ────────────────────────────────────

    implied_mom = (implied - implied.shift(21)).round(4)
    frames.append(_rows(ccy, f"{ccy}_implied_moves_mom", implied_mom))

    # ── implied_moves_mom_blend = 20d/63d/126d (30/50/20) ─────────────────

    if len(implied) > MOM_WINDOWS[0]:
        components = []
        for window, weight in zip(MOM_WINDOWS, MOM_WEIGHTS):
            if len(implied) > window:
                components.append(weight * (implied - implied.shift(window)))
            else:
                components.append(weight * (implied - implied.shift(MOM_WINDOWS[0])))
        implied_blend = sum(components).round(4)
        frames.append(_rows(ccy, f"{ccy}_implied_moves_mom_blend", implied_blend))

        # stale = level − momentum (persistent mispricing, consensus position)
        implied_stale = (implied - implied_blend).round(4)
        frames.append(_rows(ccy, f"{ccy}_implied_moves_stale", implied_stale))

    # ── implied_moves_wow_std = rolling 52w std of 5d changes ─────────────

    if not implied_wow.empty:
        wow_std = implied_wow.rolling(260, min_periods=60).std().round(4)
        frames.append(_rows(ccy, f"{ccy}_implied_moves_wow_std", wow_std))

        # z-score: how unusual is this week's repricing?
        wow_z = (implied_wow / wow_std).round(4)
        frames.append(_rows(ccy, f"{ccy}_implied_moves_wow_z", wow_z))

        # 4-week rolling sum of wow_z — gives shock factor "memory"
        # 4 consecutive +0.5σ weeks → +2.0σ cumulative; drains as weeks roll off
        wow_z_cum = wow_z.rolling(20, min_periods=5).sum().round(4)
        frames.append(_rows(ccy, f"{ccy}_implied_moves_wow_z_cum", wow_z_cum))

    # ── Log implied summary ───────────────────────────────────────────────

    if not implied.empty:
        latest_impl = implied.iloc[-1]
        latest_wow = implied_wow.dropna().iloc[-1] if not implied_wow.dropna().empty else None
        latest_mom = implied_mom.dropna().iloc[-1] if not implied_mom.dropna().empty else None
        logger.info("  ✓ implied: %.1f %s | WoW %+.1f | MoM %+.1f",
                    abs(latest_impl), "hikes" if latest_impl >= 0 else "cuts",
                    latest_wow or 0, latest_mom or 0)

    # ── real_policy_rate = PR − INFTEFF ───────────────────────────────────

    if not infteff_daily.empty:
        real_pr = (policy_daily - infteff_daily).round(4).dropna()
        frames.append(_rows(ccy, f"{ccy}_real_policy_rate", real_pr))
        if not real_pr.empty:
            logger.info("  ✓ real_policy_rate: %.2f%%", real_pr.iloc[-1])

    # ── fwd_real_rate = 2YY − INFTEFF ────────────────────────────────────

    if not infteff_daily.empty:
        fwd_real = (yield_2y - infteff_daily).round(4).dropna()
        frames.append(_rows(ccy, f"{ccy}_fwd_real_rate", fwd_real))
        if not fwd_real.empty:
            logger.info("  ✓ fwd_real_rate: %.2f%%", fwd_real.iloc[-1])

    # ── target_drift = INFTEFF − stated_target ────────────────────────────

    if not infteff_daily.empty:
        target_drift = (infteff_daily - stated_target).round(4).dropna()
        frames.append(_rows(ccy, f"{ccy}_target_drift", target_drift))
        if not target_drift.empty:
            logger.info("  ✓ target_drift: %.2f%%", target_drift.iloc[-1])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Table setup
# ---------------------------------------------------------------------------

def _ensure_table(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.yields_derived (
                currency                TEXT          NOT NULL,
                series_id               TEXT          NOT NULL,
                time                    DATE          NOT NULL,
                value                   FLOAT,
                estimated_release_date  DATE,
                updated_at              TIMESTAMPTZ   DEFAULT NOW(),
                PRIMARY KEY (currency, series_id, time)
            );
            CREATE INDEX IF NOT EXISTS idx_yields_derived_time
                ON {schema}.yields_derived(time);
            CREATE INDEX IF NOT EXISTS idx_yields_derived_currency
                ON {schema}.yields_derived(currency);
        """)
    conn.commit()


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(df: pd.DataFrame, conn, schema: str) -> None:
    if df.empty:
        return
    now = datetime.now(timezone.utc)
    cols = ["currency", "series_id", "time", "value", "estimated_release_date"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]

    logger.info("Upserting %d rows into %s.yields_derived", len(rows), schema)
    sql = f"""
        INSERT INTO {schema}.yields_derived (
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
        logger.info("Upsert complete: %s.yields_derived rows=%d", schema, len(rows))
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        logger.error("Table %s.yields_derived does not exist — run setup SQL first.", schema)
        raise
    except Exception:
        conn.rollback()
        logger.exception("Upsert failed: %s.yields_derived", schema)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_and_store_yields_derived(
    conn,
    schema: str = "macro",
    currencies: list[str] | None = None,
    lookback_days: int = 0,
) -> None:
    """
    Compute and store yields-derived credibility metrics for all currencies.
    Must run AFTER rates_derived (depends on spliced yield_2y).
    No dependency on labour_derived — computes INFTEFF from raw CPI.
    """
    configs = CURRENCY_CONFIGS
    if currencies:
        configs = {k: v for k, v in CURRENCY_CONFIGS.items() if k in currencies}

    _ensure_table(conn, schema)

    all_frames = []
    for currency, cfg in configs.items():
        df = _build_currency(currency, cfg, conn, schema)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No yields derived data produced")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    if lookback_days > 0:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        combined["time"] = pd.to_datetime(combined["time"])
        before = len(combined)
        combined = combined[combined["time"] >= cutoff]
        logger.info("Lookback filter: %d → %d rows (last %d days)",
                    before, len(combined), lookback_days)

    _upsert(combined, conn, schema)

    for currency in configs:
        ccy_rows = combined[combined["currency"] == currency]
        metrics = ccy_rows["series_id"].unique()
        logger.info("  %s: %d metrics, %d rows", currency.upper(), len(metrics), len(ccy_rows))