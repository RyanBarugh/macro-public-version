"""
Growth Derived v2 — Unified Multi-Currency
============================================

Computes growth derived metrics for all 8 currencies.
Writes to macro.growth_derived (same table as v1 — additive only).

NO scoring, NO z-scores, NO std_scale. Pure math transforms only.

Changes from v1:
  - Added {ccy}_ip_6m_excess_trend (IP 6M/6M ann minus 60M rolling median)
  - Removed COVID exclusion from rolling baselines (plain rolling median)

JPMaQS growth factors served by this layer:
  F1: Relative output growth — GDP excess + IP excess + IP 6M/6M excess
  F2: Manufacturing business confidence — BCI 3M/3M change + Q/Q change
  F3: Labour market tightening — computed in labour_derived
  F4: Relative private consumption — retail excess

Output metrics per currency:
  {ccy}_gdp_yoy_pct           — Real GDP YoY % (quarterly frequency)
  {ccy}_gdp_excess_trend      — GDP YoY minus 20Q rolling median
  {ccy}_ip_yoy_pct            — IP YoY %
  {ccy}_ip_6m_ann             — IP 6M/6M annualised %
  {ccy}_ip_excess_trend       — IP YoY minus 60M rolling median
  {ccy}_ip_6m_excess_trend    — IP 6M/6M ann minus 60M rolling median  [NEW]
  {ccy}_retail_yoy_pct        — Retail sales YoY %
  {ccy}_retail_excess_trend   — Retail YoY minus 60M rolling median
  {ccy}_bci_3m_chg            — BCI level 3M/3M change
  {ccy}_bci_qoq_chg           — BCI quarterly-averaged Q/Q change

Reads from:  macro.series_data
Writes to:   macro.growth_derived
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
from ...engine.release_dates import get_release_date_mapper

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Currency configs
# ---------------------------------------------------------------------------

CURRENCY_CONFIGS = {
    "usd": {
        "gdp_real_id":   "usd_gdp_real_bea_sa",
        "ip_id":         "usd_ip_total_fred_sa",
        "ip_freq":       "M",
        "retail_id":     "us_retail_total_marts_sa_sm",
        "retail_freq":   "M",
        "bci_id":        "usd_bci_sa",
    },
    "eur": {
        "gdp_real_id":   "eur_gdp_real",
        "ip_id":         "eur_ip_total_sts_sa",
        "ip_freq":       "M",
        "retail_id":     "eur_retail_total_sts_sa",
        "retail_freq":   "M",
        "bci_id":        "eur_bci_sa",
    },
    "gbp": {
        "gdp_real_id":   "gbp_gdp_real",
        "ip_id":         "gbp_ip_total_ons_sa",
        "ip_freq":       "M",
        "retail_id":     "gbp_retail_total_vol_sa",
        "retail_freq":   "M",
        "bci_id":        "gbp_bci_sa",
    },
    "aud": {
        "gdp_real_id":          "aud_gdp_real",
        "ip_id":                "aud_ip_total_gva_sa",
        "ip_freq":              "Q",
        "retail_id":            "aud_retail_total_hsi_sa",       # monthly HSI (2012+)
        "retail_freq":          "M",
        "retail_backfill_id":   "aud_retail_total_rt_q_sa",     # quarterly RT (1983+)
        "retail_backfill_freq": "Q",
        "bci_id":               "aud_bci_sa",
    },
    "cad": {
        "gdp_real_id":   "cad_gdp_real",
        "ip_id":         "cad_ip_total_sa",
        "ip_freq":       "M",
        "retail_id":     "cad_retail_total_sa",
        "retail_freq":   "M",
        "bci_id":        "cad_bci_sa",
    },
    "jpy": {
        "gdp_real_id":          "jpy_gdp_real",
        "ip_id":                "jpy_ip_total_iip_sa",          # METI preliminary (fresh data)
        "ip_freq":              "M",
        "ip_backfill_id":       "jpy_ip_total_fred_sa",         # FRED OECD (1955+, 2015=100)
        "ip_backfill_freq":     "M",
        "retail_id":            "jpy_retail_total_cdts_sa",     # METI preliminary (fresh data)
        "retail_freq":          "M",
        "retail_backfill_id":   "jpy_retail_total_fred_sa",     # FRED OECD (1960+, 2015=100)
        "retail_backfill_freq": "M",
        "bci_id":               "jpy_bci_sa",
    },
    "nzd": {
        "gdp_real_id":   "nzd_gdp_real_sa",
        "ip_id":         "nzd_ip_manufacturing_sa",   # manufacturing GVA as IP proxy
        "ip_freq":       "Q",
        "retail_id":     "nzd_retail_total_deflated_sa",
        "retail_freq":   "Q",
        "bci_id":        "nzd_bci_sa",
    },

    "chf": {
        "gdp_real_id":   "chf_gdp_real_eurostat",
        "ip_id":         "chf_ip_total_fso_sa",
        "ip_freq":       "Q",
        "retail_id":     "chf_retail_total_sa",
        "retail_freq":   "M",
        "bci_id":        "chf_bci_sa",
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


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def _yoy(s: pd.Series) -> pd.Series:
    """YoY for monthly data: shift(12)."""
    return ((s / s.shift(12)) - 1.0).mul(100.0).round(4)


def _yoy_quarterly(s: pd.Series) -> pd.Series:
    """YoY for quarterly data: shift(4)."""
    return ((s / s.shift(4)) - 1.0).mul(100.0).round(4)


def _ann_6m(s: pd.Series) -> pd.Series:
    """6M/6M annualised: ((s_t / s_{t-6})^2 - 1) × 100."""
    ratio = s / s.shift(6)
    return ((ratio ** 2) - 1.0).mul(100.0).round(4)


def _chg(s: pd.Series, n: int) -> pd.Series:
    """N-period absolute change."""
    return (s - s.shift(n)).round(4)


def _rolling_median(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Plain rolling median — no COVID exclusion (v2 change)."""
    return s.rolling(window=window, min_periods=min_obs).median()


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
    currency: str, cfg: dict, engine, schema: str, mapper,
) -> pd.DataFrame:
    logger.info("Building growth derived v2 for %s...", currency.upper())
    frames = []

    # ── GDP (quarterly) ───────────────────────────────────────────────────
    raw_g = _load_raw(cfg["gdp_real_id"], engine, schema)
    if not raw_g.empty:
        g = _to_quarterly(raw_g).dropna()
        rd_gdp = _rd(g.index, mapper, cfg["gdp_real_id"])

        gdp_yoy = _yoy_quarterly(g)
        frames.append(_rows(currency, f"{currency}_gdp_yoy_pct", gdp_yoy, rd_gdp))

        # Excess vs 5Y trend (20Q rolling median, no COVID exclusion)
        gdp_excess = (gdp_yoy - _rolling_median(gdp_yoy, window=20, min_obs=8)).round(4)
        frames.append(_rows(currency, f"{currency}_gdp_excess_trend", gdp_excess, rd_gdp))

        logger.info("  ✓ GDP: yoy=%.2f%%  excess=%.2f",
                    gdp_yoy.dropna().iloc[-1] if not gdp_yoy.dropna().empty else float("nan"),
                    gdp_excess.dropna().iloc[-1] if not gdp_excess.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: GDP missing", currency)

    # ── IP ────────────────────────────────────────────────────────────────
    raw_ip = _load_raw(cfg["ip_id"], engine, schema)
    ip = pd.Series(dtype=float)
    rd_ip = pd.Series(dtype="object")
    ip_splice_cutoff = None

    if not raw_ip.empty and cfg.get("ip_backfill_id"):
        # Splice: backfill for history, primary for recent
        raw_bf = _load_raw(cfg["ip_backfill_id"], engine, schema)
        if not raw_bf.empty:
            ip, ip_splice_cutoff = _splice_series(
                raw_ip, raw_bf, cfg["ip_freq"], cfg.get("ip_backfill_freq", "M"),
            )
            logger.info("  ✓ IP spliced: backfill %s → primary %s",
                        cfg["ip_backfill_id"], cfg["ip_id"])
        else:
            ip = _normalise_to_monthly(raw_ip, cfg["ip_freq"])
    elif not raw_ip.empty:
        ip = _normalise_to_monthly(raw_ip, cfg["ip_freq"])
    elif cfg.get("ip_backfill_id"):
        # Primary empty, use backfill alone
        raw_bf = _load_raw(cfg["ip_backfill_id"], engine, schema)
        if not raw_bf.empty:
            ip = _normalise_to_monthly(raw_bf, cfg.get("ip_backfill_freq", "M"))
            logger.info("  ✓ IP from backfill only: %s", cfg["ip_backfill_id"])

    if not ip.empty:
        # Build release dates (splice-aware)
        rd_ip = _rd(
            ip.index, mapper, cfg["ip_id"],
            backfill_id=cfg.get("ip_backfill_id") if ip_splice_cutoff else None,
            cutoff=ip_splice_cutoff,
        )

        ip_yoy = _yoy(ip)
        ip_6m  = _ann_6m(ip)
        frames.append(_rows(currency, f"{currency}_ip_yoy_pct", ip_yoy, rd_ip))
        frames.append(_rows(currency, f"{currency}_ip_6m_ann", ip_6m, rd_ip))

        # Excess vs 5Y trend — YoY (60M rolling median, no COVID exclusion)
        ip_excess = (ip_yoy - _rolling_median(ip_yoy, window=60, min_obs=24)).round(4)
        frames.append(_rows(currency, f"{currency}_ip_excess_trend", ip_excess, rd_ip))

        # Excess vs 5Y trend — 6M/6M ann (60M rolling median, no COVID exclusion) [NEW in v2]
        ip_6m_excess = (ip_6m - _rolling_median(ip_6m, window=60, min_obs=24)).round(4)
        frames.append(_rows(currency, f"{currency}_ip_6m_excess_trend", ip_6m_excess, rd_ip))

        logger.info("  ✓ IP: yoy=%.2f%%  6m_ann=%.2f%%  excess=%.2f  6m_excess=%.2f",
                    ip_yoy.dropna().iloc[-1]       if not ip_yoy.dropna().empty       else float("nan"),
                    ip_6m.dropna().iloc[-1]        if not ip_6m.dropna().empty        else float("nan"),
                    ip_excess.dropna().iloc[-1]    if not ip_excess.dropna().empty    else float("nan"),
                    ip_6m_excess.dropna().iloc[-1] if not ip_6m_excess.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: IP missing", currency)

    # ── Retail ────────────────────────────────────────────────────────────
    raw_rt = _load_raw(cfg["retail_id"], engine, schema)
    rt = pd.Series(dtype=float)
    rd_rt = pd.Series(dtype="object")
    rt_splice_cutoff = None

    if not raw_rt.empty and cfg.get("retail_backfill_id"):
        # Splice: backfill for history, primary for recent
        raw_bf = _load_raw(cfg["retail_backfill_id"], engine, schema)
        if not raw_bf.empty:
            rt, rt_splice_cutoff = _splice_series(
                raw_rt, raw_bf, cfg["retail_freq"], cfg.get("retail_backfill_freq", "M"),
            )
            logger.info("  ✓ Retail spliced: backfill %s → primary %s",
                        cfg["retail_backfill_id"], cfg["retail_id"])
        else:
            rt = _normalise_to_monthly(raw_rt, cfg["retail_freq"])
    elif not raw_rt.empty:
        rt = _normalise_to_monthly(raw_rt, cfg["retail_freq"])
    elif cfg.get("retail_backfill_id"):
        # Primary empty, use backfill alone
        raw_bf = _load_raw(cfg["retail_backfill_id"], engine, schema)
        if not raw_bf.empty:
            rt = _normalise_to_monthly(raw_bf, cfg.get("retail_backfill_freq", "M"))
            logger.info("  ✓ Retail from backfill only: %s", cfg["retail_backfill_id"])

    if not rt.empty:
        # Build release dates (splice-aware)
        rd_rt = _rd(
            rt.index, mapper, cfg["retail_id"],
            backfill_id=cfg.get("retail_backfill_id") if rt_splice_cutoff else None,
            cutoff=rt_splice_cutoff,
        )

        rt_yoy = _yoy(rt)
        frames.append(_rows(currency, f"{currency}_retail_yoy_pct", rt_yoy, rd_rt))

        # Excess vs 5Y trend (60M rolling median, no COVID exclusion)
        rt_excess = (rt_yoy - _rolling_median(rt_yoy, window=60, min_obs=24)).round(4)
        frames.append(_rows(currency, f"{currency}_retail_excess_trend", rt_excess, rd_rt))

        logger.info("  ✓ Retail: yoy=%.2f%%  excess=%.2f",
                    rt_yoy.dropna().iloc[-1]    if not rt_yoy.dropna().empty    else float("nan"),
                    rt_excess.dropna().iloc[-1] if not rt_excess.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: Retail missing", currency)

    # ── BCI (monthly) ─────────────────────────────────────────────────────
    raw_bci = _load_raw(cfg["bci_id"], engine, schema)
    if not raw_bci.empty:
        bci = _to_monthly(raw_bci)
        rd_bci = _rd(bci.index, mapper, cfg["bci_id"])

        # 3M/3M change on monthly level
        bci_3m = _chg(bci, 3)
        frames.append(_rows(currency, f"{currency}_bci_3m_chg", bci_3m, rd_bci))

        # Q/Q change: quarterly average, then diff
        bci_q = bci.resample("QE").mean()
        bci_qoq = _chg(bci_q, 1)
        # BCI Q/Q uses the same raw BCI series — build rd on quarterly index
        rd_bci_q = _rd(bci_q.index, mapper, cfg["bci_id"])
        frames.append(_rows(currency, f"{currency}_bci_qoq_chg", bci_qoq, rd_bci_q))

        logger.info("  ✓ BCI: 3m_chg=%.2f  qoq_chg=%.2f",
                    bci_3m.dropna().iloc[-1]  if not bci_3m.dropna().empty  else float("nan"),
                    bci_qoq.dropna().iloc[-1] if not bci_qoq.dropna().empty else float("nan"))
    else:
        logger.warning("  %s: BCI missing", currency)

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
    else:
        result["estimated_release_date"] = None
    logger.info("  %s: %d rows, %d metrics",
                currency.upper(), len(result), result["series_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Schema migration (idempotent)
# ---------------------------------------------------------------------------

def _ensure_release_date_column(db_config: DbConfig, schema: str) -> None:
    """Add estimated_release_date column if it doesn't exist yet."""
    conn = psycopg2.connect(
        host=db_config.host, port=db_config.port,
        dbname=db_config.dbname, user=db_config.user,
        password=db_config.password, sslmode=db_config.sslmode,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    ALTER TABLE {schema}.growth_derived
                    ADD COLUMN IF NOT EXISTS estimated_release_date DATE
                """)
        logger.info("Ensured estimated_release_date column on %s.growth_derived", schema)
    except psycopg2.errors.UndefinedTable:
        logger.debug("Table %s.growth_derived does not exist yet — column will be created with table", schema)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(df: pd.DataFrame, db_config: DbConfig, schema: str) -> None:
    if df.empty:
        return

    now  = datetime.now(timezone.utc)
    cols = ["currency", "series_id", "time", "value", "estimated_release_date"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]

    logger.info("Upserting %d rows into %s.growth_derived", len(rows), schema)

    sql = f"""
        INSERT INTO {schema}.growth_derived (
            currency, series_id, time, value, estimated_release_date, updated_at
        )
        VALUES %s
        ON CONFLICT (currency, series_id, time)
        DO UPDATE SET
            value                  = EXCLUDED.value,
            estimated_release_date = EXCLUDED.estimated_release_date,
            updated_at             = EXCLUDED.updated_at
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
        logger.info("Upsert complete: %s.growth_derived rows=%d", schema, len(rows))
    except psycopg2.errors.UndefinedTable:
        logger.error("Table %s.growth_derived does not exist — run setup SQL first.", schema)
        raise
    except Exception:
        logger.exception("Upsert failed: %s.growth_derived", schema)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_and_store_growth_derived_v2(
    db_config: DbConfig,
    schema: str = "macro",
    currencies: list[str] | None = None,
) -> None:
    """
    Compute and store growth derived v2 metrics for all configured currencies.
    Writes to the same growth_derived table (additive — new series_ids only).
    """
    configs = CURRENCY_CONFIGS
    if currencies:
        configs = {k: v for k, v in CURRENCY_CONFIGS.items() if k in currencies}

    engine = create_engine(
        f"postgresql://{db_config.user}:{db_config.password}@"
        f"{db_config.host}:{db_config.port}/{db_config.dbname}"
    )

    try:
        _ensure_release_date_column(db_config, schema)
        mapper = get_release_date_mapper(engine)

        all_frames = []
        for currency, cfg in configs.items():
            df = _build_currency(currency, cfg, engine, schema, mapper)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            logger.warning("No growth derived v2 data produced")
            return

        combined = pd.concat(all_frames, ignore_index=True)
        _upsert(combined, db_config, schema)
        logger.info(
            "✓ Growth derived v2 complete: %d currencies, %d total rows",
            len(all_frames), len(combined),
        )
    except Exception as e:
        logger.error("Growth derived v2 failed: %s", e, exc_info=True)
        raise
    finally:
        engine.dispose()