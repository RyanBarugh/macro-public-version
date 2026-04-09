"""
ToT Derived — Commodity Terms-of-Trade Proxy
=============================================

Computes commodity-based terms-of-trade proxies for all 8 currencies.
Writes to macro.tot_derived.

NO scoring, NO z-scores. Pure math transforms only.
Scoring belongs exclusively in tot_signals_v2.py.

Methodology:
    JPMorgan/Macrosynergy commodity proxy approach.
    For each currency, construct a weighted geometric mean of export
    commodity prices and import commodity prices. The ToT proxy is:

        log_tot = Σ(export_w_i × ln(P_i)) - Σ(import_w_j × ln(P_j))

    This is equivalent to ln(ExportIndex / ImportIndex) where indices
    are geometric-weighted. The base period cancels in the z-score.

    Momentum signals (12m, 3m change of log_tot) are naturally centered
    around zero — consistent with neutral="zero" across all blocks.

Basket weights:
    Sourced from official trade statistics (ABS, BEA, Eurostat, etc.).
    Only commodity-priced components included; non-commodity trade
    (manufactured goods, services) is excluded and weights renormalized.
    This is standard practice — JPMorgan and Macrosynergy do the same.
    Weights should be reviewed annually.

Output metrics per currency:
    {ccy}_tot_log_level       — ln(export_index / import_index), daily
    {ccy}_tot_12m_chg         — 252-day change of log_tot (annual momentum)
    {ccy}_tot_3m_chg          — 63-day change of log_tot (quarterly momentum)

Reads from:  macro.series_data (commodity prices)
Writes to:   macro.tot_derived

Setup SQL (run once):
---------------------
    CREATE TABLE IF NOT EXISTS macro.tot_derived (
        currency                TEXT          NOT NULL,
        series_id               TEXT          NOT NULL,
        time                    DATE          NOT NULL,
        value                   FLOAT,
        estimated_release_date  DATE,
        updated_at              TIMESTAMPTZ   DEFAULT NOW(),
        PRIMARY KEY (currency, series_id, time)
    );
    CREATE INDEX IF NOT EXISTS idx_tot_derived_time
        ON macro.tot_derived(time);
    CREATE INDEX IF NOT EXISTS idx_tot_derived_currency
        ON macro.tot_derived(currency);
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ...engine.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMODITY SERIES CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each entry maps a short label to the series_id in macro.series_data.
# Register these series in your ingestion config before running.
#
# Daily series (EODHD COMM or FRED):
#   Brent, WTI, Henry Hub, Gold — daily from FRED or EODHD
#
# Monthly series (FRED World Bank pink sheets):
#   Iron ore, copper, coal, TTF gas, wheat, soybeans, corn, aluminium,
#   lumber, dairy, JKM LNG — monthly, forward-filled to daily here.
#
# Series not yet registered will log a warning and be excluded from
# the basket (remaining weights renormalized automatically).
# ═══════════════════════════════════════════════════════════════════════════════

COMMODITY_SERIES = {
    # ── Energy (daily) ────────────────────────────────────────────────────
    "brent":        "tot_brent_crude",          # FRED DCOILBRENTEU or EODHD
    "wti":          "tot_wti_crude",            # FRED DCOILWTICO or EODHD
    "henry_hub":    "tot_henry_hub",            # FRED DHHNGSP or EODHD
    # ── Energy (monthly, FRED World Bank) ─────────────────────────────────
    "ttf_gas":      "tot_ttf_gas_eu",           # FRED PNGASEUUSDM
    "jkm_lng":      "tot_jkm_lng_jp",           # FRED PNGASJPUSDM
    "coal_au":      "tot_coal_au",              # FRED PCOALAUUSDM
    # ── Metals ────────────────────────────────────────────────────────────
    "gold":         "roro_gold_daily",                # FRED GOLDAMGBD228NLBM or EODHD
    "copper":       "tot_copper",               # FRED PCOPPUSDM (monthly)
    "aluminium":    "tot_aluminium",            # FRED PALUMUSDM (monthly)
    "iron_ore":     "tot_iron_ore",             # FRED PIORECRUSDM (monthly)
    # ── Agriculture ───────────────────────────────────────────────────────
    "wheat":        "tot_wheat",                # FRED PWHEAMTUSDM (monthly)
    "soybeans":     "tot_soybeans",             # FRED PSOYBUSDM (monthly)
    "corn":         "tot_corn",                 # FRED PMAIZMTUSDM (monthly)
    "lumber":       "tot_lumber",               # FRED WPUSI012011 (monthly)
    "dairy":        "tot_dairy",                # FRED PDAIRYMBUSDM (monthly)
}


# ═══════════════════════════════════════════════════════════════════════════════
# BASKET WEIGHTS PER CURRENCY
# ═══════════════════════════════════════════════════════════════════════════════
#
# Weights represent share of the commodity-traded portion of merchandise
# trade. Non-commodity trade is excluded. Weights within each basket are
# renormalized at runtime if any series is missing.
#
# Sources: ABS (AUD), StatCan (CAD), SSB (NOK), Stats NZ (NZD),
#          BEA/Census (USD), Eurostat (EUR), ONS (GBP), BoJ (JPY),
#          FSO (CHF). Last updated: March 2026.
#
# Sign convention: export weights are positive (higher export prices =
# better ToT = currency bullish). Import weights are positive (higher
# import prices = worse ToT = currency bearish).
# ═══════════════════════════════════════════════════════════════════════════════

BASKET_CONFIGS = {
    "aud": {
        "export": {
            "iron_ore":     0.412,      # 35% of merch exports / 85% commodity
            "coal_au":      0.235,      # 20/85
            "henry_hub":    0.176,      # 15/85 — LNG proxy (Henry Hub)
            "gold":         0.118,      # 10/85
            "copper":       0.059,      # 5/85
        },
        "import": {
            "brent":        1.0,        # Oil dominates commodity imports
        },
    },
    "cad": {
        "export": {
            "wti":          0.50,       # 40/80
            "henry_hub":    0.1875,     # 15/80
            "wheat":        0.1875,     # 15/80
            "lumber":       0.125,      # 10/80
        },
        "import": {
            "brent":        1.0,        # Generic energy import proxy
        },
    },
    "nok": {
        # NOK not in 8-currency universe but included for completeness.
        # If added later, basket is ready.
        "export": {
            "brent":        0.50,       # 45/90
            "ttf_gas":      0.389,      # 35/90
            "dairy":        0.111,      # 10/90 — fish proxy (no financial series)
        },
        "import": {
            "brent":        1.0,
        },
    },
    "nzd": {
        "export": {
            "dairy":        0.462,      # 30/65
            "wheat":        0.308,      # 20/65 — meat/lamb proxy
            "lumber":       0.154,      # 10/65
            "aluminium":    0.077,      # 5/65 — wool proxy
        },
        "import": {
            "brent":        1.0,
        },
    },
    "usd": {
        "export": {
            "henry_hub":    0.364,      # 20/55 — LNG/gas
            "wti":          0.273,      # 15/55
            "soybeans":     0.121,      # agriculture split
            "corn":         0.121,
            "wheat":        0.121,
        },
        "import": {
            "brent":        1.0,        # Oil import proxy
        },
    },
    "eur": {
        "export": {
            # EUR exports are primarily non-commodity (machinery, vehicles,
            # pharma). No meaningful commodity export basket. Use a minimal
            # placeholder so the import side drives the signal — which is
            # correct: EUR ToT is dominated by energy import costs.
        },
        "import": {
            "ttf_gas":      0.462,      # 30/65
            "brent":        0.385,      # 25/65
            "wheat":        0.154,      # 10/65 — food proxy
        },
    },
    "gbp": {
        "export": {},
        "import": {
            "brent":        0.50,       # 20/40
            "ttf_gas":      0.25,       # 10/40 — NBP proxy
            "wheat":        0.25,       # 10/40 — food proxy
        },
    },
    "jpy": {
        "export": {
            # JPY exports are non-commodity (electronics, vehicles).
            # Same approach as EUR: import side drives the signal.
        },
        "import": {
            "brent":        0.40,       # 30/75
            "jkm_lng":      0.333,      # 25/75
            "coal_au":      0.133,      # 10/75
            "wheat":        0.133,      # 10/75 — food proxy
        },
    },
    "chf": {
        "export": {
            # CHF exports are non-commodity (pharma, watches, machinery).
            # Minimal ToT signal expected — safe haven dynamics dominate.
        },
        "import": {
            "brent":        1.0,        # Generic energy import
        },
    },
}

# Momentum lookbacks (business days)
MOM_12M = 252
MOM_3M = 63

TARGET_TABLE = "tot_derived"


# ═══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_raw(series_id: str, conn, schema: str) -> pd.Series:
    """Load a single series from series_data, return as time-indexed Series."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# COMMODITY PRICE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_all_commodities(conn, schema: str) -> dict[str, pd.Series]:
    """
    Load all commodity price series. Returns dict label → daily Series.
    Monthly series are forward-filled to daily business days.
    Missing series are excluded with a warning.
    """
    commodities = {}
    for label, series_id in COMMODITY_SERIES.items():
        raw = _load_raw(series_id, conn, schema)
        if raw.empty:
            logger.warning("  Commodity %s (%s) — no data, will be excluded from baskets",
                           label, series_id)
            continue

        # FFill to daily business days (handles both daily and monthly series)
        start = raw.index[0]
        end = max(raw.index[-1], pd.Timestamp.today().normalize())
        bdays = pd.bdate_range(start, end)
        daily = raw.reindex(bdays).ffill()

        # Drop any NaN at the start (before first observation)
        daily = daily.dropna()

        if daily.empty:
            logger.warning("  Commodity %s — empty after ffill", label)
            continue

        commodities[label] = daily
        logger.info("  Loaded %s: %d daily obs, %s to %s, last=%.2f",
                    label, len(daily),
                    daily.index[0].date(), daily.index[-1].date(),
                    daily.iloc[-1])

    return commodities


# ═══════════════════════════════════════════════════════════════════════════════
# BASKET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_log_basket(
    weights: dict[str, float],
    commodities: dict[str, pd.Series],
) -> pd.Series:
    """
    Compute log of geometric-weighted basket index.

    log_basket = Σ(w_i × ln(P_i_t))

    Weights are renormalized to sum to 1.0 after excluding any
    missing commodities. Returns empty Series if no commodities available.
    """
    available = {k: v for k, v in weights.items() if k in commodities}
    if not available:
        return pd.Series(dtype=float)

    # Renormalize weights
    total_w = sum(available.values())
    if total_w <= 0:
        return pd.Series(dtype=float)
    norm_w = {k: v / total_w for k, v in available.items()}

    # Align all series to common date range
    all_series = [commodities[k] for k in norm_w]
    common_idx = all_series[0].index
    for s in all_series[1:]:
        common_idx = common_idx.intersection(s.index)
    common_idx = common_idx.sort_values()

    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    # Compute weighted log sum
    log_basket = pd.Series(0.0, index=common_idx)
    for label, w in norm_w.items():
        prices = commodities[label].reindex(common_idx)
        # Safety: replace zeros/negatives to avoid log errors
        prices = prices.clip(lower=1e-6)
        log_basket += w * np.log(prices)

    return log_basket


def _compute_log_tot(
    export_weights: dict[str, float],
    import_weights: dict[str, float],
    commodities: dict[str, pd.Series],
) -> pd.Series:
    """
    Compute log(ToT) = log(ExportIndex) - log(ImportIndex).

    For pure importers (empty export basket), returns -log(ImportIndex).
    This is correct: when import prices rise, the pure importer's ToT
    deteriorates, producing a negative signal.

    For pure exporters with non-commodity imports, the import basket
    defaults to Brent (generic energy import cost).
    """
    log_export = _compute_log_basket(export_weights, commodities)
    log_import = _compute_log_basket(import_weights, commodities)

    if log_export.empty and log_import.empty:
        return pd.Series(dtype=float)

    if log_export.empty:
        # Pure importer: ToT worsens when import prices rise
        return -log_import

    if log_import.empty:
        # Pure exporter (no commodity imports): ToT = export side only
        return log_export

    # Align
    common_idx = log_export.index.intersection(log_import.index).sort_values()
    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    return log_export.reindex(common_idx) - log_import.reindex(common_idx)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _rows(
    currency: str,
    metric_id: str,
    s: pd.Series,
) -> pd.DataFrame:
    """Convert a named Series into long-format rows for DB upsert."""
    s = s.dropna()
    if s.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "currency":               currency,
        "series_id":              metric_id,
        "time":                   s.index,
        "value":                  s.values,
        "estimated_release_date": s.index,  # commodity prices: available same day
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PER-CURRENCY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_currency(
    currency: str,
    cfg: dict,
    commodities: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute all ToT derived metrics for one currency."""
    logger.info("Building ToT derived for %s...", currency.upper())
    frames = []

    export_w = cfg.get("export", {})
    import_w = cfg.get("import", {})

    # Log how many basket components are available
    avail_exp = sum(1 for k in export_w if k in commodities)
    avail_imp = sum(1 for k in import_w if k in commodities)
    logger.info("  Export basket: %d/%d commodities available", avail_exp, len(export_w))
    logger.info("  Import basket: %d/%d commodities available", avail_imp, len(import_w))

    if avail_exp == 0 and avail_imp == 0:
        logger.warning("  %s: No commodity data available — skipping", currency)
        return pd.DataFrame()

    # ── Log ToT level ────────────────────────────────────────────────────
    log_tot = _compute_log_tot(export_w, import_w, commodities)
    if log_tot.empty:
        logger.warning("  %s: log_tot empty after basket construction", currency)
        return pd.DataFrame()

    frames.append(_rows(currency, f"{currency}_tot_log_level", log_tot))

    # ── 12-month momentum ────────────────────────────────────────────────
    tot_12m = (log_tot - log_tot.shift(MOM_12M)).round(6)
    frames.append(_rows(currency, f"{currency}_tot_12m_chg", tot_12m))

    # ── 3-month momentum ─────────────────────────────────────────────────
    tot_3m = (log_tot - log_tot.shift(MOM_3M)).round(6)
    frames.append(_rows(currency, f"{currency}_tot_3m_chg", tot_3m))

    # ── Log summary ──────────────────────────────────────────────────────
    last_level = log_tot.dropna().iloc[-1] if not log_tot.dropna().empty else float("nan")
    last_12m = tot_12m.dropna().iloc[-1] if not tot_12m.dropna().empty else float("nan")
    last_3m = tot_3m.dropna().iloc[-1] if not tot_3m.dropna().empty else float("nan")
    logger.info("  ✓ %s ToT: level=%.4f  12m_chg=%.4f  3m_chg=%.4f",
                currency.upper(), last_level, last_12m, last_3m)

    result = pd.concat(frames, ignore_index=True)
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

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()

    logger.info("Upsert complete: %s.%s rows=%d", schema, TARGET_TABLE, len(rows))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_store_tot_derived(
    conn,
    schema: str = "macro",
    currencies: list[str] | None = None,
) -> None:
    """
    Compute and store ToT derived metrics for all configured currencies.

    Parameters
    ----------
    conn : psycopg2 connection
        Open connection to the database.
    schema : str
        Database schema (default "macro").
    currencies : list[str] or None
        Subset of currencies to compute. None = all 8.
    """
    configs = BASKET_CONFIGS
    if currencies:
        configs = {k: v for k, v in BASKET_CONFIGS.items() if k in currencies}

    logger.info("═══ ToT Derived: loading commodity prices ═══")
    commodities = _load_all_commodities(conn, schema)

    if not commodities:
        logger.error("No commodity price data loaded — cannot compute ToT")
        return

    logger.info("Loaded %d commodity series", len(commodities))

    all_frames = []
    for currency, cfg in configs.items():
        df = _build_currency(currency, cfg, commodities)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No ToT derived data produced")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    try:
        _upsert(combined, conn, schema)
        logger.info(
            "✓ ToT derived complete: %d currencies, %d total rows",
            len(all_frames), len(combined),
        )
    except psycopg2.errors.UndefinedTable:
        logger.error("Table %s.%s does not exist — run setup SQL first.", schema, TARGET_TABLE)
        raise