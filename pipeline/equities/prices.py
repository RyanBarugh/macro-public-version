"""
pipeline/equities/prices.py
=============================

Equity price ingestion: loads constituents, fetches EODHD daily
prices, writes to equity.constituents + equity.prices.

Single connection pattern: receives `conn` from the orchestrator.
Matches macro insert_to_db.py / provider pattern.

Functions:
    load_constituents_json()    — read constituents.json
    seed_constituents()         — populate equity.constituents table
    fetch_eod_prices()          — fetch one symbol from EODHD
    bulk_fetch_and_store()      — fetch all constituents + write to DB
"""

from __future__ import annotations

import json
import pandas as pd
import requests
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import logging
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

EODHD_BASE = 'https://eodhd.com/api'

SCHEMA = 'equity'
CONSTITUENTS_JSON = Path(__file__).parent / 'constituents.json'

BATCH_LOG_INTERVAL = 50          # log every N tickers
BACKFILL_START = '1998-01-01'
UPDATE_LOOKBACK_DAYS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTITUENT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def load_constituents_json() -> list[dict]:
    """Load constituents from the local JSON file."""
    if not CONSTITUENTS_JSON.exists():
        raise FileNotFoundError(
            f"Constituents file not found: {CONSTITUENTS_JSON}\n"
            f"Run: python -m pipeline.providers.sp1500 first"
        )
    with open(CONSTITUENTS_JSON) as f:
        data = json.load(f)
    logger.info("Loaded %d constituents from %s", len(data), CONSTITUENTS_JSON.name)
    return data


def seed_constituents(conn, constituents: list[dict]) -> None:
    """Populate equity.constituents table from JSON."""
    now = datetime.now(timezone.utc)

    rows = []
    for c in constituents:
        rows.append((
            c['ticker'],
            c.get('name', ''),
            c.get('sector', ''),
            c.get('sub_industry', ''),
            c.get('index_member', ''),
            c.get('exchange', 'US'),
            True,
            now,
        ))

    sql = f"""
        INSERT INTO {SCHEMA}.constituents
            (ticker, name, sector, sub_industry, index_member, exchange, active, added_at)
        VALUES %s
        ON CONFLICT (ticker)
        DO UPDATE SET
            name = EXCLUDED.name,
            sector = EXCLUDED.sector,
            sub_industry = EXCLUDED.sub_industry,
            index_member = EXCLUDED.index_member,
            exchange = EXCLUDED.exchange,
            active = EXCLUDED.active
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=500)
    conn.commit()

    logger.info("Seeded %d constituents to %s.constituents", len(rows), SCHEMA)

    # Summary
    counts = {}
    for c in constituents:
        idx = c.get('index_member', 'unknown')
        counts[idx] = counts.get(idx, 0) + 1
    for idx, count in sorted(counts.items()):
        logger.info("  %s: %d", idx, count)


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_eod_prices(
    symbol: str,
    start_date: str,
    session: requests.Session,
    timeout: Tuple[float, float],
    end_date: str = None,
    api_key: str = "",
) -> pd.DataFrame:
    """Fetch daily OHLCV for one symbol from EODHD using shared session."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    url = f"{EODHD_BASE}/eod/{symbol}"
    resp = session.get(url, params={
        'api_token': api_key,
        'from': start_date,
        'to': end_date,
        'fmt': 'json',
        'order': 'a',
    }, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df[['open', 'high', 'low', 'adjusted_close', 'volume']].rename(
        columns={'adjusted_close': 'close'}
    )


_TIMEOUT_BUFFER_MS = 60_000


def bulk_fetch_and_store(
    conn,
    constituents: list[dict],
    session: requests.Session,
    timeout: Tuple[float, float],
    api_key: str,
    start_date: str = BACKFILL_START,
    end_date: str = None,
    lambda_context=None,
) -> dict:
    """
    Fetch prices for all constituents and write to equity.prices.
    Uses the single conn + shared HTTP session from the orchestrator.
    Returns stats dict including latest_price_date.
    """
    now = datetime.now(timezone.utc)
    all_rows = []
    fetched = 0
    failed = 0
    skipped = 0
    failed_list = []
    latest_date = None

    logger.info("Fetching %d symbols from %s...", len(constituents), start_date)

    for i, c in enumerate(constituents):
        # ── Timeout guard (matches macro fetch loop) ──────────────
        if lambda_context is not None:
            remaining_ms = lambda_context.get_remaining_time_in_millis()
            if remaining_ms < _TIMEOUT_BUFFER_MS:
                skipped = len(constituents) - i
                logger.warning(
                    "Timeout approaching — %d tickers remaining, skipping",
                    skipped,
                )
                break

        ticker = c['ticker']
        exchange = c.get('exchange', 'US')
        symbol = f"{ticker}.{exchange}"

        try:
            df = fetch_eod_prices(
                symbol, start_date, session=session,
                timeout=timeout, end_date=end_date, api_key=api_key,
            )
            if not df.empty:
                for date, row in df.iterrows():
                    close = row['close']
                    if pd.notna(close):
                        row_date = date.date() if hasattr(date, 'date') else date
                        all_rows.append((
                            ticker,
                            row_date,
                            float(row['open']) if pd.notna(row['open']) else None,
                            float(row['high']) if pd.notna(row['high']) else None,
                            float(row['low']) if pd.notna(row['low']) else None,
                            float(close),
                            int(row['volume']) if pd.notna(row['volume']) else None,
                            now,
                        ))
                        if latest_date is None or row_date > latest_date:
                            latest_date = row_date
                fetched += 1
            else:
                logger.warning("Empty response for %s", symbol)
                failed += 1
                failed_list.append(symbol)

        except Exception as e:
            logger.warning("Failed %s: %s", symbol, str(e))
            failed += 1
            failed_list.append(symbol)

        if (i + 1) % BATCH_LOG_INTERVAL == 0:
            logger.info(
                "Progress: %d/%d fetched (%d failed, %d rows so far)",
                i + 1, len(constituents), failed, len(all_rows)
            )

    logger.info(
        "Fetch complete: %d succeeded, %d failed, %d skipped, %d total rows",
        fetched, failed, skipped, len(all_rows)
    )
    if failed_list:
        logger.info("Failed symbols (first 20): %s", failed_list[:20])

    # Write to DB using the shared connection
    if all_rows:
        _write_prices(conn, all_rows)

    return {
        'fetched': fetched,
        'failed': failed,
        'skipped': skipped,
        'rows_written': len(all_rows),
        'failed_symbols': failed_list,
        'latest_price_date': str(latest_date) if latest_date else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITE
# ═══════════════════════════════════════════════════════════════════════════════

def _write_prices(conn, rows: list[tuple], batch_size: int = 5000) -> None:
    """Write OHLCV rows to equity.prices using the shared connection."""
    sql = f"""
        INSERT INTO {SCHEMA}.prices
            (ticker, time, open, high, low, close, volume, updated_at)
        VALUES %s
        ON CONFLICT (ticker, time)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            updated_at = EXCLUDED.updated_at
    """

    with conn.cursor() as cur:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            execute_values(cur, sql, batch, page_size=1000)
            if (i + batch_size) < len(rows):
                logger.info("Written %d/%d rows...",
                            min(i + batch_size, len(rows)), len(rows))
    conn.commit()
    logger.info("Upserted %d rows to %s.prices", len(rows), SCHEMA)