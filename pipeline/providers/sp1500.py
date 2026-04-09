"""
Scrape S&P 500, 400, 600 constituents from Wikipedia → equity_constituents.json

Usage:
    python scrape_sp1500.py
    python scrape_sp1500.py --output my_file.json
"""

import json
import argparse
import requests
import pandas as pd
from io import StringIO

SOURCES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

HEADERS = {
    "User-Agent": "EdgeFlow/1.0 (macro research; contact: edgeflow@example.com)"
}

# Column mapping — Wikipedia tables vary slightly across pages
# We normalise to a consistent schema
COLUMN_MAP = {
    "symbol": "ticker",
    "ticker": "ticker",
    "ticker symbol": "ticker",
    "security": "name",
    "company": "name",
    "company name": "name",
    "gics sector": "sector",
    "sector": "sector",
    "gics sub-industry": "sub_industry",
    "gics sub industry": "sub_industry",
    "sub-industry": "sub_industry",
    "sub_industry": "sub_industry",
}


def scrape_index(url: str, index_name: str) -> list[dict]:
    """Scrape a single Wikipedia S&P constituent table."""
    print(f"  Fetching {index_name} from {url}...")

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    # The constituents table is typically the first (and largest) table
    df = None
    for t in tables:
        # Look for a table with a column that maps to 'ticker'
        cols_lower = [str(c).lower().strip() for c in t.columns]
        if any(c in COLUMN_MAP and COLUMN_MAP[c] == "ticker" for c in cols_lower):
            if df is None or len(t) > len(df):
                df = t

    if df is None:
        print(f"  WARNING: Could not find constituents table for {index_name}")
        return []

    # Normalise column names
    df.columns = [str(c).lower().strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]
    df = df.rename(columns=rename)

    if "ticker" not in df.columns:
        print(f"  WARNING: No ticker column found for {index_name}")
        print(f"  Columns: {list(df.columns)}")
        return []

    entries = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        # Handle BRK.B → BRK-B for EODHD compatibility
        ticker = ticker.replace(".", "-")

        entry = {
            "ticker": ticker,
            "name": str(row.get("name", "")).strip(),
            "sector": str(row.get("sector", "")).strip(),
            "sub_industry": str(row.get("sub_industry", "")).strip(),
            "index_member": index_name,
            "exchange": "US",
        }
        entries.append(entry)

    print(f"  ✓ {index_name}: {len(entries)} constituents")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Scrape S&P 1500 constituents from Wikipedia")
    parser.add_argument("--output", default="pipeline/equities/constituents.json", help="Output JSON path")
    args = parser.parse_args()

    print("═══════════════════════════════════════════")
    print("  S&P 1500 Constituent Scraper (Wikipedia)")
    print("═══════════════════════════════════════════")

    all_entries = []
    seen_tickers = set()

    for index_name, url in SOURCES.items():
        entries = scrape_index(url, index_name)
        for e in entries:
            if e["ticker"] not in seen_tickers:
                all_entries.append(e)
                seen_tickers.add(e["ticker"])
            else:
                print(f"  DUPE: {e['ticker']} already in set (skipping {index_name} entry)")

    # Sort by index then ticker
    index_order = {"sp500": 0, "sp400": 1, "sp600": 2}
    all_entries.sort(key=lambda x: (index_order.get(x["index_member"], 9), x["ticker"]))

    with open(args.output, "w") as f:
        json.dump(all_entries, f, indent=2)

    # Summary
    counts = {}
    sectors = {}
    for e in all_entries:
        idx = e["index_member"]
        counts[idx] = counts.get(idx, 0) + 1
        sec = e["sector"]
        sectors[sec] = sectors.get(sec, 0) + 1

    print()
    print("═══════════════════════════════════════════")
    print(f"  Total: {len(all_entries)} constituents → {args.output}")
    print("═══════════════════════════════════════════")
    for idx in ["sp500", "sp400", "sp600"]:
        print(f"  {idx}: {counts.get(idx, 0)}")
    print()
    print("  Sector breakdown:")
    for sec, count in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"    {sec}: {count}")


if __name__ == "__main__":
    main()