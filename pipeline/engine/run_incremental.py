from __future__ import annotations

"""
run_incremental.py
==================
Daily/weekly incremental run — CLI entry point.
Lookback window is computed inside run_pipeline from cfg.lookback_months.
"""

import argparse

from .core import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental macro pipeline")
    parser.add_argument("--series", nargs="+", metavar="SERIES_ID",
                        help="Run only these series IDs (space-separated)")
    parser.add_argument("--currency", nargs="+", metavar="CCY",
                        help="Run only these currencies, e.g. usd eur (space-separated)")
    parser.add_argument("--derived", nargs="+", metavar="MODULE",
                        help="Run only these derived modules (space-separated)")
    args = parser.parse_args()

    run_pipeline(
        run_type="incremental",
        series_filter=args.series,
        currencies_filter=args.currency,
        derived_filter=args.derived,
    )


if __name__ == "__main__":
    main()