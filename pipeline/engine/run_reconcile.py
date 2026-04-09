from __future__ import annotations

"""
run_reconcile.py
================
Monthly reconciliation run — fetches full history to catch revisions
outside the incremental window.
Run manually once per month, or schedule separately from incremental.
"""

import argparse

from .core import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile macro pipeline")
    parser.add_argument("--series", nargs="+", metavar="SERIES_ID",
                        help="Run only these series IDs (space-separated)")
    parser.add_argument("--currency", nargs="+", metavar="CCY",
                        help="Run only these currencies, e.g. usd eur (space-separated)")
    args = parser.parse_args()

    run_pipeline(
        run_type="reconcile",
        series_filter=args.series,
        currencies_filter=args.currency,
    )


if __name__ == "__main__":
    main()