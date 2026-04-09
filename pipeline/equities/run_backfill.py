from __future__ import annotations

"""
run_backfill.py
===============
Full equity backfill — CLI entry point.
Fetches full price history from 1998, then runs derived (breadth).

Usage:
    python -m pipeline.equities.run_backfill
"""

from .core import run_pipeline


def main() -> None:
    run_pipeline(run_type="backfill")


if __name__ == "__main__":
    main()