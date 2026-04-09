from __future__ import annotations

"""
run_incremental.py
==================
Daily incremental equity update — CLI entry point.
Fetches last 10 days of prices, then runs derived (breadth).

Usage:
    python -m pipeline.equities.run_incremental
"""

from .core import run_pipeline


def main() -> None:
    run_pipeline(run_type="incremental")


if __name__ == "__main__":
    main()