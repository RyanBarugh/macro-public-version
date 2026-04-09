from __future__ import annotations

"""
run_rs.py
=========
RS rankings only — CLI entry point.
Recomputes relative strength from existing prices, no fetch.

Usage:
    python -m pipeline.equities.run_rs
"""

from .core import run_rs_only


def main() -> None:
    run_rs_only()


if __name__ == "__main__":
    main()