from __future__ import annotations

"""
run_vcp.py
==========
VCP scanner only — CLI entry point.
Scans for volatility contraction patterns from existing prices + RS data.

Usage:
    python -m pipeline.equities.run_vcp
"""

from .core import run_vcp_only


def main() -> None:
    run_vcp_only()


if __name__ == "__main__":
    main()