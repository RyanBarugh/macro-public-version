"""
Relative Value — Benchmark Subtraction
========================================

Implements JPMaQS `make_relative_value` for FX scorecards:
subtract the benchmark currency's metric from each currency
BEFORE z-scoring.

Benchmark groups (8 majors):
    USD-based: eur, gbp, aud, cad, jpy, nzd  → subtract usd
    EUR-based: chf                            → subtract eur
    Dual:      (none currently — GBP is USD-based in our setup)

USD itself gets score 0 in the USD group (subtracts itself).
EUR gets score 0 in the EUR group.

Usage:
    panel = ...  # wide DataFrame: dates × currencies
    rel = make_relative_value(panel)
    zn = make_zn_scores(rel)  # z-score the differentials
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


# ───────────────────────────────────────────────────────────────────
# Benchmark group definitions
# ───────────────────────────────────────────────────────────────────

# Default G10 grouping — maps each currency to its benchmark(s).
# Single benchmark = subtract that currency's value.
# Multiple benchmarks = subtract their mean.
DEFAULT_BENCHMARKS: dict[str, list[str]] = {
    "usd": ["usd"],     # self-reference → 0
    "eur": ["usd"],
    "gbp": ["usd"],
    "aud": ["usd"],
    "cad": ["usd"],
    "jpy": ["usd"],
    "nzd": ["usd"],
    "chf": ["eur"],
}


# ───────────────────────────────────────────────────────────────────
# Core function
# ───────────────────────────────────────────────────────────────────

def make_relative_value(
    panel: pd.DataFrame,
    benchmarks: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Subtract each currency's benchmark from its metric values.

    Parameters
    ----------
    panel : DataFrame
        Index = dates, columns = currency codes, values = metric.
    benchmarks : dict or None
        Maps each currency to a list of benchmark currencies.
        If None, uses DEFAULT_BENCHMARKS. Only currencies present
        in both the panel columns and the benchmarks dict are processed.

    Returns
    -------
    DataFrame with same shape — each currency's values expressed
    relative to its benchmark. Currencies not in the benchmarks
    dict are passed through unchanged.
    """
    bm = benchmarks if benchmarks is not None else DEFAULT_BENCHMARKS
    panel = panel.sort_index().copy()

    result = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)

    for ccy in panel.columns:
        if ccy not in bm:
            # No benchmark defined — pass through raw values
            result[ccy] = panel[ccy]
            continue

        bm_ccys = bm[ccy]
        # Filter to benchmarks actually present in the panel
        available_bm = [b for b in bm_ccys if b in panel.columns]

        if not available_bm:
            # Benchmark currency not in panel — pass through
            result[ccy] = panel[ccy]
            continue

        # Benchmark value = mean of benchmark currencies at each date
        if len(available_bm) == 1:
            bm_vals = panel[available_bm[0]]
        else:
            bm_vals = panel[available_bm].mean(axis=1)

        result[ccy] = panel[ccy] - bm_vals

    return result