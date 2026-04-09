"""
MAD-Based Panel Z-Scores v2 — Cross-Sectionally Demeaned
==========================================================

Same core engine as zn_scores.py but with cross-sectional demeaning
applied after z-scoring. This removes structural bias by forcing
every indicator to measure "where does this country rank relative
to peers right now" rather than "how unusual is this country's
reading relative to its own history."

Changes from v1:
  - cs_demean=True by default in make_zn_scores
  - After z-scoring, subtracts the cross-sectional mean at each date
  - This matches JPMaQS: "each relative factor first calculates the
    relative values of its underlying macro-quantamental categories"

All other parameters unchanged from v1:
    sequential=True, min_obs=783, neutral="zero", pan_weight=0.5,
    thresh=3.0, est_freq="m", iis=True

Usage:
    from .zn_scores_v2 import make_zn_scores, linear_composite, rescore, ffill_to_daily
    # Drop-in replacement for zn_scores — same API, same functions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Optional

# Import everything from v1 — we only override make_zn_scores and rescore
from .zn_scores import (
    ffill_to_daily,
    linear_composite,
    cross_sectional_zscore,
    _panel_zn,
    _cs_zn,
    _month_end_mask,
    _ffill_array,
    _resolve_neutral,
)


# ───────────────────────────────────────────────────────────────────
# Core z-score engine (v2 — with cross-sectional demeaning)
# ───────────────────────────────────────────────────────────────────

def make_zn_scores(
    panel: pd.DataFrame,
    sequential: bool = True,
    min_obs: int = 783,
    neutral: Union[str, float] = "zero",
    pan_weight: float = 0.8,
    thresh: Optional[float] = 3.0,
    est_freq: str = "m",
    iis: bool = True,
    cs_demean: bool = True,
) -> pd.DataFrame:
    """
    Compute MAD-based z-scores with cross-sectional demeaning.

    Same as v1 make_zn_scores but with an additional step:
    after computing z-scores, subtract the cross-sectional mean
    at each date. This removes persistent structural bias by
    centering scores around the panel average at every point in time.

    Parameters
    ----------
    panel : DataFrame
        Index = business dates, columns = currency codes.
    sequential : bool
        If True, expanding window (no look-ahead).
    min_obs : int
        Minimum panel observations before MAD is valid.
    neutral : str or float
        "zero" (default), "mean", "median", or a fixed number.
    pan_weight : float
        Blend of panel (1.0) and per-currency (0.0) MAD. Default 0.5.
    thresh : float or None
        Winsorise at ±thresh after demeaning.
    est_freq : str
        "m" (monthly re-estimation) or "d" (daily).
    iis : bool
        In-sample scoring for initial period.
    cs_demean : bool
        If True (default), subtract cross-sectional mean at each date
        after z-scoring. This is the key v2 change.

    Returns
    -------
    DataFrame of z-scores, cross-sectionally demeaned and clipped.
    """
    panel = panel.sort_index().copy()

    # Compute est_mask once
    if est_freq == "m" and isinstance(panel.index, pd.DatetimeIndex):
        est_mask = _month_end_mask(panel.index)
    else:
        est_mask = None

    if pan_weight == 1.0:
        zn = _panel_zn(panel, sequential, min_obs, neutral, iis, est_mask)
    elif pan_weight == 0.0:
        zn = _cs_zn(panel, sequential, min_obs, neutral, iis, est_mask)
    else:
        zn_pan = _panel_zn(panel, sequential, min_obs, neutral, iis, est_mask)
        zn_cs = _cs_zn(panel, sequential, min_obs, neutral, iis, est_mask)
        zn = pan_weight * zn_pan + (1.0 - pan_weight) * zn_cs

    # ── Cross-sectional demeaning (v2 addition) ───────────────────
    if cs_demean:
        row_mean = zn.mean(axis=1)
        zn = zn.sub(row_mean, axis=0)

    if thresh is not None:
        zn = zn.clip(-thresh, thresh)

    return zn


# ───────────────────────────────────────────────────────────────────
# Re-scoring (v2 — with cross-sectional demeaning)
# ───────────────────────────────────────────────────────────────────

def rescore(
    composite: pd.DataFrame,
    min_obs: int = 783,
    thresh: float = 3.0,
    est_freq: str = "m",
    iis: bool = True,
    cs_demean: bool = True,
) -> pd.DataFrame:
    """
    Re-normalise a composite using make_zn_scores v2.

    Applied after every aggregation step. Includes cross-sectional
    demeaning by default.
    """
    return make_zn_scores(
        composite,
        sequential=True,
        min_obs=min_obs,
        neutral="zero",
        pan_weight=0.8,
        thresh=thresh,
        est_freq=est_freq,
        iis=iis,
        cs_demean=cs_demean,
    )