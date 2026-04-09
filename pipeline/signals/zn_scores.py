"""
MAD-Based Panel Z-Scores — JPMaQS-Aligned Scoring Engine
==========================================================

Implements the core normalisation from macrosynergy's make_zn_scores:

    zn = (value - neutral) / expanding_panel_MAD

where MAD is the expanding mean of |value - neutral| pooled across
all currencies at all dates up to the estimation point.

Parameters locked to JPMaQS FX scorecard overrides:
    sequential=True   — expanding window, no look-ahead
    min_obs=783       — 3 years of business days (261 × 3)
    neutral="zero"    — excess metrics centred by construction
    pan_weight=1.0    — pure panel pooling
    thresh=3.0        — winsorise at ±3 MADs
    est_freq="m"      — MAD re-estimated monthly, ffilled between
    iis=True          — backfill initial period with first valid MAD

Also provides:
    ffill_to_daily            — expand monthly/quarterly panel to daily business days
    linear_composite          — equal-weight average of multiple score panels
    rescore                   — re-apply make_zn_scores to a composite (legacy)
    cross_sectional_zscore    — z-score across currencies at each date (Pete's Step 2)

All functions operate on wide-format DataFrames:
    index  = dates (sorted ascending)
    columns = currency codes (e.g. usd, eur, gbp, ...)
    values  = metric values or z-scores
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Optional


# ───────────────────────────────────────────────────────────────────
# Daily ffill — convert native-frequency panel to business days
# ───────────────────────────────────────────────────────────────────

def ffill_to_daily(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a monthly/quarterly panel to daily business day frequency
    using forward-fill, matching JPMaQS storage convention.

    Parameters
    ----------
    panel : DataFrame
        Index = dates (any frequency), columns = currencies.

    Returns
    -------
    DataFrame reindexed to business days with forward-filled values.
    """
    panel = panel.sort_index().copy()
    if panel.empty:
        return panel

    start = panel.index[0]
    today = pd.Timestamp.today().normalize()
    end = today
    bdays = pd.bdate_range(start, end)

    return panel.reindex(bdays).ffill()


# ───────────────────────────────────────────────────────────────────
# Month-end mask for est_freq="m"
# ───────────────────────────────────────────────────────────────────

def _month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Boolean mask: True at the last business day of each month.
    """
    months = index.to_period("M")
    is_last = np.zeros(len(index), dtype=bool)
    for i in range(len(index) - 1):
        if months[i] != months[i + 1]:
            is_last[i] = True
    is_last[-1] = True  # last date is always an estimation point
    return is_last


def _ffill_array(arr: np.ndarray, est_mask: np.ndarray) -> np.ndarray:
    """
    Forward-fill a 1D array, keeping values only at est_mask=True
    positions and filling forward to all other positions.
    """
    result = np.full_like(arr, np.nan)
    result[est_mask] = arr[est_mask]

    last_valid = np.nan
    for i in range(len(result)):
        if est_mask[i]:
            last_valid = result[i]
        else:
            result[i] = last_valid

    return result


# ───────────────────────────────────────────────────────────────────
# Core z-score engine
# ───────────────────────────────────────────────────────────────────

def make_zn_scores(
    panel: pd.DataFrame,
    sequential: bool = True,
    min_obs: int = 783,
    neutral: Union[str, float] = "zero",
    pan_weight: float = 1.0,
    thresh: Optional[float] = 3.0,
    est_freq: str = "m",
    iis: bool = True,
) -> pd.DataFrame:
    """
    Compute MAD-based z-scores across a panel of currencies.

    Parameters
    ----------
    panel : DataFrame
        Index = business dates (sorted), columns = currency codes.
        Should be ffilled to daily via ffill_to_daily() before calling.
        NaN where data is not yet available for a currency.
    sequential : bool
        If True, use expanding window (no look-ahead).
    min_obs : int
        Minimum total panel observations (across all currencies and dates)
        before MAD is considered valid. 783 = 3 years × 261 business days.
    neutral : str or float
        "zero" — deviations from zero (default for excess metrics).
        "mean" — deviations from expanding panel mean.
        float — deviations from a fixed value.
    pan_weight : float
        1.0 = pure panel-pooled MAD (default).
        0.0 = pure per-currency MAD.
        Between 0–1 = linear blend.
    thresh : float or None
        Winsorise z-scores at ±thresh. JPMaQS uses 3.0.
    est_freq : str
        "m" — re-estimate MAD at month-ends, ffill between (default).
        "d" — re-estimate daily (every observation).
    iis : bool
        In-sample scoring: backfill initial min_obs period using
        the first valid MAD.

    Returns
    -------
    DataFrame with same shape as input, containing z-scores.
    """
    panel = panel.sort_index().copy()

    # Compute est_mask once
    if est_freq == "m" and isinstance(panel.index, pd.DatetimeIndex):
        est_mask = _month_end_mask(panel.index)
    else:
        est_mask = None  # estimate at every date

    if pan_weight == 1.0:
        zn = _panel_zn(panel, sequential, min_obs, neutral, iis, est_mask)
    elif pan_weight == 0.0:
        zn = _cs_zn(panel, sequential, min_obs, neutral, iis, est_mask)
    else:
        zn_pan = _panel_zn(panel, sequential, min_obs, neutral, iis, est_mask)
        zn_cs = _cs_zn(panel, sequential, min_obs, neutral, iis, est_mask)
        zn = pan_weight * zn_pan + (1.0 - pan_weight) * zn_cs

    if thresh is not None:
        zn = zn.clip(-thresh, thresh)

    return zn


# ───────────────────────────────────────────────────────────────────
# Panel-pooled z-scores
# ───────────────────────────────────────────────────────────────────

def _panel_zn(
    panel: pd.DataFrame,
    sequential: bool,
    min_obs: int,
    neutral: Union[str, float],
    iis: bool,
    est_mask: Optional[np.ndarray],
) -> pd.DataFrame:
    """Z-scores using panel-wide (pooled across all currencies) MAD."""

    vals = panel.values.astype(float)
    mask = ~np.isnan(vals)
    T, N = vals.shape

    neutral_arr = _resolve_neutral(vals, mask, neutral, sequential)

    dev = vals - neutral_arr
    abs_dev = np.abs(dev)
    abs_dev_clean = np.where(mask, abs_dev, 0.0)

    if sequential:
        row_sum = np.nansum(abs_dev_clean, axis=1)
        row_count = mask.sum(axis=1)

        cum_sum = np.cumsum(row_sum)
        cum_count = np.cumsum(row_count)

        with np.errstate(divide="ignore", invalid="ignore"):
            panel_mad = np.where(
                cum_count >= min_obs,
                cum_sum / cum_count,
                np.nan,
            )

        # est_freq="m": only commit MAD at month-ends, ffill between
        if est_mask is not None:
            panel_mad = _ffill_array(panel_mad, est_mask)

        # IIS: backfill first valid MAD
        if iis:
            valid_mask = ~np.isnan(panel_mad)
            if valid_mask.any():
                first_valid = np.argmax(valid_mask)
                panel_mad[:first_valid] = panel_mad[first_valid]

        with np.errstate(divide="ignore", invalid="ignore"):
            zn = dev / panel_mad[:, np.newaxis]

    else:
        all_abs = abs_dev[mask]
        full_mad = np.mean(all_abs) if len(all_abs) > 0 else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            zn = dev / full_mad

    zn = np.where(mask, zn, np.nan)

    return pd.DataFrame(zn, index=panel.index, columns=panel.columns)


# ───────────────────────────────────────────────────────────────────
# Per-currency (cross-section) z-scores
# ───────────────────────────────────────────────────────────────────

def _cs_zn(
    panel: pd.DataFrame,
    sequential: bool,
    min_obs: int,
    neutral: Union[str, float],
    iis: bool,
    est_mask: Optional[np.ndarray],
) -> pd.DataFrame:
    """Z-scores using per-currency expanding MAD."""

    vals = panel.values.astype(float)
    mask = ~np.isnan(vals)
    T, N = vals.shape

    neutral_arr = _resolve_neutral(vals, mask, neutral, sequential)
    dev = vals - neutral_arr
    abs_dev = np.abs(dev)
    abs_dev_clean = np.where(mask, abs_dev, 0.0)

    if sequential:
        cum_sum = np.cumsum(abs_dev_clean, axis=0)
        cum_count = np.cumsum(mask.astype(int), axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            cs_mad = np.where(
                cum_count >= min_obs,
                cum_sum / cum_count,
                np.nan,
            )

        if est_mask is not None:
            for j in range(N):
                cs_mad[:, j] = _ffill_array(cs_mad[:, j], est_mask)

        if iis:
            for j in range(N):
                col_mad = cs_mad[:, j]
                valid_mask = ~np.isnan(col_mad)
                if valid_mask.any():
                    valid_idx = np.argmax(valid_mask)
                    col_mad[:valid_idx] = col_mad[valid_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            zn = dev / cs_mad

    else:
        cs_mad = np.nanmean(abs_dev, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            zn = dev / cs_mad[np.newaxis, :]

    zn = np.where(mask, zn, np.nan)

    return pd.DataFrame(zn, index=panel.index, columns=panel.columns)


# ───────────────────────────────────────────────────────────────────
# Neutral computation
# ───────────────────────────────────────────────────────────────────

def _resolve_neutral(
    vals: np.ndarray,
    mask: np.ndarray,
    neutral: Union[str, float],
    sequential: bool,
) -> Union[float, np.ndarray]:
    """Compute the neutral level."""
    if neutral == "zero" or neutral == 0:
        return 0.0

    if isinstance(neutral, (int, float)):
        return float(neutral)

    T, N = vals.shape
    clean = np.where(mask, vals, 0.0)  

    if neutral == "mean":
        if sequential:
            row_sum = np.sum(clean, axis=1)
            row_count = mask.sum(axis=1)
            cum_sum = np.cumsum(row_sum)
            cum_count = np.cumsum(row_count)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(cum_count > 0, cum_sum / cum_count, 0.0)
            return result[:, np.newaxis]
        else:
            return np.nanmean(vals)
    

    if neutral == "median":
        if sequential:
            result = np.empty((T, N))
            for j in range(N):
                for t in range(T):
                    col = vals[:t + 1, j]
                    col_mask = mask[:t + 1, j]
                    pool = col[col_mask]
                    result[t, j] = np.median(pool) if len(pool) > 0 else 0.0
            return result  # shape (T, N) — per-country
        else:
            return np.nanmedian(vals, axis=0)

    raise ValueError(f"Unknown neutral: {neutral!r}")


# ───────────────────────────────────────────────────────────────────
# Compositing: equal-weight average
# ───────────────────────────────────────────────────────────────────

def linear_composite(
    panels: dict[str, pd.DataFrame],
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Equal-weight (or custom-weight) average of multiple score panels.

    Follows JPMaQS `linear_composite` with `complete_xcats=False`:
    missing constituents are excluded from the average (partial data OK).
    """
    if not panels:
        raise ValueError("panels dict is empty")

    labels = list(panels.keys())
    dfs = list(panels.values())

    all_idx = dfs[0].index
    for df in dfs[1:]:
        all_idx = all_idx.union(df.index)
    all_idx = all_idx.sort_values()

    cols = dfs[0].columns
    aligned = [df.reindex(index=all_idx, columns=cols) for df in dfs]

    if weights is None:
        stacked = np.stack([df.values for df in aligned], axis=0)
        with np.errstate(invalid="ignore"):
            result = np.nanmean(stacked, axis=0)
        all_nan = np.all(np.isnan(stacked), axis=0)
        result = np.where(all_nan, np.nan, result)
    else:
        w = np.array([weights.get(lab, 1.0) for lab in labels])
        stacked = np.stack([df.values for df in aligned], axis=0)
        mask = ~np.isnan(stacked)
        w_3d = w[:, np.newaxis, np.newaxis] * mask.astype(float)
        w_sum = w_3d.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                w_sum > 0,
                np.nansum(stacked * w_3d, axis=0) / w_sum,
                np.nan,
            )

    return pd.DataFrame(result, index=all_idx, columns=cols)


# ───────────────────────────────────────────────────────────────────
# Re-scoring convenience
# ───────────────────────────────────────────────────────────────────

def rescore(
    composite: pd.DataFrame,
    min_obs: int = 783,
    thresh: float = 3.0,
    est_freq: str = "m",
    iis: bool = True,
) -> pd.DataFrame:
    """
    Re-normalise a composite using make_zn_scores.

    Applied after every aggregation step to compensate for
    diversification shrinkage (averaging N z-scores compresses
    MAD to ~1/sqrt(N)).
    """
    return make_zn_scores(
        composite,
        sequential=True,
        min_obs=min_obs,
        neutral="zero",
        pan_weight=0.3,
        thresh=thresh,
        est_freq=est_freq,
        iis=iis,
    )


# ───────────────────────────────────────────────────────────────────
# Cross-sectional z-score (Pete's Step 2)
# ───────────────────────────────────────────────────────────────────

def cross_sectional_zscore(
    panel: pd.DataFrame,
    thresh: float = 2.5,
    min_currencies: int = 4,
) -> pd.DataFrame:
    """
    Z-score across currencies at each date (cross-sectional, not through time).

    At each row: z_i = (x_i - mean(x)) / std(x) where mean/std are
    computed across the 8 currencies at that single point in time.

    This converts "how strong is this country's macro" into "where does
    this country rank relative to the other 7 right now."  No expanding
    window, no estimation drift — the score only changes when the
    underlying values change.

    Parameters
    ----------
    panel : DataFrame
        Index = dates, columns = currency codes, values = raw composite scores.
    thresh : float
        Clip at ±thresh. Pete uses 2.5.
    min_currencies : int
        Minimum non-NaN currencies required at a date to compute z-scores.

    Returns
    -------
    DataFrame of cross-sectional z-scores, clipped at ±thresh.
    """
    panel = panel.copy()

    # Count non-NaN currencies per row
    valid_count = panel.notna().sum(axis=1)

    # Cross-sectional mean and std at each date
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1, ddof=1)  # sample std across currencies

    # Guard against zero std (all currencies identical)
    row_std = row_std.replace(0.0, np.nan)

    # Z-score: (value - cross_sectional_mean) / cross_sectional_std
    zn = panel.sub(row_mean, axis=0).div(row_std, axis=0)

    # Mask rows with insufficient currencies
    insufficient = valid_count < min_currencies
    if insufficient.any():
        zn.loc[insufficient] = np.nan

    return zn.clip(-thresh, thresh)