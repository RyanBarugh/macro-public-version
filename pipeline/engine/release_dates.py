"""
pipeline/engine/release_dates.py

Computes estimated release dates for macro series data using the
publication_lags table. Used by derived layer to stamp each output row
with when the market actually had that information.

Usage in derived files:
    from pipeline.engine.release_dates import get_release_date_mapper

    mapper = get_release_date_mapper(conn)
    # For single-input metrics:
    df['estimated_release_date'] = df.apply(
        lambda r: mapper(r['series_id'], r['time']), axis=1
    )
    # For multi-input metrics (e.g. ppi_cpi_spread):
    df['estimated_release_date'] = df['time'].apply(
        lambda t: max(mapper('usd_ppi_final_demand_sa', t),
                      mapper('us_cpi_all_items_sa', t))
    )
"""

from __future__ import annotations

import pandas as pd
from datetime import timedelta


def period_end(observation_date: pd.Timestamp, frequency: str) -> pd.Timestamp:
    """
    Convert an observation date (period-start convention) to the last day
    of that period.

    series_data stores time as period-start:
        Monthly:   2025-01-01 → 2025-01-31
        Quarterly: 2025-01-01 → 2025-03-31
        Weekly:    2025-01-06 → 2025-01-06 (already the observation point)
        Daily:     2025-01-06 → 2025-01-06

    Uses pd.offsets.MonthEnd(0) which handles 28/29/30/31 day months
    automatically.
    """
    t = pd.Timestamp(observation_date)

    if frequency == 'M':
        return t + pd.offsets.MonthEnd(0)
    elif frequency == 'Q':
        return t + pd.offsets.QuarterEnd(0)
    elif frequency in ('W', 'D'):
        return t
    else:
        raise ValueError(f"Unknown frequency: {frequency}")


def estimated_release_date(
    observation_date: pd.Timestamp,
    frequency: str,
    avg_lag_days: int
) -> pd.Timestamp:
    """
    Compute when the market received this observation.

    estimated_release = period_end(observation_date, frequency) + avg_lag_days

    Negative lags are valid (current-month surveys published before
    the reference period ends).
    """
    pe = period_end(observation_date, frequency)
    return pe + timedelta(days=int(avg_lag_days))


def load_publication_lags(conn) -> pd.DataFrame:
    """
    Load the full publication_lags table into a DataFrame, indexed by
    series_id for fast lookup.
    """
    query = """
        SELECT series_id, currency, frequency, avg_lag_days,
               source_agency, publication_name
        FROM macro.publication_lags
    """
    df = pd.read_sql(query, conn)
    return df.set_index('series_id')


def get_release_date_mapper(conn):
    """
    Returns a function: mapper(series_id, observation_date) -> release_date

    Caches the lag table in memory on first call. Use this in derived files
    to stamp each row with its estimated release date.

    Example:
        mapper = get_release_date_mapper(conn)
        release = mapper('usd_ppi_final_demand_sa', pd.Timestamp('2025-01-01'))
        # → 2025-02-13 (Jan period end Jan 31 + 13 days)
    """
    lags_df = load_publication_lags(conn)

    def mapper(series_id: str, observation_date) -> pd.Timestamp:
        if series_id not in lags_df.index:
            raise KeyError(
                f"Series '{series_id}' not found in publication_lags. "
                f"Add it before computing release dates."
            )
        row = lags_df.loc[series_id]
        return estimated_release_date(
            observation_date, row['frequency'], row['avg_lag_days']
        )

    return mapper


def add_release_dates(
    df: pd.DataFrame,
    series_id: str,
    conn,
    time_col: str = 'time',
    _mapper=None
) -> pd.DataFrame:
    """
    Add estimated_release_date column to a DataFrame of single-series data.
    """
    if _mapper is None:
        _mapper = get_release_date_mapper(conn)

    df = df.copy()
    df['estimated_release_date'] = df[time_col].apply(
        lambda t: _mapper(series_id, t)
    )
    return df


def add_release_dates_multi(
    df: pd.DataFrame,
    input_series_ids: list,
    conn,
    time_col: str = 'time',
    _mapper=None
) -> pd.DataFrame:
    """
    Add estimated_release_date for a derived metric that depends on
    multiple input series. The release date is the MAX (latest) of
    all inputs' release dates for that observation period.
    """
    if _mapper is None:
        _mapper = get_release_date_mapper(conn)

    df = df.copy()
    df['estimated_release_date'] = df[time_col].apply(
        lambda t: max(_mapper(sid, t) for sid in input_series_ids)
    )
    return df