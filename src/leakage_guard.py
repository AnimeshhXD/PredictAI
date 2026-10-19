"""
src/leakage_guard.py
====================
Data-leakage prevention utilities for PredictAI.

Rule
----
Any DataFrame or sequence of dates used for forecasting or retrieval MUST
contain only rows strictly BEFORE the scenario cutoff date.  Any access
to rows on or after the cutoff is a data-leakage violation.

Usage
-----
>>> from src.leakage_guard import assert_no_leakage, filter_before_cutoff
>>> import pandas as pd
>>>
>>> df = pd.read_csv("prices.csv", parse_dates=["date"])
>>> cutoff = pd.Timestamp("2023-01-01")
>>>
>>> clean = filter_before_cutoff(df, date_col="date", cutoff=cutoff)
>>> assert_no_leakage(clean, date_col="date", cutoff=cutoff)   # passes
>>> assert_no_leakage(df, date_col="date", cutoff=cutoff)      # raises if df has future rows
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

import pandas as pd

DateLike = Union[str, date, datetime, pd.Timestamp]


def _to_timestamp(d: DateLike) -> pd.Timestamp:
    """Coerce any date-like value to a tz-naive :class:`pd.Timestamp`."""
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


class DataLeakageError(ValueError):
    """Raised when future data is detected in a forecasting or retrieval context."""


def assert_no_leakage(
    df: pd.DataFrame,
    date_col: str,
    cutoff: DateLike,
) -> None:
    """
    Assert that every row in *df* has a date strictly before *cutoff*.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    date_col:
        Name of the datetime column to check.
    cutoff:
        The scenario cutoff date (exclusive upper bound).

    Raises
    ------
    DataLeakageError
        If one or more rows have ``df[date_col] >= cutoff``.
    KeyError
        If *date_col* is not present in *df*.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame. "
                       f"Available columns: {list(df.columns)}")

    cutoff_ts = _to_timestamp(cutoff)
    dates = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    leaking = df[dates >= cutoff_ts]

    if not leaking.empty:
        n = len(leaking)
        earliest_bad = dates[leaking.index].min()
        raise DataLeakageError(
            f"Data leakage detected! {n} row(s) with dates >= cutoff "
            f"'{cutoff_ts.date()}'. Earliest offending date: {earliest_bad.date()}. "
            "Remove future rows before passing data to forecasting or retrieval."
        )


def filter_before_cutoff(
    df: pd.DataFrame,
    date_col: str,
    cutoff: DateLike,
) -> pd.DataFrame:
    """
    Return a copy of *df* containing only rows strictly before *cutoff*.

    Parameters
    ----------
    df:
        Input DataFrame.
    date_col:
        Name of the datetime column used for filtering.
    cutoff:
        The scenario cutoff date (exclusive upper bound).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with rows where ``df[date_col] < cutoff``.

    Raises
    ------
    KeyError
        If *date_col* is not present in *df*.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame. "
                       f"Available columns: {list(df.columns)}")

    cutoff_ts = _to_timestamp(cutoff)
    dates = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    mask = dates < cutoff_ts
    filtered = df.loc[mask].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def validate_index_dates(
    series: pd.Series,
    cutoff: DateLike,
    *,
    label: str = "series",
) -> None:
    """
    Assert that a :class:`pd.Series` with a :class:`pd.DatetimeIndex` has no
    entries on or after *cutoff*.

    Parameters
    ----------
    series:
        Time-series with a DatetimeIndex.
    cutoff:
        Exclusive upper bound.
    label:
        Human-readable name used in the error message.

    Raises
    ------
    DataLeakageError
        If the index contains timestamps >= cutoff.
    TypeError
        If the series does not have a DatetimeIndex.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected a DatetimeIndex on '{label}', "
            f"got {type(series.index).__name__}."
        )

    cutoff_ts = _to_timestamp(cutoff)
    idx = series.index
    if idx.tzinfo is not None:
        idx = idx.tz_localize(None)

    bad = idx[idx >= cutoff_ts]
    if len(bad):
        raise DataLeakageError(
            f"Leakage in '{label}': {len(bad)} index timestamps >= cutoff "
            f"'{cutoff_ts.date()}'. First offender: {bad[0].date()}."
        )
