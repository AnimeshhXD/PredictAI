"""
tests/test_leakage_guard.py
===========================
Tests for the data-leakage prevention module.

CRITICAL: The test ``test_future_data_raises`` MUST FAIL (i.e. the code under
test must raise) when future rows are present.  If the leakage guard ever lets
future data through silently, this test will fail, alerting the team.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.leakage_guard import (
    DataLeakageError,
    assert_no_leakage,
    filter_before_cutoff,
    validate_index_dates,
)

CUTOFF = "2023-01-01"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_df(start: str, periods: int, freq: str = "D") -> pd.DataFrame:
    """Return a DataFrame with a 'date' column and a dummy 'value' column."""
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    return pd.DataFrame({"date": dates, "value": np.arange(periods, dtype=float)})


# ── assert_no_leakage ──────────────────────────────────────────────────────────


class TestAssertNoLeakage:
    """Tests for :func:`assert_no_leakage`."""

    def test_all_before_cutoff_passes(self) -> None:
        """A DataFrame entirely before the cutoff must NOT raise."""
        df = _make_df("2022-01-01", 365)  # 2022 only
        assert_no_leakage(df, date_col="date", cutoff=CUTOFF)  # no exception

    def test_future_data_raises(self) -> None:
        """
        CRITICAL: A DataFrame with rows on or after the cutoff MUST raise
        DataLeakageError.  If this test passes without an exception being
        raised then data leakage is silently allowed — which is a bug.
        """
        df = _make_df("2022-12-25", 30)  # spans into 2023
        with pytest.raises(DataLeakageError):
            assert_no_leakage(df, date_col="date", cutoff=CUTOFF)

    def test_exactly_on_cutoff_raises(self) -> None:
        """A row exactly on the cutoff date is future data (cutoff is exclusive)."""
        df = pd.DataFrame({"date": [pd.Timestamp(CUTOFF)], "value": [0.0]})
        with pytest.raises(DataLeakageError):
            assert_no_leakage(df, date_col="date", cutoff=CUTOFF)

    def test_one_day_before_cutoff_passes(self) -> None:
        """A row one day before the cutoff must be accepted."""
        df = pd.DataFrame({"date": [pd.Timestamp("2022-12-31")], "value": [0.0]})
        assert_no_leakage(df, date_col="date", cutoff=CUTOFF)  # no exception

    def test_missing_date_col_raises_key_error(self) -> None:
        """A KeyError must be raised when the requested column is absent."""
        df = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=5)})
        with pytest.raises(KeyError):
            assert_no_leakage(df, date_col="date", cutoff=CUTOFF)

    def test_empty_dataframe_passes(self) -> None:
        """An empty DataFrame has no rows, so no leakage is possible."""
        df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": []})
        assert_no_leakage(df, date_col="date", cutoff=CUTOFF)

    def test_error_message_contains_count(self) -> None:
        """The error message must report how many offending rows were found."""
        df = _make_df("2023-01-01", 10)  # 10 rows on/after cutoff
        with pytest.raises(DataLeakageError, match="10 row"):
            assert_no_leakage(df, date_col="date", cutoff=CUTOFF)


# ── filter_before_cutoff ───────────────────────────────────────────────────────


class TestFilterBeforeCutoff:
    """Tests for :func:`filter_before_cutoff`."""

    def test_filters_future_rows(self) -> None:
        """Rows on/after the cutoff must be dropped."""
        df = _make_df("2022-12-25", 20)  # ends 2023-01-13
        filtered = filter_before_cutoff(df, date_col="date", cutoff=CUTOFF)
        assert (pd.to_datetime(filtered["date"]) < pd.Timestamp(CUTOFF)).all()

    def test_filtered_result_has_no_leakage(self) -> None:
        """The output of filter_before_cutoff must always pass assert_no_leakage."""
        df = _make_df("2022-01-01", 730)  # two years
        filtered = filter_before_cutoff(df, date_col="date", cutoff=CUTOFF)
        assert_no_leakage(filtered, date_col="date", cutoff=CUTOFF)

    def test_all_before_cutoff_unchanged_length(self) -> None:
        """If all rows are before the cutoff, the length must be preserved."""
        df = _make_df("2022-01-01", 365)
        filtered = filter_before_cutoff(df, date_col="date", cutoff=CUTOFF)
        assert len(filtered) == len(df)

    def test_all_after_cutoff_returns_empty(self) -> None:
        """If all rows are on/after the cutoff, the result must be empty."""
        df = _make_df("2023-01-01", 30)
        filtered = filter_before_cutoff(df, date_col="date", cutoff=CUTOFF)
        assert filtered.empty

    def test_index_is_reset(self) -> None:
        """The returned DataFrame must have a clean 0-based integer index."""
        df = _make_df("2022-06-01", 300)
        filtered = filter_before_cutoff(df, date_col="date", cutoff=CUTOFF)
        assert list(filtered.index) == list(range(len(filtered)))

    def test_missing_col_raises(self) -> None:
        df = pd.DataFrame({"ts": pd.date_range("2022-01-01", periods=5)})
        with pytest.raises(KeyError):
            filter_before_cutoff(df, date_col="date", cutoff=CUTOFF)


# ── validate_index_dates ───────────────────────────────────────────────────────


class TestValidateIndexDates:
    """Tests for :func:`validate_index_dates`."""

    def _make_series(self, start: str, periods: int) -> pd.Series:
        idx = pd.date_range(start=start, periods=periods, freq="D")
        return pd.Series(np.random.rand(periods), index=idx)

    def test_clean_series_passes(self) -> None:
        s = self._make_series("2022-01-01", 365)
        validate_index_dates(s, cutoff=CUTOFF)

    def test_future_index_raises(self) -> None:
        s = self._make_series("2022-12-01", 60)  # extends into 2023
        with pytest.raises(DataLeakageError):
            validate_index_dates(s, cutoff=CUTOFF)

    def test_non_datetime_index_raises_type_error(self) -> None:
        s = pd.Series([1, 2, 3], index=[0, 1, 2])
        with pytest.raises(TypeError):
            validate_index_dates(s, cutoff=CUTOFF)
