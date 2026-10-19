"""
tests/test_data_loader.py
=========================
Tests for src/data_loader.py data loading and synthetic M5 generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    generate_synthetic_m5,
    load_m5,
    m5_files_present,
)


class TestSyntheticM5Generator:
    """Tests for synthetic M5 dataset generator."""

    def test_synthetic_schema(self) -> None:
        """Synthetic daily DataFrame must have all required M5 columns."""
        daily, cal, prices = generate_synthetic_m5(
            seed=42, n_days=60, items_per_dept=2
        )
        expected_cols = {
            "date",
            "item_id",
            "store_id",
            "series_id",
            "sales",
            "sell_price",
            "snap",
            "event_name",
            "event_type",
        }
        assert expected_cols.issubset(set(daily.columns))
        assert not daily.empty
        assert not cal.empty
        assert not prices.empty

    def test_synthetic_reproducibility(self) -> None:
        """Same seed must produce identical sales and price arrays."""
        d1, _, p1 = generate_synthetic_m5(seed=123, n_days=50, items_per_dept=1)
        d2, _, p2 = generate_synthetic_m5(seed=123, n_days=50, items_per_dept=1)

        pd.testing.assert_frame_equal(d1, d2)
        pd.testing.assert_frame_equal(p1, p2)

    def test_synthetic_different_seeds_differ(self) -> None:
        """Different seeds must yield different sales series."""
        d1, _, _ = generate_synthetic_m5(seed=1, n_days=50, items_per_dept=1)
        d2, _, _ = generate_synthetic_m5(seed=2, n_days=50, items_per_dept=1)

        assert not np.array_equal(d1["sales"].to_numpy(), d2["sales"].to_numpy())

    def test_synthetic_value_constraints(self) -> None:
        """Sales must be non-negative integers; prices must be strictly positive."""
        daily, cal, prices = generate_synthetic_m5(
            seed=42, n_days=60, items_per_dept=2
        )

        assert (daily["sales"] >= 0).all()
        assert (daily["sell_price"] > 0).all()
        assert set(daily["snap"].unique()).issubset({0, 1})
        assert pd.api.types.is_datetime64_any_dtype(daily["date"])

    def test_synthetic_custom_dimensions(self) -> None:
        """n_days and items_per_dept must govern output row count."""
        n_days = 40
        items_per_dept = 2  # 7 depts * 2 items * 10 stores = 140 series
        daily, cal, prices = generate_synthetic_m5(
            seed=42, n_days=n_days, items_per_dept=items_per_dept
        )

        n_series = 7 * items_per_dept * 10
        assert daily["series_id"].nunique() == n_series
        assert len(daily) == n_series * n_days
        assert len(cal) == n_days


class TestLoadM5:
    """Tests for top-level load_m5 wrapper."""

    def test_load_m5_returns_tuple_of_three(self) -> None:
        """load_m5 must return (daily, calendar, prices)."""
        daily, cal, prices = load_m5(synthetic_n_days=30, synthetic_items_per_dept=1)
        assert isinstance(daily, pd.DataFrame)
        assert isinstance(cal, pd.DataFrame)
        assert isinstance(prices, pd.DataFrame)
        assert len(daily) > 0

    def test_m5_files_present_returns_bool(self) -> None:
        """m5_files_present returns a boolean indicating if raw files exist."""
        res = m5_files_present()
        assert isinstance(res, bool)
