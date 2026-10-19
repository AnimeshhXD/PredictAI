"""
tests/test_episode_detector.py
==============================
Tests for src/episode_detector.py.

Verifies detection rules for:
1. price_up (8-12% sustained price increase)
2. demand_surge (15-25% demand lift without price cuts)
3. supply_delay (3-7 consecutive zero-sales days after normal sales)
4. Data leakage prevention and horizon integrity (28-day window)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_m5
from src.episode_detector import (
    HORIZON_DAYS,
    PRICE_UP_MAX_PCT,
    PRICE_UP_MIN_PCT,
    STOCKOUT_MAX_DAYS,
    STOCKOUT_MIN_DAYS,
    SURGE_MAX_PCT,
    SURGE_MIN_PCT,
    _detect_demand_surge,
    _detect_price_up,
    _detect_supply_delay,
    detect_all_episodes,
    summary_report,
)
from src.leakage_guard import assert_no_leakage


def _make_dummy_series(
    n_days: int = 250,
    base_sales: float = 5.0,
    base_price: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Construct a clean single-series DataFrame with default dates and values."""
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    sales = rng.poisson(lam=base_sales, size=n_days).astype(float)
    prices = np.full(n_days, base_price, dtype=float)

    return pd.DataFrame({
        "date": dates,
        "item_id": "FOODS_1_001",
        "store_id": "CA_1",
        "series_id": "FOODS_1_001_CA_1",
        "sales": sales,
        "sell_price": prices,
        "snap": 0,
        "event_name": None,
        "event_type": None,
    })


class TestPriceUpDetector:
    """Tests for price_up episode detection."""

    def test_sustained_price_increase_detected(self) -> None:
        """A 10% price increase sustained for 14 days should trigger detection."""
        df = _make_dummy_series(n_days=250, base_price=10.0)
        # Apply 10% price bump starting at day 120 (well past MIN_LOOKBACK_DAYS=90)
        df.loc[120:, "sell_price"] = 11.0  # +10%

        rng = np.random.default_rng(42)
        ep = _detect_price_up(df, rng)

        assert ep is not None
        assert ep.scenario_type == "price_up"
        assert ep.meta["pct_change"] >= PRICE_UP_MIN_PCT * 100
        assert ep.meta["pct_change"] <= PRICE_UP_MAX_PCT * 100
        assert len(ep.actuals) == HORIZON_DAYS

    def test_price_increase_too_small_ignored(self) -> None:
        """A 4% price increase is below the 8% threshold and must be ignored."""
        df = _make_dummy_series(n_days=250, base_price=10.0)
        df.loc[120:, "sell_price"] = 10.4  # +4%

        rng = np.random.default_rng(42)
        ep = _detect_price_up(df, rng)
        assert ep is None

    def test_price_increase_too_large_ignored(self) -> None:
        """A 25% price increase is above the 12% threshold and must be ignored."""
        df = _make_dummy_series(n_days=250, base_price=10.0)
        df.loc[120:, "sell_price"] = 12.5  # +25%

        rng = np.random.default_rng(42)
        ep = _detect_price_up(df, rng)
        assert ep is None

    def test_price_increase_not_sustained_ignored(self) -> None:
        """A price spike lasting only 2 days (< 7 days) must not be detected."""
        df = _make_dummy_series(n_days=250, base_price=10.0)
        df.loc[120:121, "sell_price"] = 11.0  # only 2 days
        df.loc[122:, "sell_price"] = 10.0

        rng = np.random.default_rng(42)
        ep = _detect_price_up(df, rng)
        assert ep is None


class TestDemandSurgeDetector:
    """Tests for demand_surge episode detection."""

    def test_demand_surge_detected(self) -> None:
        """A 20% demand surge over 4 weeks without price cut must be detected."""
        df = _make_dummy_series(n_days=300, base_sales=10.0, seed=1)
        # Prior 8 weeks (days 100..155) mean demand ~10
        # Surge 4 weeks (days 156..183) mean demand ~12 (+20%)
        df.loc[156:183, "sales"] = 12.0
        df.loc[100:155, "sales"] = 10.0

        rng = np.random.default_rng(42)
        ep = _detect_demand_surge(df, rng)

        assert ep is not None
        assert ep.scenario_type == "demand_surge"
        assert ep.meta["pct_change"] >= SURGE_MIN_PCT * 100
        assert ep.meta["pct_change"] <= SURGE_MAX_PCT * 100
        assert len(ep.actuals) == HORIZON_DAYS

    def test_demand_surge_with_price_cut_ignored(self) -> None:
        """A demand surge accompanied by a price cut is confounded and must be ignored."""
        df = _make_dummy_series(n_days=300, base_sales=10.0, seed=1)
        df.loc[156:183, "sales"] = 12.0
        df.loc[100:155, "sales"] = 10.0
        # Discount price during the surge by 15%
        df.loc[156:183, "sell_price"] = 8.5

        rng = np.random.default_rng(42)
        ep = _detect_demand_surge(df, rng)
        assert ep is None


class TestSupplyDelayDetector:
    """Tests for supply_delay episode detection."""

    def test_stockout_run_detected(self) -> None:
        """A run of 5 consecutive zero-sales days after normal sales is detected."""
        df = _make_dummy_series(n_days=250, base_sales=5.0)
        # Insert 5 zero days at day 120
        df.loc[120:124, "sales"] = 0.0

        rng = np.random.default_rng(42)
        ep = _detect_supply_delay(df, rng)

        assert ep is not None
        assert ep.scenario_type == "supply_delay"
        assert ep.meta["stockout_days"] == 5
        assert len(ep.actuals) == HORIZON_DAYS

    def test_stockout_too_short_ignored(self) -> None:
        """A run of 1 zero day (< 3 days) is not a supply delay episode."""
        df = _make_dummy_series(n_days=250, base_sales=5.0)
        df.loc[120:120, "sales"] = 0.0

        rng = np.random.default_rng(42)
        ep = _detect_supply_delay(df, rng)
        assert ep is None

    def test_stockout_too_long_ignored(self) -> None:
        """A run of 12 zero days (> 7 days) indicates delisting, not temporary stockout."""
        df = _make_dummy_series(n_days=250, base_sales=5.0)
        df.loc[120:131, "sales"] = 0.0

        rng = np.random.default_rng(42)
        ep = _detect_supply_delay(df, rng)
        assert ep is None


class TestLeakageAndIntegrity:
    """Tests guaranteeing zero data leakage and proper scenario horizon."""

    def test_all_episode_actuals_and_training_leakage_free(self) -> None:
        """
        Verify that for every detected episode:
        1. Training slice strictly precedes cutoff_date.
        2. Actuals start on cutoff_date.
        3. Actuals length is exactly 28 days.
        """
        df = _make_dummy_series(n_days=250, base_price=10.0)
        df.loc[120:, "sell_price"] = 11.0

        rng = np.random.default_rng(42)
        ep = _detect_price_up(df, rng)
        assert ep is not None

        # Verify training data slice has NO leakage
        train_df = df[df["date"] < ep.cutoff_date]
        assert_no_leakage(train_df, date_col="date", cutoff=ep.cutoff_date)

        # Verify actuals start exactly on cutoff_date
        assert pd.Timestamp(ep.actual_dates[0]) == ep.cutoff_date
        assert len(ep.actuals) == 28


class TestDetectAllEpisodes:
    """End-to-end multi-series episode detection test."""

    def test_detect_all_episodes_synthetic(self) -> None:
        """detect_all_episodes on synthetic data returns target episodes per type."""
        daily, _, _ = generate_synthetic_m5(seed=42, n_days=800, items_per_dept=6)
        min_target = 5
        episodes_df = detect_all_episodes(
            daily, min_per_type=min_target, seed=42, max_series_scan=150
        )

        assert not episodes_df.empty
        counts = episodes_df.groupby("scenario_type").size().to_dict()
        for stype in ("price_up", "demand_surge", "supply_delay"):
            assert counts.get(stype, 0) >= min_target

        # Check required schema columns
        required_cols = {
            "series_id",
            "scenario_type",
            "cutoff_date",
            "scenario_text",
            "actuals",
            "actual_dates",
        }
        assert required_cols.issubset(set(episodes_df.columns))

        # Check summary report
        report = summary_report(episodes_df)
        assert "PredictAI - Episode Detection Summary" in report
        assert "price_up" in report
