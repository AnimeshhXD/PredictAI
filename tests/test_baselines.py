"""
tests/test_baselines.py
=======================
Tests for B1 (ARIMA + multiplier), B2 (ARIMAX / Elasticity), and
B3 (XGBoost with lag, calendar, scenario features).

Verifies:
1. 28-day forecast generation and non-negativity constraints.
2. Scenario logic (price drop response, demand surge lift, stockout zeroing).
3. Data leakage guard integration — ANY baseline called with data on or after
   cutoff_date MUST RAISE DataLeakageError.
4. Metric calculations (MAE, RMSE, WAPE, SMAPE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    HORIZON_DAYS,
    calculate_metrics,
    forecast_b1_arima_multiplier,
    forecast_b2_arimax,
    forecast_b3_xgboost,
    get_pre_cutoff_series,
    run_baseline,
)
from src.leakage_guard import DataLeakageError


def _make_dummy_pre_series(
    n_days: int = 150,
    base_sales: float = 6.0,
    base_price: float = 8.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a clean historical daily series preceding 2022-01-01."""
    dates = pd.date_range("2021-08-01", periods=n_days, freq="D")
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


CUTOFF = "2022-01-01"


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculateMetrics:
    """Tests for metric calculations."""

    def test_metrics_values(self) -> None:
        actual = np.array([10.0, 20.0, 30.0])
        pred = np.array([12.0, 18.0, 30.0])
        metrics = calculate_metrics(actual, pred)

        assert metrics["mae"] == round((2.0 + 2.0 + 0.0) / 3.0, 4)
        assert metrics["rmse"] == round(np.sqrt((4.0 + 4.0 + 0.0) / 3.0), 4)
        assert metrics["wape"] == round(4.0 / 60.0 * 100.0, 4)

    def test_metric_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_metrics(np.array([1.0, 2.0]), np.array([1.0]))


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 1: ARIMA + Multiplier Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestB1ArimaMultiplier:
    """Tests for B1 baseline."""

    def test_b1_price_up_reduces_forecast(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        cutoff = df["date"].max() + pd.Timedelta(days=1)

        f_base = forecast_b1_arima_multiplier(df, cutoff, "none", {}, horizon_days=28)
        f_price_up = forecast_b1_arima_multiplier(
            df, cutoff, "price_up", {"pct_change": 10.0}, horizon_days=28
        )

        assert len(f_price_up) == HORIZON_DAYS
        assert (f_price_up >= 0.0).all()
        # Higher price should lower demand compared to unadjusted baseline
        assert np.mean(f_price_up) < np.mean(f_base)

    def test_b1_demand_surge_increases_forecast(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        cutoff = df["date"].max() + pd.Timedelta(days=1)

        f_base = forecast_b1_arima_multiplier(df, cutoff, "none", {}, horizon_days=28)
        f_surge = forecast_b1_arima_multiplier(
            df, cutoff, "demand_surge", {"pct_change": 20.0}, horizon_days=28
        )

        assert len(f_surge) == HORIZON_DAYS
        assert np.mean(f_surge) > np.mean(f_base)

    def test_b1_supply_delay_zeroes_initial_days(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        cutoff = df["date"].max() + pd.Timedelta(days=1)
        stockout_days = 4

        f_delay = forecast_b1_arima_multiplier(
            df, cutoff, "supply_delay", {"stockout_days": stockout_days}, horizon_days=28
        )

        assert len(f_delay) == HORIZON_DAYS
        assert np.all(f_delay[:stockout_days] == 0.0)
        assert np.all(f_delay[stockout_days:] >= 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 2: ARIMAX / Elasticity Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestB2Arimax:
    """Tests for B2 baseline."""

    def test_b2_output_length_and_non_negativity(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        cutoff = df["date"].max() + pd.Timedelta(days=1)

        f_price_up = forecast_b2_arimax(
            df, cutoff, "price_up", {"pct_change": 10.0, "baseline_price": 8.0, "new_price": 8.8}, horizon_days=28
        )
        assert len(f_price_up) == HORIZON_DAYS
        assert (f_price_up >= 0.0).all()

    def test_b2_supply_delay_stockout_zeroes(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        cutoff = df["date"].max() + pd.Timedelta(days=1)
        stockout_days = 5

        f_delay = forecast_b2_arimax(
            df, cutoff, "supply_delay", {"stockout_days": stockout_days}, horizon_days=28
        )
        assert len(f_delay) == HORIZON_DAYS
        assert np.all(f_delay[:stockout_days] == 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 3: XGBoost Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestB3XGBoost:
    """Tests for B3 baseline."""

    def test_b3_forecast_length_and_non_negativity(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        cutoff = df["date"].max() + pd.Timedelta(days=1)

        f_xgb = forecast_b3_xgboost(
            df, cutoff, "price_up", {"pct_change": 10.0, "new_price": 8.8}, horizon_days=28, seed=42
        )
        assert len(f_xgb) == HORIZON_DAYS
        assert (f_xgb >= 0.0).all()

    def test_b3_deterministic_seeded(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        cutoff = df["date"].max() + pd.Timedelta(days=1)

        f1 = forecast_b3_xgboost(df, cutoff, "demand_surge", {"pct_change": 15.0}, seed=42)
        f2 = forecast_b3_xgboost(df, cutoff, "demand_surge", {"pct_change": 15.0}, seed=42)
        np.testing.assert_allclose(f1, f2)

    def test_b3_supply_delay_zeroes_stockout(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        cutoff = df["date"].max() + pd.Timedelta(days=1)
        stockout_days = 3

        f_delay = forecast_b3_xgboost(
            df, cutoff, "supply_delay", {"stockout_days": stockout_days}, horizon_days=28, seed=42
        )
        assert np.all(f_delay[:stockout_days] == 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Strict Data Leakage Prevention Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaselineLeakageGuards:
    """
    CRITICAL: Any baseline invoked with a DataFrame that touches or exceeds
    the scenario cutoff MUST fail by raising DataLeakageError.
    """

    def test_b1_raises_on_future_data(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        # Put cutoff inside the history
        leaky_cutoff = df["date"].iloc[100]

        with pytest.raises(DataLeakageError):
            forecast_b1_arima_multiplier(df, leaky_cutoff, "price_up", {})

    def test_b2_raises_on_future_data(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        leaky_cutoff = df["date"].iloc[100]

        with pytest.raises(DataLeakageError):
            forecast_b2_arimax(df, leaky_cutoff, "price_up", {})

    def test_b3_raises_on_future_data(self) -> None:
        df = _make_dummy_pre_series(n_days=150)
        leaky_cutoff = df["date"].iloc[100]

        with pytest.raises(DataLeakageError):
            forecast_b3_xgboost(df, leaky_cutoff, "price_up", {})


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Dispatcher Test
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunBaselineDispatcher:
    """Tests for run_baseline dispatcher."""

    def test_dispatcher_valid_methods(self) -> None:
        df = _make_dummy_pre_series(n_days=100)
        cutoff = df["date"].max() + pd.Timedelta(days=1)

        for m in ("b1_arima", "b2_arimax", "b3_xgboost"):
            f = run_baseline(m, df, cutoff, "demand_surge", {"pct_change": 15.0})
            assert len(f) == HORIZON_DAYS

    def test_dispatcher_unknown_method_raises(self) -> None:
        df = _make_dummy_pre_series(n_days=100)
        cutoff = df["date"].max() + pd.Timedelta(days=1)

        with pytest.raises(ValueError, match="Unknown baseline method"):
            run_baseline("unknown_model", df, cutoff, "price_up", {})
