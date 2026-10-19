"""
src/baselines.py
================
Implementation of statistical and machine learning baselines for what-if
time-series forecasting.

All models strictly use data dated prior to the scenario cutoff date.
Any attempt to pass future data will raise a DataLeakageError via the
leakage guard.

Baselines
---------
B1 : ARIMA + Literal Scenario Multiplier
    Fits an ARIMA model on pre-cutoff sales and applies the what-if
    intervention as a direct multiplier:
    * price_up: (1 / (1 + pct_change))
    * demand_surge: (1 + pct_change)
    * supply_delay: 0.0 for stockout days, 1.0 afterwards

B2 : ARIMAX / Log-Log Price Elasticity Regression
    Econometric baseline with price as an exogenous covariate:
    * ARIMAX(p,d,q) with sell_price as exog
    * Fallback to log-log elasticity regression:
      ln(sales + 1) = alpha + beta * ln(sell_price) + gamma * dayofweek
      Scales ARIMA baseline by (P_new / P_base)^beta

B3 : XGBoost with Lag, Calendar, and Scenario Features
    Recursive 28-day forecasting using:
    * Lags: 1, 2, 7, 14, 21, 28
    * Rolling stats: 7, 14, 28-day mean, 7-day std
    * Calendar: dayofweek, day, month, is_weekend, snap
    * Scenario: price_ratio, is_stockout, surge_ratio
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

from src.leakage_guard import assert_no_leakage, filter_before_cutoff

logger = logging.getLogger(__name__)

HORIZON_DAYS = 28


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics on the held-out evaluation window.

    Parameters
    ----------
    actual:
        Ground truth values (length 28).
    forecast:
        Forecast values (length 28).

    Returns
    -------
    dict with keys: 'mae', 'rmse', 'wape', 'smape'
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError(f"Length mismatch: {len(actual)} actuals vs {len(forecast)} forecast.")

    errors = forecast - actual
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # WAPE (Weighted Absolute Percentage Error) standard in retail demand forecasting
    sum_actual = float(np.sum(actual))
    wape = float(np.sum(abs_errors) / max(sum_actual, 1e-6) * 100.0)

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denom = np.maximum(np.abs(actual) + np.abs(forecast), 1e-6)
    smape = float(np.mean(2.0 * abs_errors / denom) * 100.0)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "smape": round(smape, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Pre-Cutoff Data Extraction with Leakage Verification
# ═══════════════════════════════════════════════════════════════════════════════


def get_pre_cutoff_series(
    series_df: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """
    Extract history strictly prior to the cutoff date and verify zero data leakage.

    Parameters
    ----------
    series_df:
        Full or historical time series DataFrame with 'date' and 'sales'.
    cutoff_date:
        Exclusive upper bound date for the training period.

    Returns
    -------
    pd.DataFrame:
        Filtered, sorted DataFrame strictly before cutoff.
    """
    cutoff = pd.Timestamp(cutoff_date)
    pre = filter_before_cutoff(series_df, date_col="date", cutoff=cutoff)
    assert_no_leakage(pre, date_col="date", cutoff=cutoff)
    return pre.sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 1: ARIMA + Literal Scenario Multiplier
# ═══════════════════════════════════════════════════════════════════════════════


def _fit_arima_forecast(
    sales_series: pd.Series | np.ndarray,
    horizon_days: int = HORIZON_DAYS,
) -> np.ndarray:
    """
    Fit an ARIMA(1, 0, 1) or (1, 1, 0) model and produce baseline forecasts.
    Falls back gracefully to historical trailing mean if ARIMA fails.
    """
    y = np.asarray(sales_series, dtype=float)
    if len(y) == 0:
        return np.zeros(horizon_days, dtype=float)

    # Check for near-constant series
    if np.all(y == y[0]) or np.std(y) < 1e-5:
        return np.full(horizon_days, fill_value=max(0.0, y[-1]), dtype=float)

    # Try standard ARIMA specifications
    for order in [(1, 0, 1), (1, 1, 0), (1, 0, 0)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(y, order=order)
                res = model.fit()
                pred = res.forecast(steps=horizon_days)
                return np.maximum(pred, 0.0)
        except Exception:
            continue

    # Fallback: 28-day trailing moving average
    window = min(28, len(y))
    trail_mean = float(np.mean(y[-window:]))
    return np.full(horizon_days, fill_value=max(0.0, trail_mean), dtype=float)


def forecast_b1_arima_multiplier(
    series_pre_cutoff: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
    scenario_type: str,
    meta: Dict[str, Any],
    horizon_days: int = HORIZON_DAYS,
) -> np.ndarray:
    """
    B1 Baseline: ARIMA baseline forecast multiplied by literal scenario impact.

    Parameters
    ----------
    series_pre_cutoff:
        Historical DataFrame strictly before cutoff.
    cutoff_date:
        Cutoff date for leakage verification.
    scenario_type:
        One of 'price_up', 'demand_surge', 'supply_delay'.
    meta:
        Episode detector metadata (e.g. pct_change, stockout_days).
    horizon_days:
        Forecast horizon (default 28).

    Returns
    -------
    np.ndarray of length 28 (forecasted units).
    """
    # 1. Leakage guard assertion
    assert_no_leakage(series_pre_cutoff, date_col="date", cutoff=cutoff_date)

    # 2. Fit unconstrained ARIMA baseline
    baseline = _fit_arima_forecast(series_pre_cutoff["sales"], horizon_days=horizon_days)

    # 3. Apply literal multiplier
    multiplier = np.ones(horizon_days, dtype=float)

    if scenario_type == "price_up":
        # Econometric elasticity approximation: delta_q / q = -1 * delta_p / p
        pct = float(meta.get("pct_change", 10.0)) / 100.0
        scale = max(0.0, 1.0 / (1.0 + pct))
        multiplier[:] = scale

    elif scenario_type == "demand_surge":
        # Direct positive demand lift
        pct = float(meta.get("pct_change", 20.0)) / 100.0
        scale = 1.0 + pct
        multiplier[:] = scale

    elif scenario_type == "supply_delay":
        # Stockout days: 0 units; recovery: normal sales
        stockout_days = int(meta.get("stockout_days", 3))
        stockout_days = min(stockout_days, horizon_days)
        multiplier[:stockout_days] = 0.0
        multiplier[stockout_days:] = 1.0

    forecast = baseline * multiplier
    return np.maximum(forecast, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 2: ARIMAX / Log-Log Elasticity Regression
# ═══════════════════════════════════════════════════════════════════════════════


def forecast_b2_arimax(
    series_pre_cutoff: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
    scenario_type: str,
    meta: Dict[str, Any],
    horizon_days: int = HORIZON_DAYS,
) -> np.ndarray:
    """
    B2 Baseline: ARIMAX / Econometric Log-Log Elasticity Regression.

    Estimates price elasticity from historical pre-cutoff price-sales variations
    and projects the scenario outcome with exogenous price covariates.

    Parameters
    ----------
    series_pre_cutoff:
        Historical DataFrame strictly before cutoff.
    cutoff_date:
        Cutoff date for leakage verification.
    scenario_type:
        One of 'price_up', 'demand_surge', 'supply_delay'.
    meta:
        Episode detector metadata.
    horizon_days:
        Forecast horizon (default 28).

    Returns
    -------
    np.ndarray of length 28.
    """
    assert_no_leakage(series_pre_cutoff, date_col="date", cutoff=cutoff_date)

    sales = series_pre_cutoff["sales"].to_numpy(dtype=float)
    prices = series_pre_cutoff["sell_price"].to_numpy(dtype=float)

    # Construct future price trajectory for the 28 days
    last_price = float(prices[-1]) if len(prices) > 0 else 1.0

    if scenario_type == "price_up":
        new_price = float(meta.get("new_price", last_price * (1.0 + meta.get("pct_change", 10.0) / 100.0)))
        future_prices = np.full(horizon_days, fill_value=new_price, dtype=float)
    else:
        future_prices = np.full(horizon_days, fill_value=last_price, dtype=float)

    # Try fitting ARIMAX with sell_price as exogenous variable
    fitted_arimax = False
    forecast = np.zeros(horizon_days, dtype=float)

    if np.std(prices) > 1e-4 and len(sales) >= 30:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(endog=sales, exog=prices, order=(1, 0, 1))
                res = model.fit()
                pred = res.forecast(steps=horizon_days, exog=future_prices)
                forecast = np.maximum(pred, 0.0)
                fitted_arimax = True
        except Exception:
            fitted_arimax = False

    if not fitted_arimax:
        # Fallback to Log-Log Elasticity Regression:
        # ln(Sales + 1) = alpha + beta * ln(Price) + gamma * DayOfWeek
        baseline = _fit_arima_forecast(sales, horizon_days=horizon_days)

        if scenario_type == "price_up":
            # Estimate beta from pre-cutoff data
            valid_mask = (sales >= 0) & (prices > 0)
            if np.sum(valid_mask) > 14 and np.std(prices[valid_mask]) > 1e-3:
                log_p = np.log(prices[valid_mask]).reshape(-1, 1)
                log_q = np.log(sales[valid_mask] + 1.0)
                reg = LinearRegression().fit(log_p, log_q)
                beta = float(reg.coef_[0])
                # In retail economics, price elasticity is negative; clamp plausible bounds [-3.0, -0.2]
                if beta >= 0:
                    beta = -1.0
                else:
                    beta = max(-3.0, min(-0.2, beta))
            else:
                beta = -1.0  # standard unit elasticity default

            baseline_p = float(meta.get("baseline_price", last_price))
            new_p = float(meta.get("new_price", future_prices[0]))
            price_ratio = new_p / max(baseline_p, 1e-4)
            elasticity_multiplier = float(price_ratio ** beta)
            forecast = baseline * elasticity_multiplier

        elif scenario_type == "demand_surge":
            # Exogenous demand shift
            pct = float(meta.get("pct_change", 20.0)) / 100.0
            forecast = baseline * (1.0 + pct)

        elif scenario_type == "supply_delay":
            stockout_days = min(int(meta.get("stockout_days", 3)), horizon_days)
            forecast = baseline.copy()
            forecast[:stockout_days] = 0.0
        else:
            forecast = baseline

    # If supply_delay, ensure stockout days are strictly 0.0
    if scenario_type == "supply_delay":
        stockout_days = min(int(meta.get("stockout_days", 3)), horizon_days)
        forecast[:stockout_days] = 0.0

    return np.maximum(forecast, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 3: XGBoost with Lag, Calendar, and Scenario Features
# ═══════════════════════════════════════════════════════════════════════════════


def _build_xgb_feature_row(
    sales_history: list[float],
    current_date: pd.Timestamp,
    sell_price: float,
    baseline_price: float,
    scenario_type: str,
    scenario_day_idx: int,
    meta: Dict[str, Any],
    snap: int = 0,
) -> np.ndarray:
    """
    Construct the feature vector for a single forecast step.
    Uses strictly available historical sales (or previous autoregressive predictions).
    """
    # Lag features
    n = len(sales_history)
    lag_1 = sales_history[-1] if n >= 1 else 0.0
    lag_2 = sales_history[-2] if n >= 2 else lag_1
    lag_7 = sales_history[-7] if n >= 7 else lag_1
    lag_14 = sales_history[-14] if n >= 14 else lag_7
    lag_21 = sales_history[-21] if n >= 21 else lag_14
    lag_28 = sales_history[-28] if n >= 28 else lag_21

    # Rolling window statistics
    roll_7 = float(np.mean(sales_history[-min(n, 7):])) if n > 0 else 0.0
    roll_14 = float(np.mean(sales_history[-min(n, 14):])) if n > 0 else roll_7
    roll_28 = float(np.mean(sales_history[-min(n, 28):])) if n > 0 else roll_14
    roll_std_7 = float(np.std(sales_history[-min(n, 7):])) if n > 1 else 0.0

    # Calendar features
    dayofweek = float(current_date.dayofweek)
    day = float(current_date.day)
    month = float(current_date.month)
    is_weekend = 1.0 if current_date.dayofweek in (5, 6) else 0.0

    # Scenario features
    price_ratio = sell_price / max(baseline_price, 1e-4)

    is_price_up = 1.0 if scenario_type == "price_up" else 0.0
    is_demand_surge = 1.0 if scenario_type == "demand_surge" else 0.0
    is_supply_delay = 1.0 if scenario_type == "supply_delay" else 0.0

    # Stockout active indicator
    stockout_k = int(meta.get("stockout_days", 0)) if scenario_type == "supply_delay" else 0
    is_stockout_active = 1.0 if (scenario_type == "supply_delay" and scenario_day_idx < stockout_k) else 0.0

    surge_ratio = 1.0 + float(meta.get("pct_change", 0.0)) / 100.0 if scenario_type == "demand_surge" else 1.0

    return np.array([
        lag_1, lag_2, lag_7, lag_14, lag_21, lag_28,
        roll_7, roll_14, roll_28, roll_std_7,
        dayofweek, day, month, is_weekend, float(snap),
        price_ratio, is_price_up, is_demand_surge, is_supply_delay,
        is_stockout_active, surge_ratio,
    ], dtype=np.float32)


def forecast_b3_xgboost(
    series_pre_cutoff: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
    scenario_type: str,
    meta: Dict[str, Any],
    horizon_days: int = HORIZON_DAYS,
    seed: int = 42,
) -> np.ndarray:
    """
    B3 Baseline: XGBoost Regressor with autoregressive recursive forecasting.

    Parameters
    ----------
    series_pre_cutoff:
        Historical DataFrame strictly before cutoff.
    cutoff_date:
        Cutoff date for leakage verification.
    scenario_type:
        One of 'price_up', 'demand_surge', 'supply_delay'.
    meta:
        Episode detector metadata.
    horizon_days:
        Forecast horizon (default 28).
    seed:
        Random state for reproducibility.

    Returns
    -------
    np.ndarray of length 28.
    """
    import xgboost as xgb

    assert_no_leakage(series_pre_cutoff, date_col="date", cutoff=cutoff_date)

    df = series_pre_cutoff.sort_values("date").reset_index(drop=True)
    sales_arr = df["sales"].to_numpy(dtype=float)
    prices_arr = df["sell_price"].to_numpy(dtype=float) if "sell_price" in df.columns else np.ones(len(df))
    dates_arr = pd.to_datetime(df["date"]).values
    snap_arr = df["snap"].to_numpy(dtype=int) if "snap" in df.columns else np.zeros(len(df), dtype=int)

    base_price = float(np.mean(prices_arr[-28:])) if len(prices_arr) >= 28 else float(prices_arr[-1])

    # Minimum history needed for 28-day lags
    min_train_start = 28
    if len(df) <= min_train_start:
        # Fallback to ARIMA if series is too short for lag-28
        return _fit_arima_forecast(sales_arr, horizon_days=horizon_days)

    # 1. Build training matrix X, y using pre-cutoff sliding windows
    X_train_list = []
    y_train_list = []

    # Historical scenario features are baseline (none active)
    for i in range(min_train_start, len(df)):
        hist = list(sales_arr[:i])
        curr_d = pd.Timestamp(dates_arr[i])
        curr_p = float(prices_arr[i])
        curr_snap = int(snap_arr[i])

        feat = _build_xgb_feature_row(
            sales_history=hist,
            current_date=curr_d,
            sell_price=curr_p,
            baseline_price=base_price,
            scenario_type="none",
            scenario_day_idx=0,
            meta={},
            snap=curr_snap,
        )
        X_train_list.append(feat)
        y_train_list.append(sales_arr[i])

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)

    # 2. Fit XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # 3. Recursive 28-day forecasting
    cutoff_ts = pd.Timestamp(cutoff_date)
    rolling_sales = list(sales_arr)
    forecasts = []

    # Determine scenario price
    if scenario_type == "price_up":
        future_price = float(meta.get("new_price", base_price * (1.0 + meta.get("pct_change", 10.0) / 100.0)))
    else:
        future_price = float(prices_arr[-1])

    for h in range(horizon_days):
        step_date = cutoff_ts + pd.Timedelta(days=h)
        step_feat = _build_xgb_feature_row(
            sales_history=rolling_sales,
            current_date=step_date,
            sell_price=future_price,
            baseline_price=base_price,
            scenario_type=scenario_type,
            scenario_day_idx=h,
            meta=meta,
            snap=0,
        ).reshape(1, -1)

        pred_val = float(model.predict(step_feat)[0])
        pred_val = max(0.0, pred_val)

        # Enforce exact stockout zeroing if in supply_delay window
        if scenario_type == "supply_delay":
            stockout_days = int(meta.get("stockout_days", 3))
            if h < stockout_days:
                pred_val = 0.0

        forecasts.append(pred_val)
        rolling_sales.append(pred_val)

    return np.array(forecasts, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════


def run_baseline(
    method: str,
    series_pre_cutoff: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
    scenario_type: str,
    meta: Dict[str, Any],
    horizon_days: int = HORIZON_DAYS,
    seed: int = 42,
) -> np.ndarray:
    """
    Run the requested baseline forecast method.

    Parameters
    ----------
    method:
        'b1_arima', 'b2_arimax', or 'b3_xgboost'.
    series_pre_cutoff:
        Pre-cutoff historical DataFrame.
    cutoff_date:
        Cutoff date.
    scenario_type:
        Episode scenario type.
    meta:
        Episode metadata dictionary.
    horizon_days:
        Forecast horizon (28 days).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of length 28.
    """
    method = method.lower().strip()
    if method in ("b1", "b1_arima", "arima_multiplier", "arima"):
        return forecast_b1_arima_multiplier(
            series_pre_cutoff=series_pre_cutoff,
            cutoff_date=cutoff_date,
            scenario_type=scenario_type,
            meta=meta,
            horizon_days=horizon_days,
        )
    elif method in ("b2", "b2_arimax", "arimax", "elasticity"):
        return forecast_b2_arimax(
            series_pre_cutoff=series_pre_cutoff,
            cutoff_date=cutoff_date,
            scenario_type=scenario_type,
            meta=meta,
            horizon_days=horizon_days,
        )
    elif method in ("b3", "b3_xgboost", "xgboost"):
        return forecast_b3_xgboost(
            series_pre_cutoff=series_pre_cutoff,
            cutoff_date=cutoff_date,
            scenario_type=scenario_type,
            meta=meta,
            horizon_days=horizon_days,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown baseline method '{method}'. "
            f"Expected one of ['b1_arima', 'b2_arimax', 'b3_xgboost']."
        )
