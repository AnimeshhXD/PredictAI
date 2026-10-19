"""
src/data_loader.py
==================
Load and pre-process M5 Forecasting Competition data into a clean
per-item-store daily sales DataFrame.

Supports two modes:
* **Real** – reads CSVs from ``data/raw/m5/`` (downloaded from Kaggle).
* **Synthetic** – generates a seeded synthetic dataset that faithfully
  mimics M5 schema, scale, and statistical properties for development/CI.
  Activated automatically when the raw files are absent.

Public API
----------
>>> from src.data_loader import load_m5
>>> daily, calendar, prices = load_m5()
>>> daily.columns
Index(['date', 'item_id', 'store_id', 'series_id', 'sales', 'sell_price',
       'snap', 'event_name', 'event_type'], dtype='object')
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
_RAW = _ROOT / "data" / "raw" / "m5"

_SALES_EVAL = _RAW / "sales_train_evaluation.csv"
_CALENDAR   = _RAW / "calendar.csv"
_PRICES     = _RAW / "sell_prices.csv"

# ── Types ──────────────────────────────────────────────────────────────────────
M5Tuple = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

# ── Constants matching M5 dataset ─────────────────────────────────────────────
_STORES = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3",
]
_CATS = {
    "FOODS":     ["FOODS_1",     "FOODS_2",     "FOODS_3"],
    "HOBBIES":   ["HOBBIES_1",   "HOBBIES_2"],
    "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
}
_START_DATE = pd.Timestamp("2011-01-29")
_N_REAL_DAYS = 1941   # evaluation set length


# ═══════════════════════════════════════════════════════════════════════════════
# Real-data loader
# ═══════════════════════════════════════════════════════════════════════════════


def _load_real() -> M5Tuple:
    """
    Load the three M5 CSVs, melt sales to long format, merge calendar and
    prices, and return ``(daily, calendar, prices)``.

    Returns
    -------
    daily : pd.DataFrame
        Columns: date, item_id, store_id, series_id, sales,
                 sell_price, snap, event_name, event_type.
    calendar : pd.DataFrame
        Raw calendar table (parsed dates).
    prices : pd.DataFrame
        Raw sell_prices table.

    Raises
    ------
    FileNotFoundError
        If any of the three expected CSVs are missing.
    """
    for p in (_SALES_EVAL, _CALENDAR, _PRICES):
        if not p.exists():
            raise FileNotFoundError(f"Missing M5 file: {p}")

    logger.info("Loading M5 sales…")
    sales = pd.read_csv(_SALES_EVAL, dtype={"id": str})

    logger.info("Loading M5 calendar…")
    cal = pd.read_csv(_CALENDAR, parse_dates=["date"])
    cal["date"] = pd.to_datetime(cal["date"])

    logger.info("Loading M5 sell_prices…")
    prices = pd.read_csv(_PRICES)

    # ── Melt sales wide→long ──────────────────────────────────────────────────
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    logger.info("Melting %d day columns…", len(day_cols))
    long = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    long["sales"] = long["sales"].astype(np.float32)

    # ── Merge calendar ────────────────────────────────────────────────────────
    cal_slim = cal[["d", "date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI",
                     "event_name_1", "event_type_1"]].copy()
    long = long.merge(cal_slim, on="d", how="left")

    # Collapse per-state SNAP into a single column matching the row's state
    def _snap(row: pd.Series) -> int:
        return int(row[f"snap_{row['state_id']}"])

    long["snap"] = long.apply(_snap, axis=1).astype(np.int8)

    # ── Merge prices ──────────────────────────────────────────────────────────
    long = long.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    # ── Final clean-up ────────────────────────────────────────────────────────
    long["series_id"] = long["item_id"] + "_" + long["store_id"]
    daily = long[[
        "date", "item_id", "store_id", "series_id",
        "sales", "sell_price",
        "snap", "event_name_1", "event_type_1",
    ]].rename(columns={
        "event_name_1": "event_name",
        "event_type_1": "event_type",
    }).sort_values(["series_id", "date"]).reset_index(drop=True)

    daily["date"] = pd.to_datetime(daily["date"])
    daily["sales"] = daily["sales"].astype(np.float32)
    daily["sell_price"] = daily["sell_price"].astype(np.float32)

    logger.info(
        "Loaded %d rows | %d unique series | date range %s – %s",
        len(daily),
        daily["series_id"].nunique(),
        daily["date"].min().date(),
        daily["date"].max().date(),
    )
    return daily, cal, prices


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic generator (M5-mimicking)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_synthetic(
    n_items_per_dept: int = 5,
    n_days: int = 700,
    seed: int = 42,
    items_per_dept: int | None = None,
) -> M5Tuple:
    """
    Generate a synthetic dataset that mimics M5 structure and statistics.

    Activated automatically when ``data/raw/m5/`` files are absent.

    Parameters
    ----------
    n_items_per_dept:
        Number of items generated per (dept, store) pair.
    n_days:
        Number of calendar days to simulate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Same ``(daily, calendar, prices)`` tuple structure as :func:`_load_real`.
    """
    if items_per_dept is not None:
        n_items_per_dept = items_per_dept

    rng = np.random.default_rng(seed)
    logger.warning(
        "M5 raw files not found – using SYNTHETIC data (seed=%d, n_days=%d). "
        "Download real data to data/raw/m5/ for paper results.",
        seed,
        n_days,
    )

    dates = pd.date_range(_START_DATE, periods=n_days, freq="D")

    # ── Calendar ──────────────────────────────────────────────────────────────
    cal = pd.DataFrame({"date": dates})
    cal["d"] = [f"d_{i+1}" for i in range(n_days)]
    cal["wm_yr_wk"] = cal["date"].dt.isocalendar().year * 100 + \
                      cal["date"].dt.isocalendar().week
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype(int)
    cal["weekday"]  = cal["date"].dt.day_name()
    cal["wday"]     = (cal["date"].dt.dayofweek + 2) % 7 + 1
    cal["month"]    = cal["date"].dt.month
    cal["year"]     = cal["date"].dt.year
    # Synthetic SNAP: ~20 % of days
    for state in ("CA", "TX", "WI"):
        cal[f"snap_{state}"] = (rng.random(n_days) < 0.20).astype(np.int8)
    # Synthetic events
    event_names = ["SuperBowl", "ValentinesDay", "Christmas", "Thanksgiving", ""]
    event_types = ["Sporting", "Cultural", "National", "National", ""]
    eidx = rng.integers(0, len(event_names), size=n_days)
    cal["event_name_1"] = [event_names[i] for i in eidx]
    cal["event_type_1"] = [event_types[i] for i in eidx]

    # ── Build item list ────────────────────────────────────────────────────────
    items = []
    for cat, depts in _CATS.items():
        for dept in depts:
            for idx in range(1, n_items_per_dept + 1):
                items.append({
                    "item_id":  f"{dept}_{idx:03d}",
                    "dept_id":  dept,
                    "cat_id":   cat,
                })
    items_df = pd.DataFrame(items)

    # ── Prices ────────────────────────────────────────────────────────────────
    price_rows = []
    for _, item in items_df.iterrows():
        for store in _STORES:
            base_price = float(rng.uniform(1.0, 15.0))
            for wk in cal["wm_yr_wk"].unique():
                # Occasional 5–15 % price changes
                shock = float(rng.choice(
                    [1.0, 1.0, 1.0, 1.0,
                     rng.uniform(1.05, 1.15),
                     rng.uniform(0.85, 0.95)],
                ))
                base_price = round(base_price * shock, 2)
                price_rows.append({
                    "store_id":   store,
                    "item_id":    item["item_id"],
                    "wm_yr_wk":   wk,
                    "sell_price": max(0.5, base_price),
                })
    prices_df = pd.DataFrame(price_rows)

    # ── Daily sales (vectorised) ───────────────────────────────────────────────
    state_map = {s: s[:2] for s in _STORES}

    # Pre-index price lookup: (item_id, store_id, wm_yr_wk) → sell_price
    price_index = prices_df.set_index(
        ["item_id", "store_id", "wm_yr_wk"]
    )["sell_price"].to_dict()

    wm_yr_wk_arr  = cal["wm_yr_wk"].values
    wday_arr       = cal["wday"].values  # 1–7
    event_name_arr = cal["event_name_1"].values
    event_type_arr = cal["event_type_1"].values
    date_arr       = cal["date"].values

    # day-of-week multiplier (0=Mon…6=Sun in pandas, but synthetic wday 1=Sat…7=Fri)
    dow_mult_table = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4])  # len 7

    all_frames: list[pd.DataFrame] = []

    for _, item in items_df.iterrows():
        item_id  = item["item_id"]
        dept_id  = item["dept_id"]
        cat_id   = item["cat_id"]

        for store in _STORES:
            state    = state_map[store]
            series_id = f"{item_id}_{store}"
            base_demand = float(rng.uniform(2.0, 12.0))

            # Vectorised price lookup
            prices_arr = np.array([
                price_index.get((item_id, store, int(w)), 3.0)
                for w in wm_yr_wk_arr
            ], dtype=np.float32)

            snap_arr = cal[f"snap_{state}"].values.astype(np.float32)
            dow_mult  = dow_mult_table[(wday_arr - 1) % 7]

            mu = (
                base_demand
                * dow_mult
                * (1.0 + 0.3 * snap_arr)
                / np.maximum(1.0 + 0.05 * (prices_arr - 3.0), 0.1)
            )
            mu = np.maximum(mu, 0.01)
            sales_arr = rng.poisson(mu).astype(np.float32)

            frame = pd.DataFrame({
                "date":       date_arr,
                "item_id":    item_id,
                "dept_id":    dept_id,
                "cat_id":     cat_id,
                "store_id":   store,
                "state_id":   state,
                "series_id":  series_id,
                "sales":      sales_arr,
                "sell_price": prices_arr,
                "wm_yr_wk":   wm_yr_wk_arr,
                "snap":       snap_arr.astype(np.int8),
                "event_name": event_name_arr,
                "event_type": event_type_arr,
            })
            all_frames.append(frame)

    daily = pd.concat(all_frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["series_id", "date"]).reset_index(drop=True)

    logger.info(
        "Synthetic data: %d rows | %d series | %s – %s",
        len(daily),
        daily["series_id"].nunique(),
        daily["date"].min().date(),
        daily["date"].max().date(),
    )
    return daily, cal, prices_df


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def m5_files_present() -> bool:
    """Return True if all three M5 CSVs are present in ``data/raw/m5/``."""
    return all(p.exists() for p in (_SALES_EVAL, _CALENDAR, _PRICES))


def load_m5(
    synthetic_seed: int = 42,
    synthetic_n_days: int = 700,
    synthetic_items_per_dept: int = 5,
    force_synthetic: bool = False,
) -> M5Tuple:
    """
    Load M5 data, falling back to a seeded synthetic dataset if real files
    are absent.

    Parameters
    ----------
    synthetic_seed:
        Seed for the synthetic generator (only used when real data absent).
    synthetic_n_days:
        Number of days in the synthetic dataset.
    synthetic_items_per_dept:
        Items per (dept, store) combo in synthetic mode.
    force_synthetic:
        If True, skip real-file check and always use synthetic data.

    Returns
    -------
    daily : pd.DataFrame
        Columns: date, item_id, store_id, series_id, sales, sell_price,
                 snap, event_name, event_type.
    calendar : pd.DataFrame
        Raw calendar (parsed dates).
    prices : pd.DataFrame
        Raw sell_prices table.
    """
    if not force_synthetic and m5_files_present():
        return _load_real()

    return _make_synthetic(
        n_items_per_dept=synthetic_items_per_dept,
        n_days=synthetic_n_days,
        seed=synthetic_seed,
    )


def load_series(
    daily: pd.DataFrame,
    series_id: str,
) -> pd.DataFrame:
    """
    Extract a single series from the daily DataFrame, sorted by date.

    Parameters
    ----------
    daily:
        Full daily DataFrame returned by :func:`load_m5`.
    series_id:
        The ``series_id`` value (e.g. ``"FOODS_1_001_CA_1"``).

    Returns
    -------
    pd.DataFrame
        Rows for the requested series, index reset.

    Raises
    ------
    KeyError
        If ``series_id`` is not found in the DataFrame.
    """
    mask = daily["series_id"] == series_id
    if not mask.any():
        raise KeyError(
            f"series_id '{series_id}' not found. "
            f"Sample IDs: {daily['series_id'].unique()[:5].tolist()}"
        )
    return daily[mask].sort_values("date").reset_index(drop=True)


# Public alias for testing and synthetic generation benchmarks
generate_synthetic_m5 = _make_synthetic

__all__ = [
    "load_m5",
    "load_series",
    "m5_files_present",
    "generate_synthetic_m5",
]
