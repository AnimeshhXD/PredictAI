"""
src/episode_detector.py
=======================
Detect real what-if episodes from the M5 daily series.

Episode types
-------------
price_up
    A sustained price rise of 8–12 % held for at least 7 consecutive days,
    with no simultaneous demand drop that would explain the increase.

demand_surge
    The 4-week rolling mean demand is 15–25 % above the prior 8-week baseline,
    without a concurrent price cut (which would confound the signal).

supply_delay
    3–7 consecutive zero-sales days after a period of normal (non-zero) sales —
    a stockout / supply-chain disruption proxy.

All episode detectors:
* Enforce a minimum ``lookback`` of history before the cutoff.
* Emit a ``cutoff_date``: the first day of the episode (the day the
  simulator's user asks "what if this had happened?").
* Attach the actual next 28 days as the ground-truth ``actuals`` array.
* Generate a plain-language ``scenario_text`` for LLM consumption.
* Pass ``assert_no_leakage`` — training data is strictly before cutoff.

Usage
-----
>>> from src.episode_detector import detect_all_episodes
>>> episodes = detect_all_episodes(daily, min_per_type=30, seed=42)
>>> print(episodes.groupby("scenario_type").size())
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.leakage_guard import assert_no_leakage

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
HORIZON_DAYS      = 28          # ground-truth actuals window
MIN_LOOKBACK_DAYS = 90          # series must have ≥90 days before cutoff
MAX_EPISODES_PER_SERIES = 1     # at most 1 episode per (series, type)

# price_up thresholds
PRICE_UP_MIN_PCT   = 0.08       # 8 %
PRICE_UP_MAX_PCT   = 0.12       # 12 %
PRICE_UP_MIN_DAYS  = 7          # sustained for ≥7 days

# demand_surge thresholds
SURGE_WINDOW_DAYS  = 28         # 4 weeks
SURGE_BASE_DAYS    = 56         # 8 weeks baseline
SURGE_MIN_PCT      = 0.15       # 15 %
SURGE_MAX_PCT      = 0.25       # 25 %
SURGE_PRICE_CUT    = -0.03      # ignore if price dropped > 3 %

# supply_delay thresholds
STOCKOUT_MIN_DAYS  = 3
STOCKOUT_MAX_DAYS  = 7
NORMAL_SALES_MIN   = 1.0        # mean sales must be ≥1 before zeros


# ── Episode record ─────────────────────────────────────────────────────────────


@dataclass
class Episode:
    """
    One detected what-if episode.

    Attributes
    ----------
    series_id:
        Unique series identifier (``item_id_store_id``).
    scenario_type:
        One of ``price_up``, ``demand_surge``, ``supply_delay``.
    cutoff_date:
        The date marking the start of the episode (exclusive upper bound for
        training data).
    scenario_text:
        Plain-language description suitable for LLM prompting.
    actuals:
        Actual sales for the next ``HORIZON_DAYS`` days after the cutoff.
    actual_dates:
        Corresponding dates for ``actuals``.
    meta:
        Arbitrary detector metadata (e.g. price change magnitude).
    """

    series_id:     str
    scenario_type: str
    cutoff_date:   pd.Timestamp
    scenario_text: str
    actuals:       np.ndarray
    actual_dates:  np.ndarray                    # dtype datetime64[ns]
    meta:          dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a flat dict (actuals stored as JSON-serialisable list)."""
        return {
            "series_id":     self.series_id,
            "scenario_type": self.scenario_type,
            "cutoff_date":   str(self.cutoff_date.date()),
            "scenario_text": self.scenario_text,
            "actuals":       self.actuals.tolist(),
            "actual_dates":  [str(pd.Timestamp(d).date()) for d in self.actual_dates],
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-series detectors
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_price_up(
    s: pd.DataFrame,
    rng: np.random.Generator,
) -> Optional[Episode]:
    """
    Detect a single ``price_up`` episode in series *s*.

    A price-up episode is a window where the weekly average sell_price rises
    by PRICE_UP_MIN_PCT to PRICE_UP_MAX_PCT relative to the prior 4-week
    baseline and stays elevated for at least PRICE_UP_MIN_DAYS.

    Parameters
    ----------
    s:
        Single-series daily DataFrame (columns: date, sales, sell_price, …).
    rng:
        Seeded NumPy random generator (used for tie-breaking when multiple
        candidate windows exist — picks one uniformly at random).

    Returns
    -------
    Episode or None
    """
    s = s.sort_values("date").reset_index(drop=True)
    if len(s) < MIN_LOOKBACK_DAYS + HORIZON_DAYS + 1:
        return None

    prices = s["sell_price"].values
    dates  = pd.to_datetime(s["date"]).values

    candidates: list[int] = []

    # Slide a window: at index i, compare mean(prices[i:i+7]) to mean(prices[i-28:i])
    baseline_window = 28
    check_window    = 7

    for i in range(baseline_window, len(s) - check_window - HORIZON_DAYS):
        # A price increase must initiate at day i
        if prices[i] <= prices[i - 1]:
            continue

        baseline = np.nanmean(prices[i - baseline_window: i])
        if baseline <= 0:
            continue
        new_price = np.nanmean(prices[i: i + check_window])
        pct_change = (new_price - baseline) / baseline

        if PRICE_UP_MIN_PCT <= pct_change <= PRICE_UP_MAX_PCT:
            # Verify it's sustained for PRICE_UP_MIN_DAYS
            sustained = prices[i: i + PRICE_UP_MIN_DAYS]
            if np.all(sustained >= new_price * 0.98):
                # Ensure enough history before cutoff
                if i >= MIN_LOOKBACK_DAYS:
                    candidates.append(i)

    if not candidates:
        return None

    idx = int(rng.choice(candidates))
    cutoff   = pd.Timestamp(dates[idx])
    baseline_price = float(np.nanmean(prices[idx - baseline_window: idx]))
    new_p    = float(np.nanmean(prices[idx: idx + check_window]))
    pct      = (new_p - baseline_price) / baseline_price * 100

    actuals      = s["sales"].values[idx: idx + HORIZON_DAYS].astype(float)
    actual_dates = dates[idx: idx + HORIZON_DAYS]

    # Leakage check: training slice must be entirely before cutoff
    train_slice = s.iloc[:idx]
    assert_no_leakage(train_slice, date_col="date", cutoff=cutoff)

    series_id = s["series_id"].iloc[0]
    item_id   = s["item_id"].iloc[0]
    store_id  = s["store_id"].iloc[0]

    text = textwrap.dedent(f"""
        What-if scenario: price increase for {item_id} at {store_id}.

        On {cutoff.date()}, the sell price of item {item_id} at store {store_id}
        rose by approximately {pct:.1f}% from a prior 4-week average of
        ${baseline_price:.2f} to ${new_p:.2f}. This sustained price rise
        (held for at least {PRICE_UP_MIN_DAYS} days) is within the historical
        range of {PRICE_UP_MIN_PCT*100:.0f}-{PRICE_UP_MAX_PCT*100:.0f}%.

        Question: Forecast daily unit sales for the next {HORIZON_DAYS} days
        given this price increase, assuming no other promotional changes.
    """).strip()

    return Episode(
        series_id=series_id,
        scenario_type="price_up",
        cutoff_date=cutoff,
        scenario_text=text,
        actuals=actuals,
        actual_dates=actual_dates,
        meta={
            "baseline_price": round(baseline_price, 4),
            "new_price":      round(new_p, 4),
            "pct_change":     round(pct, 2),
        },
    )


def _detect_demand_surge(
    s: pd.DataFrame,
    rng: np.random.Generator,
) -> Optional[Episode]:
    """
    Detect a single ``demand_surge`` episode in series *s*.

    A demand surge is when the 4-week rolling mean is 15–25 % above the prior
    8-week baseline and no price cut (≥3 %) occurred in the window.

    Parameters
    ----------
    s:
        Single-series daily DataFrame.
    rng:
        Seeded NumPy random generator for candidate selection.

    Returns
    -------
    Episode or None
    """
    s = s.sort_values("date").reset_index(drop=True)
    needed = SURGE_BASE_DAYS + SURGE_WINDOW_DAYS + HORIZON_DAYS + MIN_LOOKBACK_DAYS
    if len(s) < needed:
        return None

    sales  = s["sales"].values.astype(float)
    prices = s["sell_price"].values.astype(float)
    dates  = pd.to_datetime(s["date"]).values
    start  = SURGE_BASE_DAYS + MIN_LOOKBACK_DAYS

    candidates: list[int] = []

    for i in range(start, len(s) - SURGE_WINDOW_DAYS - HORIZON_DAYS):
        baseline = np.mean(sales[i - SURGE_BASE_DAYS: i])
        if baseline < NORMAL_SALES_MIN:
            continue
        surge_mean = np.mean(sales[i: i + SURGE_WINDOW_DAYS])
        pct_change = (surge_mean - baseline) / baseline

        if SURGE_MIN_PCT <= pct_change <= SURGE_MAX_PCT:
            # Reject if price fell by more than SURGE_PRICE_CUT
            baseline_price = np.nanmean(prices[i - SURGE_BASE_DAYS: i])
            surge_price    = np.nanmean(prices[i: i + SURGE_WINDOW_DAYS])
            if baseline_price > 0:
                price_pct = (surge_price - baseline_price) / baseline_price
                if price_pct < SURGE_PRICE_CUT:
                    continue   # price cut drove the surge — not a pure demand shock
            candidates.append(i)

    if not candidates:
        return None

    idx = int(rng.choice(candidates))
    cutoff       = pd.Timestamp(dates[idx])
    baseline_val = float(np.mean(sales[idx - SURGE_BASE_DAYS: idx]))
    surge_val    = float(np.mean(sales[idx: idx + SURGE_WINDOW_DAYS]))
    pct          = (surge_val - baseline_val) / baseline_val * 100

    actuals      = sales[idx: idx + HORIZON_DAYS]
    actual_dates = dates[idx: idx + HORIZON_DAYS]

    train_slice = s.iloc[:idx]
    assert_no_leakage(train_slice, date_col="date", cutoff=cutoff)

    series_id = s["series_id"].iloc[0]
    item_id   = s["item_id"].iloc[0]
    store_id  = s["store_id"].iloc[0]

    text = textwrap.dedent(f"""
        What-if scenario: unexpected demand surge for {item_id} at {store_id}.

        Starting {cutoff.date()}, daily sales of {item_id} at {store_id}
        averaged {surge_val:.1f} units/day over 4 weeks - a {pct:.1f}% increase
        over the prior 8-week baseline of {baseline_val:.1f} units/day.
        No price discount was offered during this window, suggesting an
        organic demand shock (e.g. social-media trend, external event).

        Question: Forecast daily unit sales for the next {HORIZON_DAYS} days,
        assuming the elevated demand partially persists then reverts to baseline.
    """).strip()

    return Episode(
        series_id=series_id,
        scenario_type="demand_surge",
        cutoff_date=cutoff,
        scenario_text=text,
        actuals=actuals,
        actual_dates=actual_dates,
        meta={
            "baseline_demand": round(baseline_val, 4),
            "surge_demand":    round(surge_val, 4),
            "pct_change":      round(pct, 2),
        },
    )


def _detect_supply_delay(
    s: pd.DataFrame,
    rng: np.random.Generator,
) -> Optional[Episode]:
    """
    Detect a single ``supply_delay`` episode in series *s*.

    A supply delay is a run of STOCKOUT_MIN_DAYS to STOCKOUT_MAX_DAYS
    consecutive zero-sales days preceded by a period of normal (≥1 unit/day)
    sales — a stockout / supply-disruption proxy.

    Parameters
    ----------
    s:
        Single-series daily DataFrame.
    rng:
        Seeded NumPy random generator for candidate selection.

    Returns
    -------
    Episode or None
    """
    s = s.sort_values("date").reset_index(drop=True)
    if len(s) < MIN_LOOKBACK_DAYS + HORIZON_DAYS + STOCKOUT_MAX_DAYS + 1:
        return None

    sales  = s["sales"].values.astype(float)
    dates  = pd.to_datetime(s["date"]).values

    candidates: list[int] = []
    i = MIN_LOOKBACK_DAYS

    while i < len(s) - HORIZON_DAYS - STOCKOUT_MAX_DAYS:
        # A stockout episode starts on day i only if preceded by non-zero sales
        if i > 0 and sales[i - 1] == 0.0:
            i += 1
            continue

        # Count run of zeros starting at i
        run = 0
        while i + run < len(s) and sales[i + run] == 0.0:
            run += 1

        if STOCKOUT_MIN_DAYS <= run <= STOCKOUT_MAX_DAYS:
            # Confirm prior period had normal sales (mean >= NORMAL_SALES_MIN)
            lookback_start = max(0, i - 28)
            prior_mean = np.mean(sales[lookback_start: i])
            if prior_mean >= NORMAL_SALES_MIN:
                # Ensure enough actuals after the run
                if i + HORIZON_DAYS <= len(s):
                    candidates.append(i)
            i += run  # skip past the zero run
        elif run > STOCKOUT_MAX_DAYS:
            i += run  # skip prolonged zero run (delisting proxy)
        else:
            i += 1

    if not candidates:
        return None

    idx = int(rng.choice(candidates))
    cutoff = pd.Timestamp(dates[idx])

    # Count run length at chosen index
    run_len = 0
    while idx + run_len < len(s) and sales[idx + run_len] == 0.0:
        run_len += 1

    prior_mean   = float(np.mean(sales[max(0, idx - 28): idx]))
    actuals      = sales[idx: idx + HORIZON_DAYS]
    actual_dates = dates[idx: idx + HORIZON_DAYS]

    train_slice = s.iloc[:idx]
    assert_no_leakage(train_slice, date_col="date", cutoff=cutoff)

    series_id = s["series_id"].iloc[0]
    item_id   = s["item_id"].iloc[0]
    store_id  = s["store_id"].iloc[0]

    text = textwrap.dedent(f"""
        What-if scenario: supply disruption / stockout for {item_id} at {store_id}.

        On {cutoff.date()}, {item_id} at {store_id} recorded zero sales for
        {run_len} consecutive days - a likely stockout or supply-chain delay.
        In the 4 weeks prior, the item averaged {prior_mean:.1f} units/day,
        indicating normal replenishment cycles had been running.

        Question: Forecast daily unit sales for the next {HORIZON_DAYS} days,
        modelling the expected recovery trajectory once stock is replenished.
    """).strip()

    return Episode(
        series_id=series_id,
        scenario_type="supply_delay",
        cutoff_date=cutoff,
        scenario_text=text,
        actuals=actuals,
        actual_dates=actual_dates,
        meta={
            "stockout_days": run_len,
            "prior_mean":    round(prior_mean, 4),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


def detect_all_episodes(
    daily: pd.DataFrame,
    min_per_type: int = 30,
    seed: int = 42,
    max_series_scan: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run all three episode detectors over every unique series in *daily*.

    Stops early once ``min_per_type`` episodes are collected for *each* type,
    unless all series have been scanned.

    Parameters
    ----------
    daily:
        Full daily DataFrame from :func:`src.data_loader.load_m5`.
    min_per_type:
        Target minimum number of episodes per scenario type.
    seed:
        Master random seed passed to each per-series detector.
    max_series_scan:
        If set, scan at most this many series (useful for quick testing).

    Returns
    -------
    pd.DataFrame
        One row per episode. Columns:
        series_id, scenario_type, cutoff_date, scenario_text,
        actuals (JSON list), actual_dates (JSON list), meta_*.
    """
    rng = np.random.default_rng(seed)
    all_series = np.array(daily["series_id"].unique())
    rng.shuffle(all_series)

    if max_series_scan is not None:
        all_series = all_series[:max_series_scan]

    episodes: list[Episode] = []
    counts   = {"price_up": 0, "demand_surge": 0, "supply_delay": 0}
    detectors = {
        "price_up":     _detect_price_up,
        "demand_surge": _detect_demand_surge,
        "supply_delay": _detect_supply_delay,
    }

    for sid in all_series:
        if all(v >= min_per_type for v in counts.values()):
            logger.info("Target %d episodes/type reached — stopping scan.", min_per_type)
            break

        s = daily[daily["series_id"] == sid].sort_values("date").reset_index(drop=True)

        for etype, detector in detectors.items():
            if counts[etype] >= min_per_type:
                continue
            try:
                ep = detector(s, rng)
                if ep is not None:
                    episodes.append(ep)
                    counts[etype] += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("Detector %s failed on %s: %s", etype, sid, exc)

    logger.info("Episodes detected: %s", counts)
    missing = {k: v for k, v in counts.items() if v < min_per_type}
    if missing:
        logger.warning(
            "Could not reach target of %d episodes/type. Shortfall: %s. "
            "Consider using a longer series history or a larger dataset.",
            min_per_type,
            missing,
        )

    if not episodes:
        raise RuntimeError(
            "No episodes detected. The dataset may be too short or too sparse. "
            "Try increasing synthetic_n_days in load_m5()."
        )

    rows = [ep.to_dict() for ep in episodes]
    df   = pd.DataFrame(rows)
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    return df


def summary_report(episodes_df: pd.DataFrame) -> str:
    """
    Produce a plain-text summary report of the detected episodes.

    Parameters
    ----------
    episodes_df:
        DataFrame returned by :func:`detect_all_episodes`.

    Returns
    -------
    str
        Multi-line report string.
    """
    lines = [
        "=" * 60,
        "PredictAI - Episode Detection Summary",
        "=" * 60,
        f"Total episodes : {len(episodes_df)}",
        f"Unique series  : {episodes_df['series_id'].nunique()}",
        "",
        "Counts by scenario type:",
    ]
    for stype, grp in episodes_df.groupby("scenario_type"):
        lines.append(f"  {stype:<15} {len(grp):>4}")

    lines += [
        "",
        "Date range of cutoffs:",
        f"  Earliest : {episodes_df['cutoff_date'].min().date()}",
        f"  Latest   : {episodes_df['cutoff_date'].max().date()}",
    ]

    # Per-type metadata summary
    for stype, grp in episodes_df.groupby("scenario_type"):
        lines += ["", f"-- {stype} --"]
        if stype == "price_up" and "meta_pct_change" in grp.columns:
            lines.append(
                f"  Price change  : "
                f"mean={grp['meta_pct_change'].mean():.1f}%  "
                f"[{grp['meta_pct_change'].min():.1f}%, {grp['meta_pct_change'].max():.1f}%]"
            )
        elif stype == "demand_surge" and "meta_pct_change" in grp.columns:
            lines.append(
                f"  Demand lift   : "
                f"mean={grp['meta_pct_change'].mean():.1f}%  "
                f"[{grp['meta_pct_change'].min():.1f}%, {grp['meta_pct_change'].max():.1f}%]"
            )
        elif stype == "supply_delay" and "meta_stockout_days" in grp.columns:
            lines.append(
                f"  Stockout days : "
                f"mean={grp['meta_stockout_days'].mean():.1f}  "
                f"[{grp['meta_stockout_days'].min()}, {grp['meta_stockout_days'].max()}]"
            )

    lines.append("=" * 60)
    return "\n".join(lines)
