"""
scripts/run_baselines.py
========================
CLI entrypoint to evaluate all three baselines (B1 ARIMA+multiplier, B2 ARIMAX,
B3 XGBoost) across all detected what-if episodes.

Results are written to ``results/baselines.csv``.
Never hardcode or invent results; all figures and tables in the paper must
be generated from these CSV records.

Usage
-----
    python scripts/run_baselines.py [--episodes data/processed/episodes.parquet]
                                    [--output results/baselines.csv]
                                    [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env", override=False)

from src.baselines import (
    calculate_metrics,
    get_pre_cutoff_series,
    run_baseline,
)
from src.data_loader import load_m5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_baselines")

DEFAULT_EPISODES_PATH = _ROOT / "data" / "processed" / "episodes.parquet"
DEFAULT_OUTPUT_PATH = _ROOT / "results" / "baselines.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate forecasting baselines B1, B2, and B3 on what-if episodes."
    )
    p.add_argument(
        "--episodes",
        type=Path,
        default=DEFAULT_EPISODES_PATH,
        help=f"Path to episodes parquet file (default: {DEFAULT_EPISODES_PATH}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("RANDOM_SEED", "42")),
        help="Random seed (default: RANDOM_SEED env var or 42).",
    )
    p.add_argument(
        "--synthetic-days",
        type=int,
        default=900,
        help="Days generated in synthetic mode (default: 900).",
    )
    p.add_argument(
        "--synthetic-items",
        type=int,
        default=8,
        help="Items per dept/store in synthetic mode (default: 8).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of episodes to process (for quick smoke tests).",
    )
    return p.parse_args()


def extract_meta_dict(row: pd.Series) -> dict:
    """Extract metadata dictionary from an episode row."""
    meta = {}
    for col in row.index:
        if col.startswith("meta_"):
            val = row[col]
            if pd.notna(val):
                key = col[len("meta_"):]
                meta[key] = val
    return meta


def main() -> None:
    """Run baseline evaluations over all episodes and write results/baselines.csv."""
    args = parse_args()
    t_start = time.perf_counter()

    if not args.episodes.exists():
        logger.error(
            "Episodes file not found at %s. Please run 'python scripts/generate_episodes.py' first.",
            args.episodes,
        )
        sys.exit(1)

    logger.info("Loading episodes from %s...", args.episodes)
    episodes_df = pd.read_parquet(args.episodes)

    if args.limit is not None and args.limit > 0:
        episodes_df = episodes_df.head(args.limit)
        logger.info("Limited run to %d episodes.", len(episodes_df))

    total_episodes = len(episodes_df)
    logger.info("Loaded %d episodes to evaluate.", total_episodes)

    # Load daily M5 series
    logger.info("Loading daily series for pre-cutoff feature extraction...")
    daily, _, _ = load_m5(
        synthetic_seed=args.seed,
        synthetic_n_days=args.synthetic_days,
        synthetic_items_per_dept=args.synthetic_items,
    )

    # Group daily data by series_id for fast retrieval
    logger.info("Indexing daily data by series_id...")
    series_groups = {sid: grp for sid, grp in daily.groupby("series_id")}

    baselines = [
        ("b1_arima", "B1: ARIMA + Multiplier"),
        ("b2_arimax", "B2: ARIMAX / Elasticity"),
        ("b3_xgboost", "B3: XGBoost (Lag/Cal/Scenario)"),
    ]

    records = []
    logger.info("Evaluating baselines...")

    for idx, (_, ep) in enumerate(episodes_df.iterrows(), start=1):
        sid = ep["series_id"]
        stype = ep["scenario_type"]
        cutoff = pd.Timestamp(ep["cutoff_date"])
        actuals = np.asarray(ep["actuals"], dtype=float)
        meta = extract_meta_dict(ep)

        if sid not in series_groups:
            logger.warning("Series %s not found in daily dataset. Skipping.", sid)
            continue

        raw_series = series_groups[sid]
        # Strict zero-leakage pre-cutoff slice
        pre_cutoff_series = get_pre_cutoff_series(raw_series, cutoff)

        if len(pre_cutoff_series) == 0:
            logger.warning("No pre-cutoff history for series %s at %s. Skipping.", sid, cutoff)
            continue

        for method_id, method_label in baselines:
            t0 = time.perf_counter()
            try:
                pred = run_baseline(
                    method=method_id,
                    series_pre_cutoff=pre_cutoff_series,
                    cutoff_date=cutoff,
                    scenario_type=stype,
                    meta=meta,
                    horizon_days=len(actuals),
                    seed=args.seed,
                )
                elapsed = time.perf_counter() - t0
                metrics = calculate_metrics(actuals, pred)

                rec = {
                    "run_id": str(uuid4()),
                    "series_id": sid,
                    "scenario_type": stype,
                    "cutoff_date": cutoff.isoformat(),
                    "method": method_id,
                    "method_label": method_label,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "wape": metrics["wape"],
                    "smape": metrics["smape"],
                    "elapsed_seconds": round(elapsed, 4),
                    "seed": args.seed,
                }
                records.append(rec)
            except Exception as e:
                logger.error("Error evaluating %s on %s (%s): %s", method_id, sid, stype, e)

        if idx % 15 == 0 or idx == total_episodes:
            logger.info("Progress: %d / %d episodes evaluated...", idx, total_episodes)

    # Save to CSV
    results_df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    logger.info("Saved %d baseline experiment records -> %s", len(results_df), args.output)

    # Print summary table
    elapsed_total = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print("PredictAI - Baseline Evaluation Summary")
    print("=" * 70)
    print(f"Total Evaluations : {len(results_df)}")
    print(f"Total Episodes    : {total_episodes}")
    print(f"Output File       : {args.output}")
    print(f"Elapsed Time      : {elapsed_total:.2f}s")
    print("-" * 70)

    summary_table = (
        results_df.groupby(["scenario_type", "method"])
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_wape=("wape", "mean"),
            count=("mae", "count"),
        )
        .round(3)
    )
    print(summary_table.to_string())

    print("-" * 70)
    overall_table = (
        results_df.groupby("method")
        .agg(
            overall_mae=("mae", "mean"),
            overall_rmse=("rmse", "mean"),
            overall_wape=("wape", "mean"),
        )
        .round(3)
    )
    print("Overall Baseline Averages:")
    print(overall_table.to_string())
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
