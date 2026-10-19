"""
scripts/run_experiments.py
==========================
Master experimental benchmarking and evaluation script for PredictAI.

Executes:
1. Forecasting baselines (B1 ARIMA+multiplier, B2 ARIMAX, B3 XGBoost)
2. PredictAI Full Pipeline (Base + Episodic Memory + LLM Reasoning + Verify Guardrail)
3. Ablation studies:
   - w/o Episodic Retrieval (memory=None)
   - w/o Guardrail Verification (unverified LLM projection)
   - w/o LLM (pure unadjusted statistical base)
4. Metrics:
   - MAE, RMSE, WAPE (%)
   - Scenario-Response Accuracy at +/-10%, +/-15%, +/-20% tolerance
   - Paired Wilcoxon signed-rank tests vs PredictAI
   - 95% Bootstrap Confidence Intervals (1000 resamples)
   - Real-world latency & API cost tracking
5. Generation of publication-ready journal artifacts:
   - results/evaluation_results.csv
   - results/table1_main_results.csv
   - results/table2_ablation_study.csv
   - results/table3_by_scenario_type.csv
   - results/figures/figure1_model_comparison_wape.{png,pdf}
   - results/figures/figure2_scenario_response_accuracy_sensitivity.{png,pdf}
   - results/figures/figure3_ablation_breakdown.{png,pdf}
   - results/figures/figure4_trajectory_examples.{png,pdf}

Usage:
------
    python scripts/run_experiments.py [--episodes data/processed/episodes.parquet]
                                      [--seed 42]
                                      [--limit 90]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env", override=False)

import numpy as np
import pandas as pd

from src.data_loader import load_m5
from src.evaluator import (
    generate_journal_figures,
    generate_summary_tables,
    run_pipeline_and_ablations,
)
from src.memory import EpisodeMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiments")

DEFAULT_EPISODES_PATH = _ROOT / "data" / "processed" / "episodes.parquet"
DEFAULT_MEMORY_DIR = _ROOT / "data" / "processed" / "memory"
DEFAULT_RESULTS_DIR = _ROOT / "results"
DEFAULT_FIGURES_DIR = _ROOT / "results" / "figures"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Run master evaluation of PredictAI, baselines, and ablations."
    )
    p.add_argument(
        "--episodes",
        type=Path,
        default=DEFAULT_EPISODES_PATH,
        help=f"Path to episodes parquet file (default: {DEFAULT_EPISODES_PATH}).",
    )
    p.add_argument(
        "--memory-dir",
        type=Path,
        default=DEFAULT_MEMORY_DIR,
        help=f"Path to FAISS memory directory (default: {DEFAULT_MEMORY_DIR}).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory to save evaluation results and CSV tables (default: {DEFAULT_RESULTS_DIR}).",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Directory to save journal-style figures (default: {DEFAULT_FIGURES_DIR}).",
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
        help="Optional cap on number of episodes to process.",
    )
    return p.parse_args()


def print_table_box(title: str, df: pd.DataFrame) -> None:
    """Print a nicely formatted ASCII table to stdout."""
    print("\n" + "=" * 90)
    print(f" {title}")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


def main() -> None:
    """Run full benchmarking suite."""
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
    total_episodes = len(episodes_df)
    logger.info("Loaded %d episodes across %d scenario types.", total_episodes, episodes_df["scenario_type"].nunique())

    # Load daily series
    logger.info("Loading daily series for pre-cutoff time series modeling...")
    daily, _, _ = load_m5(
        synthetic_seed=args.seed,
        synthetic_n_days=args.synthetic_days,
        synthetic_items_per_dept=args.synthetic_items,
    )

    # Load or build EpisodeMemory
    memory = EpisodeMemory()
    faiss_file = args.memory_dir / "episodes.faiss"
    meta_file = args.memory_dir / "episodes_meta.json"

    if faiss_file.exists() and meta_file.exists():
        logger.info("Loading prebuilt episodic memory from %s...", args.memory_dir)
        memory.load(args.memory_dir)
    else:
        logger.info("Prebuilt memory not found. Constructing memory from episodes...")
        memory.build_from_episodes(episodes_df=episodes_df, daily_df=daily)
        memory.save(args.memory_dir)

    # Run experiments across all methods and ablations
    logger.info(
        "Starting comprehensive evaluation (Seed: %d, Limit: %s)...",
        args.seed,
        str(args.limit) if args.limit else "None (all 90)",
    )
    eval_df = run_pipeline_and_ablations(
        episodes_df=episodes_df,
        daily_df=daily,
        memory=memory,
        seed=args.seed,
        limit=args.limit,
    )

    # Save raw granular evaluation records
    args.results_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = args.results_dir / "evaluation_results.csv"
    eval_df.to_csv(raw_results_path, index=False)
    logger.info("Saved granular evaluation results (%d rows) -> %s", len(eval_df), raw_results_path)

    # Generate published statistical summary tables
    logger.info("Generating statistical summary tables with 95% bootstrap CIs and Wilcoxon tests...")
    tables = generate_summary_tables(eval_df, output_dir=args.results_dir)

    # Generate journal-style black and white hatched figures
    logger.info("Generating journal figures (B&W hatched PNG + PDF)...")
    generate_journal_figures(eval_df, figures_dir=args.figures_dir)

    # Display tables in console
    print_table_box("TABLE 1: MAIN FORECASTING MODEL COMPARISON (M5 WHAT-IF SIMULATION)", tables["table1"])
    print_table_box("TABLE 2: ABLATION STUDY - ARCHITECTURAL CONTRIBUTION ANALYSIS", tables["table2"])
    print_table_box("TABLE 3: PERFORMANCE DISAGGREGATED BY SCENARIO TYPE", tables["table3"])

    elapsed = time.perf_counter() - t_start
    logger.info("All experiments, statistical tests, tables, and figures completed in %.2fs. [OK]", elapsed)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
