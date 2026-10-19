"""
scripts/build_memory.py
=======================
CLI entrypoint to construct and persist the FAISS episode memory from
data/processed/episodes.parquet.

Saves:
- data/processed/memory/episodes.faiss
- data/processed/memory/episodes_meta.json

Usage:
------
    python scripts/build_memory.py [--episodes data/processed/episodes.parquet]
                                   [--output data/processed/memory]
                                   [--seed 42]
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

import pandas as pd

from src.data_loader import load_m5
from src.memory import EpisodeMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_memory")

DEFAULT_EPISODES_PATH = _ROOT / "data" / "processed" / "episodes.parquet"
DEFAULT_MEMORY_DIR = _ROOT / "data" / "processed" / "memory"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Build and persist FAISS episode memory from detected episodes."
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
        default=DEFAULT_MEMORY_DIR,
        help=f"Output directory for FAISS index and metadata (default: {DEFAULT_MEMORY_DIR}).",
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
    return p.parse_args()


def main() -> None:
    """Build and save FAISS episode memory."""
    args = parse_args()
    t0 = time.perf_counter()

    if not args.episodes.exists():
        logger.error(
            "Episodes file not found at %s. Please run 'python scripts/generate_episodes.py' first.",
            args.episodes,
        )
        sys.exit(1)

    logger.info("Loading episodes from %s...", args.episodes)
    episodes_df = pd.read_parquet(args.episodes)
    logger.info("Loaded %d episodes to index.", len(episodes_df))

    logger.info("Loading daily M5 series...")
    daily, _, _ = load_m5(
        synthetic_seed=args.seed,
        synthetic_n_days=args.synthetic_days,
        synthetic_items_per_dept=args.synthetic_items,
    )

    logger.info("Constructing EpisodeMemory...")
    memory = EpisodeMemory()
    memory.build_from_episodes(episodes_df=episodes_df, daily_df=daily)

    logger.info("Persisting memory to %s...", args.output)
    memory.save(args.output)

    elapsed = time.perf_counter() - t0
    logger.info("Episode memory successfully built and saved in %.2fs. [OK]", elapsed)

    # Smoke test retrieval
    sample_cutoff = episodes_df["cutoff_date"].max()
    sample_results = memory.retrieve(
        query_scenario="Unexpected price increase of 10% on foods",
        cutoff_date=sample_cutoff,
        k=2,
    )
    print("\n" + "=" * 65)
    print("PredictAI - Episode Memory Smoke Test Retrieval")
    print("=" * 65)
    print(f"Total Indexed Episodes: {len(memory)}")
    print(f"Sample Query Cutoff   : {sample_cutoff.date()}")
    print(f"Sample Retrieved Count : {len(sample_results)}")
    for i, res in enumerate(sample_results, 1):
        print(f"\nMatch #{i} (Score: {res.similarity_score}):")
        print(f"  Type       : {res.scenario_type}")
        print(f"  Series     : {res.series_id}")
        print(f"  Cutoff     : {res.cutoff_date.date()} -> End: {res.end_date.date()}")
        print(f"  Multipliers: +7d={res.multiplier_7:.2f}, +14d={res.multiplier_14:.2f}, +28d={res.multiplier_28:.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
