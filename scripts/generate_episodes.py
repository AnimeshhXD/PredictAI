"""
scripts/generate_episodes.py
=============================
CLI entrypoint for the episode-detection pipeline.

Invoked by ``make episodes``.

Steps
-----
1. Load M5 data (real CSVs if present, else seeded synthetic).
2. Run all three episode detectors.
3. Save ``data/processed/episodes.parquet``.
4. Write ``data/processed/episode_summary.txt``.
5. Print the summary report.

Usage
-----
    python scripts/generate_episodes.py [--min-per-type 30] [--seed 42]

Environment variables read (from .env)
---------------------------------------
RANDOM_SEED   : integer seed (default 42)
M5_CUTOFF_DATE: not used here — cutoff is determined per episode
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src/ is importable regardless of CWD
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env", override=False)

from src.data_loader import load_m5, m5_files_present
from src.episode_detector import detect_all_episodes, summary_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_episodes")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = _ROOT / "data" / "processed"
EPISODES_PATH  = PROCESSED_DIR / "episodes.parquet"
SUMMARY_PATH   = PROCESSED_DIR / "episode_summary.txt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Detect M5 what-if episodes and save to data/processed/."
    )
    p.add_argument(
        "--min-per-type",
        type=int,
        default=30,
        help="Minimum episodes per scenario type (default: 30).",
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
        help="Days to generate in synthetic mode (default: 900).",
    )
    p.add_argument(
        "--synthetic-items",
        type=int,
        default=8,
        help="Items per dept/store in synthetic mode (default: 8).",
    )
    p.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Cap series scanned (useful for quick smoke tests).",
    )
    return p.parse_args()


def main() -> None:
    """Run the episode-detection pipeline end-to-end."""
    args = parse_args()
    t0   = time.perf_counter()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    mode = "REAL" if m5_files_present() else "SYNTHETIC"
    logger.info("Data mode: %s | seed=%d", mode, args.seed)

    daily, calendar, prices = load_m5(
        synthetic_seed=args.seed,
        synthetic_n_days=args.synthetic_days,
        synthetic_items_per_dept=args.synthetic_items,
    )
    logger.info(
        "Loaded %d rows | %d series | %s – %s",
        len(daily),
        daily["series_id"].nunique(),
        daily["date"].min().date(),
        daily["date"].max().date(),
    )

    # ── 2. Detect episodes ────────────────────────────────────────────────────
    logger.info("Detecting episodes (min_per_type=%d)…", args.min_per_type)
    episodes_df = detect_all_episodes(
        daily,
        min_per_type=args.min_per_type,
        seed=args.seed,
        max_series_scan=args.max_series,
    )

    # ── 3. Save parquet ───────────────────────────────────────────────────────
    episodes_df.to_parquet(EPISODES_PATH, index=False)
    logger.info("Saved %d episodes -> %s", len(episodes_df), EPISODES_PATH)

    # ── 4. Summary report ─────────────────────────────────────────────────────
    report = summary_report(episodes_df)
    SUMMARY_PATH.write_text(report, encoding="utf-8")
    logger.info("Summary report -> %s", SUMMARY_PATH)

    elapsed = time.perf_counter() - t0
    print()
    print(report)
    print(f"\n[Done in {elapsed:.1f}s]")

    # Validation: assert minimum counts met
    counts = episodes_df.groupby("scenario_type").size().to_dict()
    for stype in ("price_up", "demand_surge", "supply_delay"):
        n = counts.get(stype, 0)
        if n < args.min_per_type:
            logger.error(
                "SHORTFALL: only %d/%d episodes for '%s'. "
                "Try --synthetic-days 1200 or --synthetic-items 12.",
                n, args.min_per_type, stype,
            )
            sys.exit(1)

    logger.info("All episode counts meet the minimum threshold. [OK]")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
