"""
src/results_writer.py
=====================
Utility for writing and reading experiment records to/from results/.

Every experiment run appends one row to ``results/experiments.csv``.
Figures and tables are always generated from this CSV — never from
hard-coded or in-memory numbers.

Usage
-----
>>> from src.results_writer import append_record, load_results
>>> from src.schemas import ExperimentRecord
>>>
>>> record = ExperimentRecord(scenario_id=..., seed=42, method="arima", mae=1.2)
>>> append_record(record)
>>>
>>> df = load_results()
>>> print(df.head())
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.schemas import ExperimentRecord

# ── Paths ──────────────────────────────────────────────────────────────────────

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
_EXPERIMENTS_CSV = _RESULTS_DIR / "experiments.csv"

# Column order written to CSV (must stay stable across runs)
_COLUMNS = [
    "run_id",
    "scenario_id",
    "seed",
    "method",
    "mae",
    "rmse",
    "mape",
    "llm_provider",
    "llm_model",
    "elapsed_seconds",
    "notes",
    "run_at",
]


def _ensure_results_dir() -> None:
    """Create results/ directory and a .gitkeep if they don't exist."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gitkeep = _RESULTS_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


# ── Write ──────────────────────────────────────────────────────────────────────


def append_record(record: ExperimentRecord) -> Path:
    """
    Append one :class:`ExperimentRecord` row to ``results/experiments.csv``.

    If the file does not yet exist the header is written first.
    Thread-safety is not guaranteed; for concurrent writes use a lock or
    switch to SQLite.

    Parameters
    ----------
    record:
        A fully-populated experiment record.

    Returns
    -------
    Path
        Absolute path to the CSV file.
    """
    _ensure_results_dir()

    row = {
        "run_id": str(record.run_id),
        "scenario_id": str(record.scenario_id),
        "seed": record.seed,
        "method": record.method.value,
        "mae": record.mae,
        "rmse": record.rmse,
        "mape": record.mape,
        "llm_provider": record.llm_provider,
        "llm_model": record.llm_model,
        "elapsed_seconds": record.elapsed_seconds,
        "notes": record.notes,
        "run_at": record.run_at.isoformat(),
    }

    df_new = pd.DataFrame([row], columns=_COLUMNS)
    write_header = not _EXPERIMENTS_CSV.exists()
    df_new.to_csv(_EXPERIMENTS_CSV, mode="a", header=write_header, index=False)
    return _EXPERIMENTS_CSV


# ── Read ───────────────────────────────────────────────────────────────────────


def load_results(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all experiment records from ``results/experiments.csv``.

    Parameters
    ----------
    csv_path:
        Override the default path (useful for tests).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per experiment run, columns as in ``_COLUMNS``.
        Returns an empty DataFrame (with correct columns) if the file doesn't
        exist yet.
    """
    path = csv_path or _EXPERIMENTS_CSV
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(path, parse_dates=["run_at"])
    return df
