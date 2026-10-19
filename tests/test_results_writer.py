"""
tests/test_results_writer.py
============================
Tests for src/results_writer.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.schemas import ExperimentRecord, ForecastMethod, WhatIfScenario


def _make_record(**kw) -> ExperimentRecord:
    sid = WhatIfScenario(
        name="test", cutoff_date="2023-01-01", target_variable="x", horizon_days=30
    ).scenario_id
    return ExperimentRecord(
        scenario_id=sid, seed=42, method=ForecastMethod.ARIMA, **kw
    )


class TestResultsWriter:
    def test_append_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """append_record must create the CSV on first call."""
        import src.results_writer as rw
        monkeypatch.setattr(rw, "_RESULTS_DIR", tmp_path)
        monkeypatch.setattr(rw, "_EXPERIMENTS_CSV", tmp_path / "experiments.csv")

        rec = _make_record(mae=1.0)
        rw.append_record(rec)
        assert (tmp_path / "experiments.csv").exists()

    def test_append_multiple_rows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Appending N records must produce a CSV with N rows (plus header)."""
        import src.results_writer as rw
        monkeypatch.setattr(rw, "_RESULTS_DIR", tmp_path)
        monkeypatch.setattr(rw, "_EXPERIMENTS_CSV", tmp_path / "experiments.csv")

        for i in range(5):
            rw.append_record(_make_record(mae=float(i)))

        df = pd.read_csv(tmp_path / "experiments.csv")
        assert len(df) == 5

    def test_load_results_empty_when_no_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_results must return an empty DataFrame when the CSV doesn't exist."""
        import src.results_writer as rw
        monkeypatch.setattr(rw, "_EXPERIMENTS_CSV", tmp_path / "experiments.csv")

        df = rw.load_results()
        assert df.empty

    def test_load_results_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Values written must be readable back without loss."""
        import src.results_writer as rw
        monkeypatch.setattr(rw, "_RESULTS_DIR", tmp_path)
        monkeypatch.setattr(rw, "_EXPERIMENTS_CSV", tmp_path / "experiments.csv")

        rec = _make_record(mae=3.14, rmse=4.56, llm_provider="openai", llm_model="gpt-4o")
        rw.append_record(rec)

        df = rw.load_results(csv_path=tmp_path / "experiments.csv")
        assert abs(df.iloc[0]["mae"] - 3.14) < 1e-6
        assert df.iloc[0]["llm_model"] == "gpt-4o"
