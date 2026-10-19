"""
tests/test_verifier.py
======================
Tests for src/verifier.py (Verify()).

Verifies:
1. Rolling-origin backtest error band estimation on pre-cutoff series.
2. Analog distribution 2-standard-deviation bounds check.
3. Basic constraint enforcement (non-negativity, finite bounds).
4. Failure recovery with widened retrieval and feedback re-prompting (max 2 retries).
5. Comprehensive audit logging to results/verification_log.csv across all states:
   passed, failed, retried, final.
6. Data leakage prevention: passing future data raises DataLeakageError.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.leakage_guard import DataLeakageError
from src.llm_reasoner import LLMReasonResult
from src.memory import EpisodeMemory, HistoricalEpisode
from src.verifier import (
    HORIZON_DAYS,
    LOG_COLUMNS,
    Verify,
    check_analog_distribution,
    check_basic_constraints,
    compute_rolling_backtest_error_band,
    log_verification_event,
)


def _make_dummy_daily_series(n_days: int = 150) -> pd.DataFrame:
    """Create clean pre-cutoff daily series."""
    dates = pd.date_range("2011-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    sales = rng.poisson(lam=8.0, size=n_days).astype(float)
    prices = np.full(n_days, 4.0, dtype=float)

    return pd.DataFrame({
        "date": dates,
        "item_id": "FOODS_1_001",
        "store_id": "CA_1",
        "series_id": "FOODS_1_001_CA_1",
        "sales": sales,
        "sell_price": prices,
    })


# ── Backtest Error Band Tests ──────────────────────────────────────────────────


class TestBacktestErrorBand:
    """Tests for compute_rolling_backtest_error_band."""

    def test_backtest_error_band_shape_and_positivity(self) -> None:
        df = _make_dummy_daily_series(n_days=120)
        cutoff = df["date"].max() + pd.Timedelta(days=1)

        rmse, lower_off, upper_off = compute_rolling_backtest_error_band(
            series_pre_cutoff=df,
            cutoff_date=cutoff,
            n_origins=2,
            horizon_days=28,
        )

        assert rmse > 0.0
        assert len(lower_off) == HORIZON_DAYS
        assert len(upper_off) == HORIZON_DAYS
        assert np.all(upper_off > 0.0)
        assert np.all(lower_off < 0.0)

    def test_backtest_raises_on_future_leakage(self) -> None:
        df = _make_dummy_daily_series(n_days=100)
        leaky_cutoff = df["date"].iloc[50]

        with pytest.raises(DataLeakageError):
            compute_rolling_backtest_error_band(
                series_pre_cutoff=df,
                cutoff_date=leaky_cutoff,
            )


# ── Analog Distribution & Constraints Tests ───────────────────────────────────


class TestAnalogDistributionAndConstraints:
    """Tests for check_analog_distribution and check_basic_constraints."""

    def test_analog_distribution_passes_within_2sigma(self) -> None:
        # Analogs have multipliers centered around 1.0 with std ~0.1
        analogs = [
            {"multiplier_7": 0.95, "multiplier_14": 1.00, "multiplier_28": 1.05},
            {"multiplier_7": 0.90, "multiplier_14": 0.98, "multiplier_28": 1.02},
        ]
        # Multipliers [0.95, 1.0, 1.02, 1.05] should easily pass
        is_ok, reasons, mu, sig = check_analog_distribution([0.95, 1.0, 1.02, 1.05], analogs)
        assert is_ok is True
        assert len(reasons) == 0
        assert 0.9 <= mu <= 1.1

    def test_analog_distribution_fails_outside_2sigma(self) -> None:
        analogs = [
            {"multiplier_7": 0.95, "multiplier_14": 1.00, "multiplier_28": 1.05},
            {"multiplier_7": 0.90, "multiplier_14": 0.98, "multiplier_28": 1.02},
        ]
        # Multiplier 2.5 is far beyond 2-sigma
        is_ok, reasons, mu, sig = check_analog_distribution([2.50, 1.0, 1.0, 1.0], analogs)
        assert is_ok is False
        assert len(reasons) == 1
        assert "outside analog 2-sigma band" in reasons[0]

    def test_basic_constraints_flags_negative_values(self) -> None:
        base = np.full(28, 10.0)
        bad_proj = np.array([-1.0] + [10.0] * 27)
        is_ok, reasons = check_basic_constraints(bad_proj, base)
        assert is_ok is False
        assert any("negative values" in r for r in reasons)

    def test_basic_constraints_flags_nan_values(self) -> None:
        base = np.full(28, 10.0)
        bad_proj = np.array([np.nan] + [10.0] * 27)
        is_ok, reasons = check_basic_constraints(bad_proj, base)
        assert is_ok is False
        assert any("non-finite" in r for r in reasons)


# ── Audit Logging Tests ────────────────────────────────────────────────────────


class TestAuditLogging:
    """Tests for log_verification_event CSV writing."""

    def test_log_file_created_and_appended(self, tmp_path: Path) -> None:
        log_file = tmp_path / "verification_log.csv"
        event1 = {
            "scenario_id": "scen_1",
            "series_id": "ser_1",
            "attempt": 0,
            "event_type": "retried",
            "passed": False,
            "failure_reasons": ["Out of bounds"],
            "k_analogs": 3,
            "analog_mean": 1.0,
            "analog_std": 0.2,
            "backtest_rmse": 1.25,
            "weekly_multipliers": [1.8, 1.0, 1.0, 1.0],
            "confidence": 0.7,
        }
        log_verification_event(event1, log_path=log_file)
        assert log_file.exists()

        event2 = {
            "scenario_id": "scen_1",
            "series_id": "ser_1",
            "attempt": 1,
            "event_type": "passed",
            "passed": True,
            "failure_reasons": [],
            "k_analogs": 5,
            "analog_mean": 1.0,
            "analog_std": 0.2,
            "backtest_rmse": 1.25,
            "weekly_multipliers": [1.1, 1.0, 1.0, 1.0],
            "confidence": 0.85,
        }
        log_verification_event(event2, log_path=log_file)

        df = pd.read_csv(log_file)
        assert len(df) == 2
        assert set(LOG_COLUMNS).issubset(set(df.columns))
        assert list(df["event_type"]) == ["retried", "passed"]


# ── Verify() Workflow Tests ───────────────────────────────────────────────────


class TestVerifyWorkflow:
    """Tests for end-to-end Verify() logic."""

    @pytest.fixture
    def mock_memory(self) -> MagicMock:
        mem = MagicMock(spec=EpisodeMemory)
        mem.__len__.return_value = 5
        # Return realistic analogs with outcome multipliers near 1.0
        analogs = [
            HistoricalEpisode(
                episode_id=f"ep_{i}",
                series_id="FOODS_1_001_CA_1",
                scenario_type="price_up",
                magnitude=10.0,
                pre_event_trend=0.0,
                seasonality={"weekend_ratio": 1.1, "peak_dow": 6},
                multiplier_7=0.92,
                multiplier_14=0.95,
                multiplier_28=0.98,
                scenario_text="10% price bump",
                cutoff_date=pd.Timestamp("2011-05-01"),
                end_date=pd.Timestamp("2011-05-29"),
                similarity_score=0.75,
            )
            for i in range(5)
        ]
        mem.retrieve.return_value = analogs
        return mem

    def test_verify_passes_first_attempt(self, mock_memory: MagicMock, tmp_path: Path) -> None:
        df = _make_dummy_daily_series(n_days=120)
        cutoff = df["date"].max() + pd.Timedelta(days=1)
        base = np.full(28, 10.0)
        log_file = tmp_path / "verification_log.csv"

        # Valid reasoner result within analog 2-sigma band (~0.95)
        mock_reason_res = LLMReasonResult(
            projected_series=np.full(28, 9.5),
            weekly_multipliers=[0.95, 0.95, 0.95, 0.95],
            confidence=0.88,
            rationale="Aligned with analog contractions.",
            assumptions=["Normal elasticity"],
            latency_seconds=0.2,
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            raw_response="{}",
        )

        with patch("src.verifier.LLM_Reason", return_value=mock_reason_res):
            res = Verify(
                base_forecast=base,
                scenario_text="10% price increase",
                series_pre_cutoff=df,
                cutoff_date=cutoff,
                memory=mock_memory,
                initial_k=3,
                max_retries=2,
                log_path=log_file,
            )

        assert res.is_valid is True
        assert res.status == "passed"
        assert res.retries_used == 0
        assert len(res.projected_series) == HORIZON_DAYS

        # Check logged events: passed and final
        df_log = pd.read_csv(log_file)
        assert list(df_log["event_type"]) == ["passed", "final"]

    def test_verify_retries_and_widens_retrieval(self, mock_memory: MagicMock, tmp_path: Path) -> None:
        df = _make_dummy_daily_series(n_days=120)
        cutoff = df["date"].max() + pd.Timedelta(days=1)
        base = np.full(28, 10.0)
        log_file = tmp_path / "verification_log.csv"

        # Attempt 0 returns multiplier 2.50 (outside 2-sigma band)
        bad_reason_res = LLMReasonResult(
            projected_series=np.full(28, 25.0),
            weekly_multipliers=[2.50, 1.0, 1.0, 1.0],
            confidence=0.5,
            rationale="Unreasonable jump.",
            assumptions=[],
            latency_seconds=0.2,
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            raw_response="{}",
        )
        # Attempt 1 returns recalibrated multipliers [0.95, 0.95, 0.95, 0.95]
        good_reason_res = LLMReasonResult(
            projected_series=np.full(28, 9.5),
            weekly_multipliers=[0.95, 0.95, 0.95, 0.95],
            confidence=0.85,
            rationale="Recalibrated within analog bounds.",
            assumptions=[],
            latency_seconds=0.2,
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
            raw_response="{}",
        )

        with patch("src.verifier.LLM_Reason", side_effect=[bad_reason_res, good_reason_res]):
            res = Verify(
                base_forecast=base,
                scenario_text="10% price bump",
                series_pre_cutoff=df,
                cutoff_date=cutoff,
                memory=mock_memory,
                initial_k=3,
                max_retries=2,
                log_path=log_file,
            )

        assert res.is_valid is True
        assert res.status == "passed"
        assert res.retries_used == 1

        # Check that memory retrieval was widened on retry (initial_k=3 -> retry k=5)
        calls = mock_memory.retrieve.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["k"] == 3
        assert calls[1].kwargs["k"] == 5

        # Check logged events: retried, passed, final
        df_log = pd.read_csv(log_file)
        assert list(df_log["event_type"]) == ["retried", "passed", "final"]
