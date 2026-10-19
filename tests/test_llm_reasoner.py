"""
tests/test_llm_reasoner.py
==========================
Tests for src/llm_reasoner.py (LLM_Reason).

Verifies:
1. Pydantic validation of LLMReasoningOutput schema (multipliers, confidence, rationale).
2. Multiplier-to-series 28-day projection.
3. Successful end-to-end LLM_Reason call with latency and token logging.
4. Retry logic upon receiving invalid JSON.
5. Graceful fallback when all retry attempts fail.
"""

from __future__ import annotations

import json
from unittest.mock import patch
import numpy as np
import pytest
from pydantic import ValidationError

from src.llm_reasoner import (
    HORIZON_DAYS,
    LLM_Reason,
    LLMReasoningOutput,
    _clean_json_response,
    multipliers_to_series,
)
from src.memory import HistoricalEpisode
import pandas as pd


# ── Schema Tests ───────────────────────────────────────────────────────────────


class TestLLMReasoningOutputSchema:
    """Tests for Pydantic schema validation."""

    def test_valid_schema_instantiation(self) -> None:
        data = {
            "weekly_multipliers": [0.90, 0.92, 0.95, 0.98],
            "confidence": 0.85,
            "rationale": "Price elasticity causes immediate 10% drop, gradually attenuating.",
            "assumptions": ["No promotion", "Inelastic demand"],
        }
        out = LLMReasoningOutput.model_validate(data)
        assert len(out.weekly_multipliers) == 4
        assert out.confidence == 0.85
        assert len(out.assumptions) == 2

    def test_wrong_multipliers_length_raises(self) -> None:
        data = {
            "weekly_multipliers": [1.0, 1.0, 1.0],  # only 3 weeks!
            "confidence": 0.8,
            "rationale": "test",
        }
        with pytest.raises(ValidationError):
            LLMReasoningOutput.model_validate(data)

    def test_negative_multiplier_raises(self) -> None:
        data = {
            "weekly_multipliers": [-0.5, 1.0, 1.0, 1.0],
            "confidence": 0.8,
            "rationale": "test",
        }
        with pytest.raises(ValidationError):
            LLMReasoningOutput.model_validate(data)

    def test_confidence_out_of_bounds_raises(self) -> None:
        data = {
            "weekly_multipliers": [1.0, 1.0, 1.0, 1.0],
            "confidence": 1.5,  # > 1.0!
            "rationale": "test",
        }
        with pytest.raises(ValidationError):
            LLMReasoningOutput.model_validate(data)


# ── Multiplier to Series Conversion Tests ──────────────────────────────────────


class TestMultipliersToSeries:
    """Tests for multipliers_to_series expansion."""

    def test_conversion_pattern(self) -> None:
        base = np.full(28, 10.0)
        multipliers = [0.9, 1.0, 1.1, 1.2]
        series = multipliers_to_series(base, multipliers)

        assert len(series) == HORIZON_DAYS
        # Week 1: days 0..6
        np.testing.assert_allclose(series[0:7], 9.0)
        # Week 2: days 7..13
        np.testing.assert_allclose(series[7:14], 10.0)
        # Week 3: days 14..20
        np.testing.assert_allclose(series[14:21], 11.0)
        # Week 4: days 21..27
        np.testing.assert_allclose(series[21:28], 12.0)

    def test_length_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="base_forecast must have length 28"):
            multipliers_to_series(np.zeros(20), [1.0, 1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="weekly_multipliers must have length 4"):
            multipliers_to_series(np.zeros(28), [1.0, 1.0])


# ── LLM_Reason Execution & Retry Tests ─────────────────────────────────────────


class TestLLMReasonExecution:
    """Tests for LLM_Reason execution with mocked model responses."""

    @pytest.fixture
    def sample_analog(self) -> HistoricalEpisode:
        return HistoricalEpisode(
            episode_id="ep_sample",
            series_id="FOODS_1_001_CA_1",
            scenario_type="price_up",
            magnitude=10.0,
            pre_event_trend=-0.01,
            seasonality={"weekend_ratio": 1.1, "peak_dow": 6},
            multiplier_7=0.88,
            multiplier_14=0.91,
            multiplier_28=0.93,
            scenario_text="10% price increase on food item",
            cutoff_date=pd.Timestamp("2012-01-01"),
            end_date=pd.Timestamp("2012-01-29"),
            similarity_score=0.82,
        )

    def test_successful_llm_reason(self, sample_analog: HistoricalEpisode) -> None:
        base = np.full(28, 10.0)
        valid_json = json.dumps({
            "weekly_multipliers": [0.88, 0.90, 0.92, 0.95],
            "confidence": 0.80,
            "rationale": "Analog evidence shows 12% initial contraction followed by partial recovery.",
            "assumptions": ["Competitors do not match price increase"],
        })

        mock_meta = {
            "latency_seconds": 0.25,
            "prompt_tokens": 180,
            "completion_tokens": 45,
            "total_tokens": 225,
            "cached": False,
        }

        with patch("src.llm_reasoner.call_llm_with_metadata", return_value=(valid_json, mock_meta)):
            res = LLM_Reason(
                base_forecast=base,
                scenario_text="What-if scenario: price increases by 10%.",
                retrieved_analogs=[sample_analog],
            )

        assert len(res.projected_series) == HORIZON_DAYS
        assert res.weekly_multipliers == [0.88, 0.90, 0.92, 0.95]
        assert res.confidence == 0.80
        assert res.retries_used == 0
        assert res.latency_seconds > 0.0
        assert res.prompt_tokens == 180
        assert res.completion_tokens == 45
        assert res.total_tokens == 225
        np.testing.assert_allclose(res.projected_series[0:7], 8.8)

    def test_retry_on_invalid_json(self) -> None:
        base = np.full(28, 10.0)
        invalid_first_response = "Here is my reasoning: {\"weekly_multipliers\": [0.8"  # Broken JSON
        valid_second_response = json.dumps({
            "weekly_multipliers": [0.85, 0.88, 0.90, 0.92],
            "confidence": 0.75,
            "rationale": "Recovered valid json after retry.",
            "assumptions": ["Standard retail elasticity"],
        })

        mock_meta = {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}

        with patch(
            "src.llm_reasoner.call_llm_with_metadata",
            side_effect=[
                (invalid_first_response, mock_meta),
                (valid_second_response, mock_meta),
            ],
        ):
            res = LLM_Reason(
                base_forecast=base,
                scenario_text="10% price bump",
                max_retries=3,
            )

        assert res.retries_used == 1
        assert res.confidence == 0.75
        assert res.weekly_multipliers == [0.85, 0.88, 0.90, 0.92]

    def test_fallback_when_retries_exhausted(self) -> None:
        base = np.full(28, 10.0)
        bad_response = "I cannot fulfill this in JSON format."
        mock_meta = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}

        with patch(
            "src.llm_reasoner.call_llm_with_metadata",
            return_value=(bad_response, mock_meta),
        ):
            res = LLM_Reason(
                base_forecast=base,
                scenario_text="10% price bump",
                max_retries=2,
            )

        # Fallback to unity multipliers with 0.0 confidence
        assert res.confidence == 0.0
        assert res.weekly_multipliers == [1.0, 1.0, 1.0, 1.0]
        np.testing.assert_allclose(res.projected_series, base)
        assert "Fallback" in res.rationale

    def test_clean_json_strips_markdown_fences(self) -> None:
        raw = "```json\n{\n  \"weekly_multipliers\": [1, 1, 1, 1],\n  \"confidence\": 0.9\n}\n```"
        cleaned = _clean_json_response(raw)
        assert cleaned.startswith("{") and cleaned.endswith("}")
