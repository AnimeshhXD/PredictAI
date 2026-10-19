"""
src/llm_reasoner.py
===================
Qualitative scenario reasoning engine (LLM_Reason).

Synthesizes:
1. Quantitative base forecast (28-day trajectory)
2. What-if scenario text (plain-language intervention)
3. Retrieved historical analogs from episodic memory (with outcome multipliers)

Outputs strict JSON validated via Pydantic:
- weekly_multipliers: [w1, w2, w3, w4]
- confidence: float between 0.0 and 1.0
- rationale: qualitative explanation
- assumptions: list of underlying scenario assumptions

Retries on invalid JSON or schema errors, converts weekly multipliers into a
projected 28-day series, and logs latency and token usage per call.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.llm import call_llm_with_metadata

logger = logging.getLogger(__name__)

HORIZON_DAYS = 28
WEEKS = 4
DAYS_PER_WEEK = 7


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Schema for Strict Output Validation
# ═══════════════════════════════════════════════════════════════════════════════


class LLMReasoningOutput(BaseModel):
    """
    Pydantic contract for LLM reasoning output.

    Attributes
    ----------
    weekly_multipliers:
        Demand multipliers for Week 1 (days 1-7), Week 2 (days 8-14),
        Week 3 (days 15-21), and Week 4 (days 22-28). Exactly 4 non-negative floats.
    confidence:
        Epistemic confidence score from 0.0 (uncertain) to 1.0 (high confidence).
    rationale:
        Qualitative explanation of the economic mechanism and analog comparison.
    assumptions:
        List of key assumptions supporting the projected multipliers.
    """

    weekly_multipliers: List[float] = Field(
        ...,
        description="Projected demand multipliers for Week 1, Week 2, Week 3, and Week 4. Exactly 4 floats >= 0.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    rationale: str = Field(
        ...,
        description="Qualitative causal rationale connecting scenario shock to demand response.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="List of key behavioral or market assumptions.",
    )

    @field_validator("weekly_multipliers")
    @classmethod
    def validate_multipliers(cls, v: List[float]) -> List[float]:
        if len(v) != WEEKS:
            raise ValueError(f"Expected exactly {WEEKS} weekly multipliers, got {len(v)}")
        for i, val in enumerate(v, start=1):
            if val < 0.0:
                raise ValueError(f"Multiplier for Week {i} cannot be negative: {val}")
        return [float(x) for x in v]


# ═══════════════════════════════════════════════════════════════════════════════
# Result Container
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LLMReasonResult:
    """
    Complete output of an LLM_Reason call.

    Attributes
    ----------
    projected_series:
        28-day forecast series formed by scaling base_forecast with weekly multipliers.
    weekly_multipliers:
        The 4 weekly multiplier values [w1, w2, w3, w4].
    confidence:
        Confidence score between 0.0 and 1.0.
    rationale:
        Qualitative reasoning provided by the LLM.
    assumptions:
        List of underlying assumptions.
    latency_seconds:
        Total wall-clock time spent in the LLM call.
    prompt_tokens:
        Estimated or measured prompt tokens.
    completion_tokens:
        Estimated or measured completion tokens.
    total_tokens:
        Total tokens (prompt + completion).
    raw_response:
        Raw text response returned by the model.
    retries_used:
        Number of retry attempts required to obtain valid JSON.
    """

    projected_series: np.ndarray
    weekly_multipliers: List[float]
    confidence: float
    rationale: str
    assumptions: List[str]
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: str
    retries_used: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════════


def _format_analogs_for_prompt(analogs: Sequence[Any]) -> str:
    """Format retrieved analog episodes into a concise Markdown section."""
    if not analogs:
        return "None available prior to cutoff date."

    lines = []
    for i, a in enumerate(analogs, start=1):
        # Support both HistoricalEpisode objects and dictionaries
        if hasattr(a, "scenario_type"):
            stype = a.scenario_type
            mag = a.magnitude
            trend = a.pre_event_trend
            m7 = a.multiplier_7
            m14 = a.multiplier_14
            m28 = a.multiplier_28
            desc = a.scenario_text
            sim = getattr(a, "similarity_score", None)
        else:
            stype = a.get("scenario_type", "unknown")
            mag = a.get("magnitude", 0.0)
            trend = a.get("pre_event_trend", 0.0)
            m7 = a.get("multiplier_7", 1.0)
            m14 = a.get("multiplier_14", 1.0)
            m28 = a.get("multiplier_28", 1.0)
            desc = a.get("scenario_text", "")
            sim = a.get("similarity_score")

        sim_str = f" (similarity: {sim:.2f})" if sim is not None else ""
        lines.append(
            f"Analog #{i}{sim_str}:\n"
            f"  - Type: {stype} (Magnitude: {mag})\n"
            f"  - Pre-event trend slope: {trend:+.4f}\n"
            f"  - Realized outcome multipliers: +7d={m7:.2f}, +14d={m14:.2f}, +28d={m28:.2f}\n"
            f"  - Description: {desc[:200]}..."
        )
    return "\n\n".join(lines)


def _build_prompt(
    base_forecast: np.ndarray,
    scenario_text: str,
    retrieved_analogs: Sequence[Any],
) -> str:
    """Construct the system prompt and instructions for the reasoning engine."""
    base_mean = float(np.mean(base_forecast))
    w1_mean = float(np.mean(base_forecast[0:7])) if len(base_forecast) >= 7 else base_mean
    w2_mean = float(np.mean(base_forecast[7:14])) if len(base_forecast) >= 14 else base_mean
    w3_mean = float(np.mean(base_forecast[14:21])) if len(base_forecast) >= 21 else base_mean
    w4_mean = float(np.mean(base_forecast[21:28])) if len(base_forecast) >= 28 else base_mean

    analogs_str = _format_analogs_for_prompt(retrieved_analogs)

    prompt = f"""You are an expert economic reasoning engine for retail time-series what-if simulation.
Your task is to analyze a hypothetical scenario and historical analog episodes to project demand adjustment multipliers over a 4-week (28-day) horizon.

### Baseline Forecast (Statistical Pre-Intervention Projection)
- 28-day baseline mean: {base_mean:.2f} units/day
- Week 1 baseline mean: {w1_mean:.2f} units/day
- Week 2 baseline mean: {w2_mean:.2f} units/day
- Week 3 baseline mean: {w3_mean:.2f} units/day
- Week 4 baseline mean: {w4_mean:.2f} units/day

### What-If Scenario Description
{scenario_text.strip()}

### Historical Analog Episodes (Observed Pre-Cutoff)
{analogs_str}

### Instructions
1. Analyze the scenario type (e.g. price increase, demand shock, supply delay/stockout).
2. Compare with historical analogs to understand typical elasticity, recovery trajectories, and persistence.
3. Determine 4 weekly multipliers:
   - Week 1 (Days 1–7): immediate shock response
   - Week 2 (Days 8–14): intermediate response or stockout continuation
   - Week 3 (Days 15–21): initial recovery / decay towards baseline
   - Week 4 (Days 22–28): long-term stabilization level
4. Provide an overall confidence score between 0.0 and 1.0.
5. Provide a concise qualitative rationale and 2–3 key assumptions.

### Required JSON Output Format
You MUST respond with valid JSON ONLY (no markdown text, no surrounding explanations):
{{
  "weekly_multipliers": [1.0, 1.0, 1.0, 1.0],
  "confidence": 0.85,
  "rationale": "Clear causal explanation referencing analogs and scenario mechanics.",
  "assumptions": [
    "Assumption 1...",
    "Assumption 2..."
  ]
}}
"""
    return prompt.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Extraction Helper
# ═══════════════════════════════════════════════════════════════════════════════


def _clean_json_response(raw_text: str) -> str:
    """Extract clean JSON substring from model response (stripping markdown fences)."""
    text = raw_text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Match first '{' to last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Multipliers -> 28-Day Forecast Conversion
# ═══════════════════════════════════════════════════════════════════════════════


def multipliers_to_series(
    base_forecast: np.ndarray,
    weekly_multipliers: Sequence[float],
) -> np.ndarray:
    """
    Expand 4 weekly multipliers into a 28-day daily forecast series.

    Parameters
    ----------
    base_forecast:
        Array of 28 baseline predictions.
    weekly_multipliers:
        Array or list of 4 weekly multipliers [w1, w2, w3, w4].

    Returns
    -------
    np.ndarray of length 28 (clipped at 0.0).
    """
    base = np.asarray(base_forecast, dtype=float)
    if len(base) != HORIZON_DAYS:
        raise ValueError(f"base_forecast must have length {HORIZON_DAYS}, got {len(base)}")

    if len(weekly_multipliers) != WEEKS:
        raise ValueError(f"weekly_multipliers must have length {WEEKS}, got {len(weekly_multipliers)}")

    # 7 days per week
    daily_mults = np.repeat(np.asarray(weekly_multipliers, dtype=float), DAYS_PER_WEEK)
    projected = base * daily_mults
    return np.maximum(projected, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Function: LLM_Reason
# ═══════════════════════════════════════════════════════════════════════════════


def LLM_Reason(
    base_forecast: Union[Sequence[float], np.ndarray],
    scenario_text: str,
    retrieved_analogs: Sequence[Any] = (),
    *,
    max_retries: int = 3,
    bypass_cache: bool = False,
) -> LLMReasonResult:
    """
    Perform qualitative scenario reasoning via LLM with strict JSON validation.

    Parameters
    ----------
    base_forecast:
        Array of 28 baseline daily demand forecasts.
    scenario_text:
        Plain-language description of the what-if scenario.
    retrieved_analogs:
        List of historical analog episodes retrieved from episodic memory.
    max_retries:
        Maximum JSON re-prompt attempts if the model produces invalid output.
    bypass_cache:
        If True, force a fresh API call instead of reading from disk cache.

    Returns
    -------
    LLMReasonResult containing projected_series (28 days), weekly_multipliers,
    confidence, rationale, assumptions, latency_seconds, and token counts.
    """
    base_arr = np.asarray(base_forecast, dtype=float)
    if len(base_arr) != HORIZON_DAYS:
        raise ValueError(f"base_forecast must have exactly {HORIZON_DAYS} days, got {len(base_arr)}")

    initial_prompt = _build_prompt(base_arr, scenario_text, retrieved_analogs)
    current_prompt = initial_prompt

    total_latency = 0.0
    total_prompt_tokens = 0
    total_comp_tokens = 0
    raw_response_last = ""
    validated_output: Optional[LLMReasoningOutput] = None

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        raw_text, meta = call_llm_with_metadata(
            prompt=current_prompt,
            bypass_cache=bypass_cache if attempt == 0 else True,
        )
        call_elapsed = float(meta.get("latency_seconds", time.perf_counter() - t0))
        total_latency += call_elapsed
        total_prompt_tokens += meta.get("prompt_tokens", 0)
        total_comp_tokens += meta.get("completion_tokens", 0)
        raw_response_last = raw_text

        try:
            clean_text = _clean_json_response(raw_text)
            data = json.loads(clean_text)
            validated_output = LLMReasoningOutput.model_validate(data)
            logger.info(
                "LLM_Reason parsed valid JSON on attempt %d in %.2fs | "
                "tokens: prompt=%d, comp=%d, total=%d | confidence=%.2f",
                attempt + 1,
                call_elapsed,
                meta.get("prompt_tokens", 0),
                meta.get("completion_tokens", 0),
                meta.get("total_tokens", 0),
                validated_output.confidence,
            )
            break
        except Exception as err:
            logger.warning(
                "LLM_Reason validation failed on attempt %d/%d: %s. Re-prompting...",
                attempt + 1,
                max_retries,
                err,
            )
            current_prompt = (
                f"{initial_prompt}\n\n"
                f"### ERROR ON PREVIOUS ATTEMPT\n"
                f"Your previous output could not be parsed: {err}\n"
                f"Previous raw output was:\n{raw_text[:300]}\n\n"
                f"Please fix the error and output STRICTLY VALID JSON ONLY."
            )

    # Fallback if all retries failed
    if validated_output is None:
        logger.error("LLM_Reason: All %d attempts failed. Applying default unity multipliers.", max_retries)
        validated_output = LLMReasoningOutput(
            weekly_multipliers=[1.0, 1.0, 1.0, 1.0],
            confidence=0.0,
            rationale="Fallback: LLM failed to return valid JSON within retry budget.",
            assumptions=["Model output parsing failed; assumed zero departure from baseline."],
        )

    # Convert multipliers into 28-day projected series
    projected_series = multipliers_to_series(base_arr, validated_output.weekly_multipliers)

    return LLMReasonResult(
        projected_series=projected_series,
        weekly_multipliers=validated_output.weekly_multipliers,
        confidence=validated_output.confidence,
        rationale=validated_output.rationale,
        assumptions=validated_output.assumptions,
        latency_seconds=round(total_latency, 4),
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_comp_tokens,
        total_tokens=total_prompt_tokens + total_comp_tokens,
        raw_response=raw_response_last,
        retries_used=attempt,
    )


# Alias
llm_reason = LLM_Reason
