"""
src/schemas.py
==============
Pydantic v2 models for PredictAI's core data contracts.

These models are the single source of truth for:
* What constitutes a "what-if scenario"
* What a forecast result looks like
* What an experiment run record contains

All timestamps are stored as UTC-aware :class:`datetime` objects.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────


class ForecastMethod(str, Enum):
    """Supported forecasting back-ends."""

    ARIMA = "arima"
    XGBOOST = "xgboost"
    LLM = "llm"
    ENSEMBLE = "ensemble"


class ScenarioStatus(str, Enum):
    """Lifecycle state of a scenario run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


# ── Scenario ───────────────────────────────────────────────────────────────────


class WhatIfScenario(BaseModel):
    """
    A single what-if scenario submitted by the user.

    Attributes
    ----------
    scenario_id:
        Auto-generated UUID.
    name:
        Short human-readable label (≤ 128 chars).
    description:
        Free-text description of the hypothetical intervention.
    cutoff_date:
        The date *before* which all training/retrieval data must fall.
        Strictly enforced by the leakage guard.
    target_variable:
        Name of the time-series column to forecast.
    horizon_days:
        Number of calendar days to forecast beyond the cutoff.
    parameters:
        Arbitrary key→value dict for scenario-specific knobs
        (e.g. {"shock_pct": 0.15, "sector": "energy"}).
    method:
        Which forecasting back-end to use.
    created_at:
        UTC timestamp of scenario creation.
    """

    scenario_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., max_length=128)
    description: str = Field(default="")
    cutoff_date: datetime = Field(
        ...,
        description="Exclusive upper bound for training data (UTC).",
    )
    target_variable: str = Field(...)
    horizon_days: int = Field(..., gt=0, le=3650)
    parameters: dict[str, Any] = Field(default_factory=dict)
    method: ForecastMethod = Field(default=ForecastMethod.ENSEMBLE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ScenarioStatus = Field(default=ScenarioStatus.PENDING)

    @field_validator("cutoff_date", mode="before")
    @classmethod
    def parse_cutoff(cls, v: Any) -> datetime:
        """Accept str, date, or datetime; return a naive UTC datetime."""
        import pandas as pd  # local import to keep module lightweight

        ts = pd.Timestamp(v)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.to_pydatetime()

    model_config = {"populate_by_name": True}


# ── Forecast result ────────────────────────────────────────────────────────────


class ForecastPoint(BaseModel):
    """
    A single point in a forecast trajectory.

    Attributes
    ----------
    date:
        Calendar date of this forecast point.
    value:
        Predicted value of the target variable.
    lower_80:
        80 % prediction-interval lower bound (optional).
    upper_80:
        80 % prediction-interval upper bound (optional).
    lower_95:
        95 % prediction-interval lower bound (optional).
    upper_95:
        95 % prediction-interval upper bound (optional).
    """

    date: datetime
    value: float
    lower_80: Optional[float] = None
    upper_80: Optional[float] = None
    lower_95: Optional[float] = None
    upper_95: Optional[float] = None


class ForecastResult(BaseModel):
    """
    Full output of a forecasting run for one scenario.

    Attributes
    ----------
    scenario_id:
        Links back to the originating :class:`WhatIfScenario`.
    method:
        Which method produced this forecast.
    points:
        Ordered list of :class:`ForecastPoint` objects.
    metadata:
        Arbitrary dict for storing method-specific diagnostics
        (AIC, RMSE on validation split, feature importances, …).
    generated_at:
        UTC timestamp of result generation.
    """

    scenario_id: UUID
    method: ForecastMethod
    points: list[ForecastPoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def points_are_sorted(self) -> "ForecastResult":
        """Ensure forecast points are in ascending date order."""
        if self.points:
            self.points = sorted(self.points, key=lambda p: p.date)
        return self


# ── Experiment run record ──────────────────────────────────────────────────────


class ExperimentRecord(BaseModel):
    """
    Metadata written to results/ for every experiment run.

    This is what gets serialised to CSV / JSON and loaded for table/figure
    generation — never hardcode numbers, always read from these records.

    Attributes
    ----------
    run_id:
        Auto-generated UUID for this specific run.
    scenario_id:
        UUID of the scenario that was evaluated.
    seed:
        Random seed used for reproducibility.
    method:
        Forecasting method used.
    mae:
        Mean Absolute Error on the held-out test window.
    rmse:
        Root Mean Squared Error on the held-out test window.
    mape:
        Mean Absolute Percentage Error (%).
    llm_provider:
        LLM provider used (from env config).
    llm_model:
        LLM model name used.
    elapsed_seconds:
        Wall-clock time for the full run.
    notes:
        Optional free-text notes.
    run_at:
        UTC timestamp of the run.
    """

    run_id: UUID = Field(default_factory=uuid4)
    scenario_id: UUID
    seed: int
    method: ForecastMethod
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    notes: str = Field(default="")
    run_at: datetime = Field(default_factory=datetime.utcnow)
