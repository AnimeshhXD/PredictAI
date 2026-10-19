"""
tests/test_schemas.py
=====================
Tests for the Pydantic v2 data contracts in src/schemas.py.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest

from src.schemas import (
    ExperimentRecord,
    ForecastMethod,
    ForecastPoint,
    ForecastResult,
    ScenarioStatus,
    WhatIfScenario,
)


class TestWhatIfScenario:
    """Tests for :class:`WhatIfScenario`."""

    def _base(self, **kw) -> dict:
        defaults: dict = dict(
            name="Oil shock",
            cutoff_date="2023-01-01",
            target_variable="oil_price",
            horizon_days=90,
        )
        defaults.update(kw)  # caller overrides win
        return defaults

    def test_valid_scenario_creates(self) -> None:
        s = WhatIfScenario(**self._base())
        assert isinstance(s.scenario_id, UUID)
        assert s.status == ScenarioStatus.PENDING
        assert s.method == ForecastMethod.ENSEMBLE

    def test_cutoff_string_parsed(self) -> None:
        s = WhatIfScenario(**self._base(cutoff_date="2022-06-15"))
        assert isinstance(s.cutoff_date, datetime)
        assert s.cutoff_date.year == 2022

    def test_horizon_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            WhatIfScenario(**self._base(horizon_days=0))

    def test_horizon_max(self) -> None:
        WhatIfScenario(**self._base(horizon_days=3650))  # should pass
        with pytest.raises(Exception):
            WhatIfScenario(**self._base(horizon_days=3651))

    def test_name_max_length(self) -> None:
        with pytest.raises(Exception):
            WhatIfScenario(**self._base(name="x" * 129))

    def test_parameters_dict(self) -> None:
        s = WhatIfScenario(**self._base(parameters={"shock": 0.2}))
        assert s.parameters["shock"] == 0.2

    def test_default_parameters_empty(self) -> None:
        s = WhatIfScenario(**self._base())
        assert s.parameters == {}


class TestForecastPoint:
    def test_required_fields_only(self) -> None:
        p = ForecastPoint(date=datetime(2023, 3, 1), value=42.5)
        assert p.lower_80 is None
        assert p.upper_95 is None

    def test_with_intervals(self) -> None:
        p = ForecastPoint(
            date=datetime(2023, 3, 1),
            value=100.0,
            lower_80=90.0,
            upper_80=110.0,
            lower_95=85.0,
            upper_95=115.0,
        )
        assert p.lower_95 == 85.0


class TestForecastResult:
    def _scenario_id(self) -> UUID:
        return WhatIfScenario(
            name="test",
            cutoff_date="2023-01-01",
            target_variable="x",
            horizon_days=30,
        ).scenario_id

    def test_points_sorted_on_creation(self) -> None:
        sid = self._scenario_id()
        pts = [
            ForecastPoint(date=datetime(2023, 3, 3), value=3.0),
            ForecastPoint(date=datetime(2023, 3, 1), value=1.0),
            ForecastPoint(date=datetime(2023, 3, 2), value=2.0),
        ]
        result = ForecastResult(scenario_id=sid, method=ForecastMethod.ARIMA, points=pts)
        dates = [p.date for p in result.points]
        assert dates == sorted(dates)

    def test_empty_points_allowed(self) -> None:
        sid = self._scenario_id()
        result = ForecastResult(scenario_id=sid, method=ForecastMethod.LLM)
        assert result.points == []


class TestExperimentRecord:
    def _sid(self) -> UUID:
        return WhatIfScenario(
            name="t", cutoff_date="2023-01-01", target_variable="x", horizon_days=10
        ).scenario_id

    def test_valid_record(self) -> None:
        rec = ExperimentRecord(
            scenario_id=self._sid(),
            seed=42,
            method=ForecastMethod.XGBOOST,
            mae=1.5,
            rmse=2.1,
        )
        assert isinstance(rec.run_id, UUID)
        assert rec.seed == 42

    def test_optional_metrics_default_none(self) -> None:
        rec = ExperimentRecord(
            scenario_id=self._sid(),
            seed=0,
            method=ForecastMethod.LLM,
        )
        assert rec.mae is None
        assert rec.mape is None
