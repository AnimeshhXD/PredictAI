"""
tests/test_app.py
=================
Smoke and integration tests for app.py Streamlit UI logic and Plotly visualization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import load_datasets, load_episodic_memory, plot_simulation


def test_plot_simulation_traces() -> None:
    """Verify that plot_simulation returns a Plotly figure with all 4 required layers."""
    dates = pd.date_range("2013-01-01", periods=60, freq="D")
    history_df = pd.DataFrame({
        "date": dates[:32],
        "sales": np.random.uniform(5, 15, size=32),
    })

    cutoff_date = dates[32]
    base_forecast = np.full(28, 10.0)
    scenario_forecast = np.full(28, 12.0)
    confidence_sigma = np.full(28, 1.5)
    actual_future = np.full(28, 11.5)

    fig = plot_simulation(
        history_df=history_df,
        cutoff_date=cutoff_date,
        base_forecast=base_forecast,
        scenario_forecast=scenario_forecast,
        confidence_sigma=confidence_sigma,
        actual_future=actual_future,
        series_id="FOODS_1_001_CA_1",
        scenario_title="Demand Surge",
    )

    trace_names = [trace.name for trace in fig.data]
    assert "Pre-Cutoff History" in trace_names
    assert "Verify() 2σ Band" in trace_names
    assert "Base Forecast (ARIMA)" in trace_names
    assert "PredictAI Scenario Projection" in trace_names
    assert "Post-Cutoff Actuals (Ground Truth)" in trace_names


def test_load_datasets_and_memory() -> None:
    """Verify cached data loading returns non-empty dataframes and initialized memory."""
    daily, episodes_df = load_datasets()
    assert not daily.empty
    assert "series_id" in daily.columns
    assert "sales" in daily.columns

    memory = load_episodic_memory()
    assert len(memory) > 0
