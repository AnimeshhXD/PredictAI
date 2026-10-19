"""
tests/test_evaluator.py
=======================
Unit and integration tests for src/evaluator.py:
- Metrics calculation (MAE, RMSE, WAPE, Acc@10, Acc@15, Acc@20, change accuracy)
- Bootstrap confidence intervals
- Paired Wilcoxon signed-rank tests
- Summary table generation (Table 1, Table 2, Table 3)
- Journal-style figure generation (Figures 1-4 in PNG and PDF)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluator import (
    bootstrap_ci,
    compute_metrics,
    estimate_cost,
    generate_journal_figures,
    generate_summary_tables,
    paired_wilcoxon_test,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Calculation Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_compute_metrics_exact_match() -> None:
    """When prediction perfectly matches actual, error is 0 and accuracy is 100%."""
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([10.0, 20.0, 30.0, 40.0])

    m = compute_metrics(actual, pred)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["wape"] == 0.0
    assert m["acc_10"] == 1.0
    assert m["acc_15"] == 1.0
    assert m["acc_20"] == 1.0


def test_compute_metrics_known_values() -> None:
    """Validate arithmetic against known hand calculations."""
    actual = np.array([100.0, 100.0])
    pred = np.array([110.0, 110.0])  # +10% error
    # sum actual = 200, sum pred = 220, abs error = [10, 10]
    # mae = 10, rmse = 10, wape = 20/200 * 100 = 10%
    # total_rel_err = 20/200 = 0.10 -> acc_10 = 1.0, acc_15 = 1.0, acc_20 = 1.0

    m = compute_metrics(actual, pred)
    assert m["mae"] == 10.0
    assert m["rmse"] == 10.0
    assert m["wape"] == 10.0
    assert m["acc_10"] == 1.0
    assert m["acc_15"] == 1.0
    assert m["acc_20"] == 1.0


def test_compute_metrics_accuracy_thresholds() -> None:
    """Test boundary conditions for 10%, 15%, 20% scenario-response accuracy."""
    actual = np.array([100.0])

    # 12% error: passes 15% and 20%, fails 10%
    m12 = compute_metrics(actual, np.array([112.0]))
    assert m12["acc_10"] == 0.0
    assert m12["acc_15"] == 1.0
    assert m12["acc_20"] == 1.0

    # 18% error: passes 20%, fails 10% and 15%
    m18 = compute_metrics(actual, np.array([118.0]))
    assert m18["acc_10"] == 0.0
    assert m18["acc_15"] == 0.0
    assert m18["acc_20"] == 1.0

    # 25% error: fails all three
    m25 = compute_metrics(actual, np.array([125.0]))
    assert m25["acc_10"] == 0.0
    assert m25["acc_15"] == 0.0
    assert m25["acc_20"] == 0.0


def test_compute_metrics_shape_mismatch() -> None:
    """Mismatched array lengths should raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_compute_metrics_with_base_forecast() -> None:
    """Verify change accuracy relative to unadjusted baseline."""
    base = np.array([100.0, 100.0])    # sum = 200
    actual = np.array([110.0, 110.0])  # sum = 220, actual_change = +20
    pred = np.array([111.0, 111.0])    # sum = 222, pred_change = +22
    # change_rel_err = |22 - 20| / 20 = 2 / 20 = 0.10 <= 0.15 -> change_acc_15 = 1.0

    m = compute_metrics(actual, pred, base_forecast=base)
    assert m["change_acc_15"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_bootstrap_ci_bounds() -> None:
    """Bootstrap CI should yield valid lower <= mean <= upper bounds."""
    data = [12.0, 14.5, 11.2, 16.0, 13.8, 15.1, 12.9, 14.0, 13.5, 14.2]
    mean_val, lower, upper = bootstrap_ci(data, n_boot=500, seed=42)

    assert lower <= mean_val <= upper
    assert 12.0 <= mean_val <= 15.0


def test_bootstrap_ci_constant_array() -> None:
    """Bootstrap CI on constant array returns the identical constant."""
    data = [5.0] * 10
    mean_val, lower, upper = bootstrap_ci(data)
    assert mean_val == 5.0
    assert lower == 5.0
    assert upper == 5.0


def test_paired_wilcoxon_test_significant_difference() -> None:
    """When treatment is systematically lower than control, test should detect difference."""
    treatment = [5.0, 6.0, 5.5, 4.8, 5.2, 5.9, 6.1, 5.0, 5.3, 5.7]
    control = [15.0, 16.0, 15.5, 14.8, 15.2, 15.9, 16.1, 15.0, 15.3, 15.7]

    stat, p_val = paired_wilcoxon_test(treatment, control)
    assert p_val < 0.05


def test_paired_wilcoxon_test_identical() -> None:
    """When treatment and control are identical, returns p=1.0."""
    treatment = [5.0] * 10
    control = [5.0] * 10

    _, p_val = paired_wilcoxon_test(treatment, control)
    assert p_val == 1.0


def test_estimate_cost() -> None:
    """Validate cost computation formula."""
    cost = estimate_cost(prompt_tokens=1000, completion_tokens=500)
    # (1000/1e6 * 0.15) + (500/1e6 * 0.60) = 0.00015 + 0.00030 = 0.00045
    assert cost == pytest.approx(0.00045, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# Table and Figure Generation Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_eval_df() -> pd.DataFrame:
    """Generate a representative evaluation DataFrame for testing report generators."""
    records = []
    methods = [
        "b1_arima",
        "b2_arimax",
        "b3_xgboost",
        "predictai_full",
        "ablation_no_retrieval",
        "ablation_no_verification",
        "ablation_no_llm",
    ]
    scenarios = ["price_up", "demand_surge", "supply_delay"]

    rng = np.random.default_rng(42)
    for ep_idx in range(1, 16):
        stype = scenarios[ep_idx % 3]
        for m in methods:
            base_wape = {
                "b1_arima": 28.0,
                "b2_arimax": 26.0,
                "b3_xgboost": 23.0,
                "ablation_no_llm": 30.0,
                "ablation_no_retrieval": 21.0,
                "ablation_no_verification": 20.0,
                "predictai_full": 16.0,
            }[m]
            wape = max(5.0, base_wape + rng.normal(0, 2.0))
            mae = wape * 0.1

            records.append({
                "episode_idx": ep_idx,
                "series_id": f"FOODS_1_{ep_idx:03d}_CA_1",
                "scenario_type": stype,
                "cutoff_date": "2013-05-01",
                "method": m,
                "mae": round(mae, 2),
                "rmse": round(mae * 1.2, 2),
                "wape": round(wape, 2),
                "acc_10": 1.0 if wape < 12.0 else 0.0,
                "acc_15": 1.0 if wape < 18.0 else 0.0,
                "acc_20": 1.0 if wape < 24.0 else 0.0,
                "change_acc_15": 1.0 if wape < 18.0 else 0.0,
                "latency_seconds": 0.25 if "predictai" in m else 0.05,
                "prompt_tokens": 250 if "predictai" in m else 0,
                "completion_tokens": 50 if "predictai" in m else 0,
                "cost_usd": 0.0001 if "predictai" in m else 0.0,
            })

    return pd.DataFrame(records)


def test_generate_summary_tables(sample_eval_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify Table 1, Table 2, and Table 3 CSV outputs exist and have correct columns."""
    tables = generate_summary_tables(sample_eval_df, output_dir=tmp_path)

    assert "table1" in tables
    assert "table2" in tables
    assert "table3" in tables

    # Table 1: Main Results
    t1_file = tmp_path / "table1_main_results.csv"
    assert t1_file.exists()
    t1_loaded = pd.read_csv(t1_file)
    assert "Method" in t1_loaded.columns
    assert "WAPE (%)" in t1_loaded.columns
    assert "Acc@15%" in t1_loaded.columns
    assert "Wilcoxon p-val" in t1_loaded.columns
    assert len(t1_loaded) == 4

    # Table 2: Ablation Study
    t2_file = tmp_path / "table2_ablation_study.csv"
    assert t2_file.exists()
    t2_loaded = pd.read_csv(t2_file)
    assert "Variant" in t2_loaded.columns
    assert "Wilcoxon p vs Full" in t2_loaded.columns
    assert len(t2_loaded) == 4

    # Table 3: Scenario Breakdown
    t3_file = tmp_path / "table3_by_scenario_type.csv"
    assert t3_file.exists()
    t3_loaded = pd.read_csv(t3_file)
    assert "Scenario Type" in t3_loaded.columns
    assert len(t3_loaded) == 9  # 3 scenarios x 3 methods


def test_generate_journal_figures(sample_eval_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify all 4 academic figures are rendered in both PNG and PDF format."""
    figures_dir = tmp_path / "figures"
    generate_journal_figures(sample_eval_df, figures_dir=figures_dir)

    expected_files = [
        "figure1_model_comparison_wape.png",
        "figure1_model_comparison_wape.pdf",
        "figure2_scenario_response_accuracy_sensitivity.png",
        "figure2_scenario_response_accuracy_sensitivity.pdf",
        "figure3_ablation_breakdown.png",
        "figure3_ablation_breakdown.pdf",
        "figure4_trajectory_examples.png",
        "figure4_trajectory_examples.pdf",
    ]

    for fname in expected_files:
        p = figures_dir / fname
        assert p.exists(), f"Figure file missing: {p}"
        assert p.stat().st_size > 1000, f"Figure file suspiciously small: {p}"
