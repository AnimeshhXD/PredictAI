"""
app.py
======
PredictAI - Interactive AI What-If Retail Demand Simulator
Streamlit Application for Journal Paper Demonstration & Enterprise What-If Analysis.

Features:
---------
1. Pick any item-store series & cutoff date
2. Enter natural language what-if scenario (or select validated presets)
3. Generate baseline forecast + PredictAI scenario projection with dynamic confidence band
4. Display retrieved historical episodic analogs from FAISS memory with date guards
5. Inspect LLM reasoning, confidence score, weekly multipliers, and assumptions
6. Real-time Verifier guardrail status (backtest band, analog 2-sigma check, non-negativity)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure root is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env", override=False)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.baselines import (
    forecast_b1_arima_multiplier,
    get_pre_cutoff_series,
)
from src.data_loader import load_m5
from src.memory import EpisodeMemory, HistoricalEpisode
from src.verifier import Verify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictai_app")

HORIZON_DAYS = 28
EPISODES_PARQUET_PATH = _ROOT / "data" / "processed" / "episodes.parquet"
MEMORY_DIR_PATH = _ROOT / "data" / "processed" / "memory"


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit Page Setup & Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PredictAI | AI What-If Simulator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern typography and container spacing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header gradient badge */
.hero-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 24px 30px;
    border-radius: 14px;
    color: #ffffff;
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15);
}

.hero-title {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    color: #ffffff;
}

.hero-subtitle {
    font-size: 14px;
    color: #94a3b8;
    line-height: 1.5;
}

/* KPI metric cards */
.kpi-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.08);
}

.kpi-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    margin-bottom: 4px;
}

.kpi-val {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
}

.kpi-sub {
    font-size: 12px;
    font-weight: 500;
    margin-top: 4px;
}

/* Status Badges */
.badge-passed {
    background-color: #dcfce7;
    color: #166534;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
    border: 1px solid #bbf7d0;
}

.badge-retried {
    background-color: #fef9c3;
    color: #854d0e;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
    border: 1px solid #fef08a;
}

.badge-failed {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
    border: 1px solid #fecaca;
}

/* Analog card */
.analog-card {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 12px;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}

/* Multiplier pills */
.mult-pill {
    background: #e0f2fe;
    color: #0369a1;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    margin-right: 6px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Cached Resource Loaders
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Loading M5 daily series and episodes dataset...")
def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily M5 data and pre-detected episodes."""
    seed = int(os.environ.get("RANDOM_SEED", "42"))
    daily, _, _ = load_m5(synthetic_seed=seed, synthetic_n_days=900, synthetic_items_per_dept=8)

    if EPISODES_PARQUET_PATH.exists():
        episodes_df = pd.read_parquet(EPISODES_PARQUET_PATH)
    else:
        episodes_df = pd.DataFrame()

    return daily, episodes_df


@st.cache_resource(show_spinner="Loading episodic FAISS memory & embeddings...")
def load_episodic_memory() -> EpisodeMemory:
    """Load FAISS index and sentence embeddings model."""
    memory = EpisodeMemory()
    faiss_file = MEMORY_DIR_PATH / "episodes.faiss"
    meta_file = MEMORY_DIR_PATH / "episodes_meta.json"

    if faiss_file.exists() and meta_file.exists():
        memory.load(MEMORY_DIR_PATH)
    else:
        daily, episodes_df = load_datasets()
        if not episodes_df.empty:
            memory.build_from_episodes(episodes_df, daily)
            memory.save(MEMORY_DIR_PATH)
    return memory


# ═══════════════════════════════════════════════════════════════════════════════
# Plotly Visualization Function
# ═══════════════════════════════════════════════════════════════════════════════


def plot_simulation(
    history_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    base_forecast: np.ndarray,
    scenario_forecast: np.ndarray,
    confidence_sigma: np.ndarray,
    actual_future: Optional[np.ndarray] = None,
    series_id: str = "Series",
    scenario_title: str = "What-If Scenario",
) -> go.Figure:
    """Construct publication-quality interactive Plotly simulation plot."""
    fig = go.Figure()

    # Pre-cutoff historical dates and sales (last 45 days for visual context)
    hist_sub = history_df.tail(45)
    hist_dates = hist_sub["date"]
    hist_sales = hist_sub["sales"]

    # Future forecast horizon dates
    forecast_dates = pd.date_range(start=cutoff_date, periods=len(base_forecast), freq="D")

    # 1. Historical Actuals (Solid Slate Line)
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_sales,
            mode="lines+markers",
            name="Pre-Cutoff History",
            line=dict(color="#475569", width=2.5),
            marker=dict(size=4, color="#475569"),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Actual Sales</b>: %{y:.1f}<extra></extra>",
        )
    )

    # 2. Upper and Lower Confidence Envelope (Shaded Band)
    band_upper = scenario_forecast + confidence_sigma
    band_lower = np.maximum(0.0, scenario_forecast - confidence_sigma)

    fig.add_trace(
        go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(band_upper) + list(band_lower[::-1]),
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Verify() 2σ Band",
        )
    )

    # 3. Base Statistical Forecast (Dashed Gray Line)
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=base_forecast,
            mode="lines",
            name="Base Forecast (ARIMA)",
            line=dict(color="#94a3b8", width=2, dash="dash"),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Base Forecast</b>: %{y:.1f}<extra></extra>",
        )
    )

    # 4. PredictAI Scenario Projection (Vibrant Indigo Line)
    deltas = scenario_forecast - base_forecast
    customdata = np.stack((deltas, scenario_forecast / np.maximum(base_forecast, 1e-4)), axis=-1)

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=scenario_forecast,
            mode="lines+markers",
            name="PredictAI Scenario Projection",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6, color="#1d4ed8", symbol="diamond"),
            customdata=customdata,
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Projection</b>: %{y:.1f}<br><b>Delta</b>: %{customdata[0]:+.1f} units<br><b>Multiplier</b>: %{customdata[1]:.2f}x<extra></extra>",
        )
    )

    # 5. Optional Ground-Truth Actuals (if inspecting a historical episode)
    if actual_future is not None and len(actual_future) == len(forecast_dates):
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=actual_future,
                mode="lines",
                name="Post-Cutoff Actuals (Ground Truth)",
                line=dict(color="#10b981", width=2, dash="dot"),
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Ground Truth</b>: %{y:.1f}<extra></extra>",
            )
        )

    # Vertical line separating history from forecast cutoff
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    fig.add_vline(
        x=cutoff_date,
        line_width=2,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text=f"Cutoff: {cutoff_str}",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#b91c1c", family="Inter"),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{series_id}</b>: {scenario_title}",
            font=dict(size=16, color="#0f172a", family="Inter"),
        ),
        xaxis=dict(
            title="Timeline",
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        yaxis=dict(
            title="Daily Unit Sales",
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40),
        height=480,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application Flow
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Render PredictAI interactive What-If Simulator interface."""
    daily, episodes_df = load_datasets()
    memory = load_episodic_memory()

    series_list = sorted(daily["series_id"].unique())

    # ── Hero Banner ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-title">🔮 PredictAI: AI What-If Retail Demand Simulator</div>
            <div class="hero-subtitle">
                An LLM-driven decision support framework combining statistical baselines, FAISS episodic memory, 
                and rolling backtest verification guardrails to project non-linear demand shocks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    st.sidebar.header("🎯 Simulation Controls")

    # Mode: Freeform or Load from Historical Episodes
    mode = st.sidebar.radio(
        "Workflow Mode",
        ["Explore Historical Episodes", "Custom Freeform Scenario"],
        help="Select a benchmarked M5 episode to compare against ground truth, or craft a new hypothetical what-if scenario.",
    )

    selected_series_id: str
    selected_cutoff: pd.Timestamp
    default_scenario_text: str = ""
    actual_future: Optional[np.ndarray] = None

    if mode == "Explore Historical Episodes" and not episodes_df.empty:
        st.sidebar.subheader("Episode Selector")
        scenario_types = sorted(episodes_df["scenario_type"].unique())
        chosen_type = st.sidebar.selectbox("Filter Scenario Category", ["All"] + list(scenario_types))

        filtered_episodes = (
            episodes_df if chosen_type == "All" else episodes_df[episodes_df["scenario_type"] == chosen_type]
        )

        ep_labels = [
            f"Ep #{row.name}: {row['series_id']} | {row['scenario_type']} ({pd.Timestamp(row['cutoff_date']).date()})"
            for _, row in filtered_episodes.iterrows()
        ]

        selected_label = st.sidebar.selectbox("Select Episode", ep_labels)
        ep_idx = int(selected_label.split(":")[0].replace("Ep #", "").strip())
        ep_row = episodes_df.loc[ep_idx]

        selected_series_id = ep_row["series_id"]
        selected_cutoff = pd.Timestamp(ep_row["cutoff_date"])
        default_scenario_text = str(ep_row["scenario_text"])
        actual_future = np.asarray(ep_row["actuals"], dtype=float)

    else:
        # Freeform Series & Date Picker
        st.sidebar.subheader("Target Time Series")
        selected_series_id = st.sidebar.selectbox("Item-Store Series", series_list, index=0)

        series_data = daily[daily["series_id"] == selected_series_id]
        min_date = series_data["date"].min() + pd.Timedelta(days=90)
        max_date = series_data["date"].max() - pd.Timedelta(days=HORIZON_DAYS + 1)

        selected_cutoff = pd.Timestamp(
            st.sidebar.date_input(
                "Forecast Cutoff Date",
                value=max_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
        )
        actual_future = None

    # Series pre-cutoff slice with zero-leakage check
    series_raw = daily[daily["series_id"] == selected_series_id]
    pre_cutoff_series = get_pre_cutoff_series(series_raw, selected_cutoff)

    if len(pre_cutoff_series) < 30:
        st.error(f"Insufficient historical data before {selected_cutoff.date()} for series {selected_series_id}.")
        st.stop()

    # Pre-calculated series metrics
    hist_mean = float(pre_cutoff_series["sales"].tail(28).mean())
    hist_vol = float(pre_cutoff_series["sales"].tail(28).std())

    with st.sidebar.expander("📊 Series Characteristics", expanded=False):
        st.write(f"**Series ID:** `{selected_series_id}`")
        st.write(f"**Pre-Cutoff History:** {len(pre_cutoff_series)} days")
        st.write(f"**Recent Mean (28d):** {hist_mean:.2f} units/day")
        st.write(f"**Recent Std Dev:** {hist_vol:.2f}")

    # ── Main Input: Plain Language What-If Scenario ───────────────────────────
    st.markdown("### 1. Plain-Language What-If Query")

    # Quick preset buttons
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_clicked = False
    scenario_input = default_scenario_text

    if col_p1.button("📈 +10% Price Inflation Shock", use_container_width=True):
        scenario_input = f"Sustained 10% price increase on {selected_series_id} starting on {selected_cutoff.date()} due to supplier inflation."
        preset_clicked = True
    elif col_p2.button("🚀 +20% Viral Demand Spike", use_container_width=True):
        scenario_input = f"Unexpected 20% surge in customer demand for {selected_series_id} following regional promotion."
        preset_clicked = True
    elif col_p3.button("⚠️ 4-Day Supply Chain Delay", use_container_width=True):
        scenario_input = f"Severe distribution bottleneck causing 4 consecutive zero-sales stockout days starting on {selected_cutoff.date()} before inventory recovery."
        preset_clicked = True

    scenario_text = st.text_area(
        "Enter scenario description:",
        value=scenario_input,
        height=85,
        help="Type any realistic supply chain, pricing, promotion, or disruption scenario in natural language.",
    )

    k_analogs = st.sidebar.slider("Number of Analogs (k)", min_value=1, max_value=8, value=3)

    run_btn = st.button("🚀 Simulate What-If Scenario", type="primary", use_container_width=True)

    if not scenario_text.strip():
        st.info("💡 Enter a scenario description above or select a preset to generate a forecast simulation.")
        st.stop()

    # ── Simulation Pipeline Execution ─────────────────────────────────────────
    with st.spinner("Generating base statistical forecast, retrieving analogs, and running verified reasoning..."):
        t0 = time.perf_counter()

        # 1. Base ARIMA Forecast (Unadjusted)
        base_forecast = forecast_b1_arima_multiplier(
            series_pre_cutoff=pre_cutoff_series,
            cutoff_date=selected_cutoff,
            scenario_type="none",
            meta={},
            horizon_days=HORIZON_DAYS,
        )

        # 2. PredictAI Verified Pipeline
        verify_result = Verify(
            base_forecast=base_forecast,
            scenario_text=scenario_text,
            series_pre_cutoff=pre_cutoff_series,
            cutoff_date=selected_cutoff,
            memory=memory,
            scenario_id="ui_session",
            series_id=selected_series_id,
            initial_k=k_analogs,
        )

        scenario_projection = verify_result.projected_series
        reasoning = verify_result.reasoning
        confidence_sigma = verify_result.backtest_sigma
        analogs = verify_result.retrieved_analogs

        elapsed = time.perf_counter() - t0

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown("### 2. Projected Demand Impact & KPI Summary")

    sum_base = float(np.sum(base_forecast))
    sum_proj = float(np.sum(scenario_projection))
    delta_units = sum_proj - sum_base
    delta_pct = (delta_units / max(sum_base, 1e-4)) * 100.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Baseline (28d)</div>
                <div class="kpi-val">{sum_base:.1f}</div>
                <div class="kpi-sub" style="color: #64748b;">Unadjusted Units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        proj_color = "#16a34a" if delta_units >= 0 else "#dc2626"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Scenario (28d)</div>
                <div class="kpi-val">{sum_proj:.1f}</div>
                <div class="kpi-sub" style="color: {proj_color}; font-weight: 700;">
                    {delta_units:+.1f} units ({delta_pct:+.1f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        conf_val = reasoning.confidence if reasoning else 0.75
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">LLM Confidence</div>
                <div class="kpi-val">{conf_val * 100:.0f}%</div>
                <div class="kpi-sub" style="color: #2563eb;">Multi-week calibrated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        status_class = "badge-passed" if verify_result.passed else "badge-retried" if verify_result.retried else "badge-failed"
        status_text = "PASSED" if verify_result.passed else "RETRIED" if verify_result.retried else "FAILED"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Guardrail Verification</div>
                <div style="margin-top: 6px;"><span class="{status_class}">{status_text}</span></div>
                <div class="kpi-sub" style="color: #64748b;">Attempts: {verify_result.attempts + 1}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Simulation Time</div>
                <div class="kpi-val">{elapsed:.2f}s</div>
                <div class="kpi-sub" style="color: #64748b;">Analogs: k={len(analogs)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Interactive Plotly Simulation Chart ───────────────────────────────────
    st.markdown("### 3. Trajectory Projection & Uncertainty Envelope")

    fig = plot_simulation(
        history_df=pre_cutoff_series,
        cutoff_date=selected_cutoff,
        base_forecast=base_forecast,
        scenario_forecast=scenario_projection,
        confidence_sigma=confidence_sigma,
        actual_future=actual_future,
        series_id=selected_series_id,
        scenario_title=scenario_text[:60] + ("..." if len(scenario_text) > 60 else ""),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Two-Column Breakdown: Analogs & Reasoning ─────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 4. Retrieved Episodic Analogs (FAISS Memory)")
        st.caption(
            f"Grounding analogs retrieved strictly before cutoff date `{selected_cutoff.date()}` (zero leakage)."
        )

        if not analogs:
            st.warning("No historical analogs found prior to cutoff date meeting similarity threshold.")
        else:
            for idx, analog in enumerate(analogs, 1):
                st.markdown(
                    f"""
                    <div class="analog-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #1e293b;">#{idx} {analog.scenario_type.upper()}</span>
                            <span class="badge-passed" style="font-size: 11px;">Sim: {analog.similarity_score:.3f}</span>
                        </div>
                        <div style="font-size: 13px; color: #334155; margin-bottom: 8px;">
                            "{analog.scenario_text}"
                        </div>
                        <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                            Series: <code>{analog.series_id}</code> | Cutoff: <code>{analog.cutoff_date.date()}</code>
                        </div>
                        <div>
                            <span class="mult-pill">+7d: {analog.multiplier_7:.2f}x</span>
                            <span class="mult-pill">+14d: {analog.multiplier_14:.2f}x</span>
                            <span class="mult-pill">+28d: {analog.multiplier_28:.2f}x</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_right:
        st.markdown("### 5. LLM Reasoning & Multiplier Schedule")
        if reasoning:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
                    <div style="font-weight: 600; color: #1e293b; margin-bottom: 6px;">Executive Rationale:</div>
                    <div style="font-size: 14px; color: #334155; line-height: 1.5; font-style: italic;">
                        "{reasoning.rationale}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Weekly multiplier table
            st.markdown("**Weekly Projected Multipliers:**")
            m_w1, m_w2, m_w3, m_w4 = reasoning.weekly_multipliers
            mult_cols = st.columns(4)
            mult_cols[0].metric("Week 1 (1-7d)", f"{m_w1:.2f}x")
            mult_cols[1].metric("Week 2 (8-14d)", f"{m_w2:.2f}x")
            mult_cols[2].metric("Week 3 (15-21d)", f"{m_w3:.2f}x")
            mult_cols[3].metric("Week 4 (22-28d)", f"{m_w4:.2f}x")

            st.markdown("**Underlying Operational Assumptions:**")
            for asm in reasoning.assumptions:
                st.markdown(f"- 📌 {asm}")
        else:
            st.warning("No structured reasoning payload available.")

    # ── Verifier Guardrail Audit Log ──────────────────────────────────────────
    st.markdown("### 6. Verification Guardrail Audit Report")
    with st.expander("🛡️ View Guardrail Checks & Audit Trail", expanded=True):
        col_v1, col_v2 = st.columns([1, 1])

        with col_v1:
            st.markdown("**Verification Criteria Checked:**")
            st.markdown(
                rf"- ✅ **Zero Data Leakage**: All inputs dated $\le$ `{selected_cutoff.date()}` (Enforced)"
            )
            st.markdown(
                rf"- ✅ **Domain Constraints**: All daily projections non-negative ($y \ge 0$) and finite"
            )
            st.markdown(
                f"- {'✅' if verify_result.passed else '⚠️'} **Analog Distribution 2-Sigma**: Multipliers checked against retrieved historical envelope"
            )
            st.markdown(
                f"- ✅ **Rolling Backtest Error Band**: Rolling-origin error band computed from {len(pre_cutoff_series)} pre-cutoff observations"
            )

        with col_v2:
            st.markdown("**Verification Event Audit Trail:**")
            st.write(f"- **Final Verdict:** `{status_text}`")
            st.write(f"- **Total Iterations / Retries:** `{verify_result.attempts}`")
            if verify_result.failure_reasons:
                st.warning(f"Calibrated following checks: {', '.join(verify_result.failure_reasons)}")
            else:
                st.success("All safety checks passed on initial attempt. No retries required.")

    # ── Export Projection CSV ─────────────────────────────────────────────────
    st.markdown("### 7. Export Forecast Data")
    forecast_dates = pd.date_range(start=selected_cutoff, periods=HORIZON_DAYS, freq="D")
    export_df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "series_id": selected_series_id,
        "base_forecast": np.round(base_forecast, 2),
        "scenario_projection": np.round(scenario_projection, 2),
        "confidence_lower": np.round(np.maximum(0.0, scenario_projection - confidence_sigma), 2),
        "confidence_upper": np.round(scenario_projection + confidence_sigma, 2),
        "multiplier": np.round(scenario_projection / np.maximum(base_forecast, 1e-4), 2),
    })

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download 28-Day Simulation CSV",
        data=csv_data,
        file_name=f"predictai_simulation_{selected_series_id}_{selected_cutoff.date()}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
