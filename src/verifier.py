"""
src/verifier.py
===============
Verification engine (Verify()) for qualitative and quantitative what-if forecasts.

Core capabilities:
1. Rolling-origin backtest error band estimation for the baseline model.
2. Analog distribution consistency check: ensures projected multipliers lie within
   2 standard deviations of pre-cutoff historical analog outcomes.
3. Basic constraint enforcement (non-negativity, finite bounds).
4. Failure recovery: widens episodic memory retrieval (larger k, relaxed outcome bounds)
   and re-prompts LLM_Reason with explicit feedback (max 2 retries).
5. Comprehensive audit trail: logs every event (passed, failed, retried, final)
   to ``results/verification_log.csv``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.baselines import forecast_b1_arima_multiplier
from src.leakage_guard import assert_no_leakage, filter_before_cutoff
from src.llm_reasoner import LLM_Reason, LLMReasonResult, multipliers_to_series
from src.memory import EpisodeMemory, HistoricalEpisode

logger = logging.getLogger(__name__)

HORIZON_DAYS = 28
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "verification_log.csv"

# Column order for verification log CSV
LOG_COLUMNS = [
    "timestamp",
    "scenario_id",
    "series_id",
    "attempt",
    "event_type",
    "passed",
    "failure_reasons",
    "k_analogs",
    "analog_mean",
    "analog_std",
    "backtest_rmse",
    "weekly_multipliers",
    "confidence",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Result Dataclass
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VerificationResult:
    """
    Output of the Verify() workflow.

    Attributes
    ----------
    is_valid:
        Whether the final projected series satisfied all verification checks.
    status:
        'passed' or 'failed'.
    projected_series:
        Final 28-day projected forecast.
    weekly_multipliers:
        Final 4-week multipliers.
    confidence:
        Confidence score from LLM_Reason.
    rationale:
        Qualitative rationale from LLM_Reason.
    backtest_rmse:
        Rolling-origin backtest RMSE of the base forecasting model.
    analog_mean:
        Mean of historical analog multipliers.
    analog_std:
        Standard deviation of historical analog multipliers.
    retries_used:
        Number of verification retries performed.
    events:
        List of audit events logged to CSV.
    """

    is_valid: bool
    status: str
    projected_series: np.ndarray
    weekly_multipliers: List[float]
    confidence: float
    rationale: str
    backtest_rmse: float
    analog_mean: float
    analog_std: float
    retries_used: int
    events: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_analogs: List[HistoricalEpisode] = field(default_factory=list)
    reasoning: Optional[LLMReasonResult] = None
    backtest_sigma: Optional[np.ndarray] = None
    passed: bool = True
    retried: bool = False
    attempts: int = 0
    failure_reasons: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Rolling-Origin Backtest Error Band
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rolling_backtest_error_band(
    series_pre_cutoff: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
    n_origins: int = 2,
    horizon_days: int = HORIZON_DAYS,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Perform rolling-origin backtesting strictly prior to cutoff_date to calculate
    the base model's typical forecast error standard deviation and error bands.

    Parameters
    ----------
    series_pre_cutoff:
        Daily series preceding cutoff_date.
    cutoff_date:
        Cutoff date for leakage assertion.
    n_origins:
        Number of rolling backtest origins to evaluate (default 2).
    horizon_days:
        Forecast horizon per origin (28 days).

    Returns
    -------
    Tuple:
        (backtest_rmse, lower_error_offset, upper_error_offset)
    """
    cutoff = pd.Timestamp(cutoff_date)
    assert_no_leakage(series_pre_cutoff, date_col="date", cutoff=cutoff)

    df = series_pre_cutoff.sort_values("date").reset_index(drop=True)
    sales = df["sales"].to_numpy(dtype=float)

    min_needed = 28 + horizon_days
    if len(sales) < min_needed:
        # Fallback for very short series: 25% of baseline mean
        mean_val = float(np.mean(sales)) if len(sales) > 0 else 1.0
        rmse = max(0.5, mean_val * 0.25)
        offset = np.full(horizon_days, 2.0 * rmse)
        return rmse, -offset, offset

    errors_list: List[float] = []

    # Slide back from the cutoff by origin steps
    for step in range(1, n_origins + 1):
        origin_idx = len(df) - (step * horizon_days)
        if origin_idx < 28:
            break

        train_slice = df.iloc[:origin_idx]
        actual_test = sales[origin_idx : origin_idx + horizon_days]

        # Forecast baseline with ARIMA
        pred = forecast_b1_arima_multiplier(
            series_pre_cutoff=train_slice,
            cutoff_date=train_slice["date"].iloc[-1] + pd.Timedelta(days=1),
            scenario_type="none",
            meta={},
            horizon_days=len(actual_test),
        )

        diff = actual_test - pred
        errors_list.extend(diff.tolist())

    if errors_list:
        rmse = float(np.sqrt(np.mean(np.array(errors_list) ** 2)))
    else:
        rmse = float(np.std(sales[-28:])) if len(sales) >= 28 else 1.0

    rmse = max(0.2, rmse)
    error_offset = np.full(horizon_days, 2.0 * rmse)
    return round(rmse, 4), -error_offset, error_offset


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Check Analog Distribution & Basic Constraints
# ═══════════════════════════════════════════════════════════════════════════════


def check_analog_distribution(
    weekly_multipliers: Sequence[float],
    analogs: Sequence[Any],
    min_sigma: float = 0.15,
) -> Tuple[bool, List[str], float, float]:
    """
    Check if projected weekly multipliers fall within 2 standard deviations of
    the retrieved analog outcome distribution.

    Parameters
    ----------
    weekly_multipliers:
        4 weekly multipliers [w1, w2, w3, w4].
    analogs:
        List of historical analog episodes.
    min_sigma:
        Standard deviation floor to prevent artificially tight bounds.

    Returns
    -------
    Tuple[bool, List[str], float, float]:
        (is_valid, failure_reasons, analog_mean, analog_std)
    """
    multipliers = [float(x) for x in weekly_multipliers]

    if not analogs:
        # No pre-cutoff analogs available: apply standard retail plausibility bounds [0.0, 2.5]
        reasons = []
        for i, w in enumerate(multipliers, start=1):
            if not (0.0 <= w <= 2.5):
                reasons.append(f"Week {i} multiplier {w:.2f} outside default plausible range [0.0, 2.5]")
        return len(reasons) == 0, reasons, 1.0, 0.5

    # Gather historical outcome multipliers across all retrieved analogs
    analog_vals: List[float] = []
    for a in analogs:
        for attr in ("multiplier_7", "multiplier_14", "multiplier_28"):
            val = getattr(a, attr, None) if hasattr(a, attr) else a.get(attr)
            if val is not None and pd.notna(val) and val >= 0:
                analog_vals.append(float(val))

    if not analog_vals:
        analog_vals = [1.0]

    mu = float(np.mean(analog_vals))
    sigma = float(np.std(analog_vals))
    sigma = max(sigma, min_sigma)

    lower_bound = max(0.0, mu - 2.0 * sigma)
    upper_bound = mu + 2.0 * sigma

    reasons: List[str] = []
    for i, w in enumerate(multipliers, start=1):
        if not (lower_bound <= w <= upper_bound):
            reasons.append(
                f"Week {i} multiplier {w:.2f} is outside analog 2-sigma band "
                f"[{lower_bound:.2f}, {upper_bound:.2f}] (mean={mu:.2f}, sigma={sigma:.2f})"
            )

    is_valid = len(reasons) == 0
    return is_valid, reasons, round(mu, 4), round(sigma, 4)


def check_basic_constraints(
    projected_series: np.ndarray,
    base_forecast: np.ndarray,
) -> Tuple[bool, List[str]]:
    """
    Verify basic physical and mathematical constraints:
    - Non-negativity
    - Finite numbers (no NaN or Inf)
    - Reasonable upper ceiling (<= 10x base mean)
    """
    reasons = []
    arr = np.asarray(projected_series, dtype=float)

    if not np.all(np.isfinite(arr)):
        reasons.append("Projected series contains non-finite (NaN or Inf) values.")

    if np.any(arr < 0.0):
        reasons.append(f"Projected series contains negative values (min: {np.min(arr)}).")

    base_mean = max(0.1, float(np.mean(base_forecast)))
    if np.any(arr > 10.0 * base_mean):
        reasons.append(f"Projected series exceeds 10x base mean ceiling (max: {np.max(arr):.1f}).")

    return len(reasons) == 0, reasons


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Audit Logging
# ═══════════════════════════════════════════════════════════════════════════════


def log_verification_event(
    event: Dict[str, Any],
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append a single verification event row to CSV."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()

    row = {
        "timestamp": event.get("timestamp", datetime.utcnow().isoformat()),
        "scenario_id": event.get("scenario_id", "unspecified"),
        "series_id": event.get("series_id", "unspecified"),
        "attempt": event.get("attempt", 0),
        "event_type": event.get("event_type", "unknown"),
        "passed": bool(event.get("passed", False)),
        "failure_reasons": "; ".join(event.get("failure_reasons", [])) if isinstance(event.get("failure_reasons"), list) else str(event.get("failure_reasons", "")),
        "k_analogs": event.get("k_analogs", 0),
        "analog_mean": event.get("analog_mean", 0.0),
        "analog_std": event.get("analog_std", 0.0),
        "backtest_rmse": event.get("backtest_rmse", 0.0),
        "weekly_multipliers": json.dumps(event.get("weekly_multipliers", [])),
        "confidence": event.get("confidence", 0.0),
    }

    df_row = pd.DataFrame([row], columns=LOG_COLUMNS)
    df_row.to_csv(log_path, mode="a", header=write_header, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main Verification Engine: Verify()
# ═══════════════════════════════════════════════════════════════════════════════


def Verify(
    base_forecast: Sequence[float] | np.ndarray,
    scenario_text: str,
    series_pre_cutoff: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
    memory: Optional[EpisodeMemory] = None,
    scenario_id: str = "scenario_0",
    series_id: str = "series_0",
    initial_k: int = 3,
    max_retries: int = 2,
    log_path: Path = DEFAULT_LOG_PATH,
    bypass_cache: bool = False,
) -> VerificationResult:
    """
    Execute the Verify() guardrail workflow.

    Steps:
    ------
    1. Compute base model rolling-origin backtest error band from pre-cutoff data.
    2. Retrieve top-k historical analogs from memory (pre-cutoff only).
    3. Call LLM_Reason.
    4. Check projection against analog 2-sigma band and basic constraints.
    5. On failure, widen retrieval (k += 2, relax window bounds) and re-prompt (max 2 retries).
    6. Log every event (passed, failed, retried, final) to results/verification_log.csv.

    Parameters
    ----------
    base_forecast:
        28-day baseline forecast.
    scenario_text:
        What-if scenario prompt.
    series_pre_cutoff:
        Historical daily sales DataFrame strictly preceding cutoff_date.
    cutoff_date:
        Cutoff date.
    memory:
        Populated EpisodeMemory object.
    scenario_id:
        Identifier for logging.
    series_id:
        Identifier for logging.
    initial_k:
        Initial analog retrieval count (default 3).
    max_retries:
        Max re-query attempts upon verification failure (default 2).
    log_path:
        Target CSV path for audit logs.
    bypass_cache:
        Force live LLM call.

    Returns
    -------
    VerificationResult with validated projection, multipliers, and audit events.
    """
    cutoff = pd.Timestamp(cutoff_date)
    base_arr = np.asarray(base_forecast, dtype=float)
    events_logged: List[Dict[str, Any]] = []

    # 1. Base model rolling-origin backtest error band
    backtest_rmse, _, _ = compute_rolling_backtest_error_band(
        series_pre_cutoff=series_pre_cutoff,
        cutoff_date=cutoff,
    )

    current_k = initial_k
    feedback_text = ""
    last_reason_result: Optional[LLMReasonResult] = None
    passed_verification = False
    failure_reasons: List[str] = []
    analog_mu = 1.0
    analog_sig = 0.5

    for attempt in range(max_retries + 1):
        # 2. Retrieve analogs (widened on retry)
        analogs: List[HistoricalEpisode] = []
        if memory is not None and len(memory) > 0:
            # On first attempt enforce full 28-day outcome completed; on retries widen window
            require_full = (attempt == 0)
            analogs = memory.retrieve(
                query_scenario=scenario_text,
                cutoff_date=cutoff,
                k=current_k,
                require_full_outcome=require_full,
            )

        # 3. Call LLM_Reason with scenario + feedback if retrying
        full_scenario_query = scenario_text
        if feedback_text:
            full_scenario_query = f"{scenario_text}\n\n[Verification Feedback]: {feedback_text}"

        reason_res = LLM_Reason(
            base_forecast=base_arr,
            scenario_text=full_scenario_query,
            retrieved_analogs=analogs,
            bypass_cache=bypass_cache if attempt == 0 else True,
        )
        last_reason_result = reason_res

        # 4. Check analog distribution (2-sigma band) and basic constraints
        dist_ok, dist_reasons, analog_mu, analog_sig = check_analog_distribution(
            reason_res.weekly_multipliers,
            analogs,
        )
        const_ok, const_reasons = check_basic_constraints(
            reason_res.projected_series,
            base_arr,
        )

        all_ok = dist_ok and const_ok
        failure_reasons = dist_reasons + const_reasons

        # 5. Determine event type and log
        if all_ok:
            event_type = "passed"
            passed_verification = True
        else:
            event_type = "retried" if attempt < max_retries else "failed"

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "scenario_id": scenario_id,
            "series_id": series_id,
            "attempt": attempt,
            "event_type": event_type,
            "passed": all_ok,
            "failure_reasons": failure_reasons,
            "k_analogs": len(analogs),
            "analog_mean": analog_mu,
            "analog_std": analog_sig,
            "backtest_rmse": backtest_rmse,
            "weekly_multipliers": reason_res.weekly_multipliers,
            "confidence": reason_res.confidence,
        }
        log_verification_event(event, log_path=log_path)
        events_logged.append(event)

        logger.info(
            "Verify() attempt %d/%d: %s | passed=%s | failure_reasons=%s",
            attempt,
            max_retries,
            event_type,
            all_ok,
            failure_reasons or "None",
        )

        if all_ok:
            break

        # Prepare for retry: widen retrieval and compose feedback
        current_k += 2
        feedback_text = (
            f"Previous multipliers {reason_res.weekly_multipliers} failed verification: "
            + "; ".join(failure_reasons)
            + f". Please calibrate multipliers within historical analog bounds [{max(0.0, analog_mu - 2*analog_sig):.2f}, {analog_mu + 2*analog_sig:.2f}]."
        )

    # Final event log
    final_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "scenario_id": scenario_id,
        "series_id": series_id,
        "attempt": attempt,
        "event_type": "final",
        "passed": passed_verification,
        "failure_reasons": failure_reasons if not passed_verification else [],
        "k_analogs": current_k,
        "analog_mean": analog_mu,
        "analog_std": analog_sig,
        "backtest_rmse": backtest_rmse,
        "weekly_multipliers": last_reason_result.weekly_multipliers if last_reason_result else [],
        "confidence": last_reason_result.confidence if last_reason_result else 0.0,
    }
    log_verification_event(final_event, log_path=log_path)
    events_logged.append(final_event)

    # If verification failed after all retries, project multipliers into valid 2-sigma bounds
    final_multipliers = list(last_reason_result.weekly_multipliers)
    if not passed_verification:
        lower_b = max(0.0, analog_mu - 2.0 * analog_sig)
        upper_b = analog_mu + 2.0 * analog_sig
        clamped_multipliers = [float(np.clip(m, lower_b, upper_b)) for m in final_multipliers]
        logger.warning(
            "Verify(): Clamping out-of-bounds multipliers from %s to %s",
            final_multipliers,
            clamped_multipliers,
        )
        final_multipliers = clamped_multipliers

    final_series = multipliers_to_series(base_arr, final_multipliers)
    backtest_sigma = np.full(len(base_arr), max(0.5, 2.0 * backtest_rmse))

    return VerificationResult(
        is_valid=passed_verification,
        status="passed" if passed_verification else "failed",
        projected_series=final_series,
        weekly_multipliers=final_multipliers,
        confidence=last_reason_result.confidence if last_reason_result else 0.0,
        rationale=last_reason_result.rationale if last_reason_result else "No result",
        backtest_rmse=backtest_rmse,
        analog_mean=analog_mu,
        analog_std=analog_sig,
        retries_used=attempt,
        events=events_logged,
        retrieved_analogs=analogs,
        reasoning=last_reason_result,
        backtest_sigma=backtest_sigma,
        passed=passed_verification,
        retried=(attempt > 0),
        attempts=attempt,
        failure_reasons=failure_reasons if not passed_verification else [],
    )


# Alias
verify = Verify
