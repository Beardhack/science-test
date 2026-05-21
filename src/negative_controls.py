from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data_model import empty_or_blank, nested_get
from .effect_estimation import estimate_effects


def negative_control_candidates(config: Mapping[str, Any]) -> pd.DataFrame:
    configured = nested_get(config, ["research_question", "candidate_negative_control_outcomes"], [])
    rows: list[dict[str, str]] = []
    if not empty_or_blank(configured):
        for candidate in configured:
            rows.append(
                {
                    "candidate": str(candidate),
                    "type": "outcome",
                    "status": "user-specified, requires clinical review",
                    "rationale": "Assumption must be documented by clinical reviewers before calibration use.",
                    "required_assumptions": "No causal effect of the study exposure on this outcome; shared sources of confounding and measurement bias.",
                    "reasons_to_reject": "Reject if a biologic pathway, contraindication, surveillance pathway, coding artifact, or poor outcome capture is plausible.",
                }
            )
    else:
        proposals = [
            (
                "acute appendicitis hospitalization",
                "Acute appendicitis should not plausibly be prevented or caused by initiation of either diabetes drug class over one year, but it can share healthcare-seeking and coding intensity bias.",
            ),
            (
                "traumatic injury encounter",
                "Short-term traumatic injury should not be a direct pharmacologic effect for this comparison; utilization and frailty may still create residual association.",
            ),
            (
                "cataract procedure",
                "Cataract procedure timing is not expected to be an acute causal consequence of initiation, but access-to-care and health-system factors can bias it.",
            ),
        ]
        for name, rationale in proposals:
            rows.append(
                {
                    "candidate": name,
                    "type": "outcome",
                    "status": "proposed, requires clinical review",
                    "rationale": rationale,
                    "required_assumptions": "No causal exposure effect, no contraindication-mediated pathway, comparable capture, and shared residual confounding structure.",
                    "reasons_to_reject": "Reject if clinical review identifies a plausible drug effect, prescribing-channel link, differential capture, or weak overlap with primary-outcome bias sources.",
                }
            )
    return pd.DataFrame(rows)


def run_negative_control_outcome_analysis(
    df: pd.DataFrame,
    weight_col: str = "weight_stabilized_iptw",
    threshold_abs_log_rr: float = 0.10,
) -> pd.DataFrame:
    if "negative_control_event_365" not in df.columns:
        return pd.DataFrame(
            [
                {
                    "negative_control": "none_available",
                    "analysis": "not_run",
                    "residual_association_detected": True,
                    "reason": "No negative-control outcome column was available.",
                }
            ]
        )
    crude = estimate_effects(df, outcome_col="negative_control_event_365", weight_col=None, label="negative_control_crude")
    adjusted = estimate_effects(df, outcome_col="negative_control_event_365", weight_col=weight_col, label="negative_control_adjusted")
    rows = []
    for row in [crude, adjusted]:
        log_rr = float(np.log(max(float(row["risk_ratio"]), 1e-9)))
        rows.append(
            {
                "negative_control": "synthetic negative-control outcome",
                "analysis": row["analysis"],
                "risk_exposed": row["risk_exposed"],
                "risk_comparator": row["risk_comparator"],
                "risk_ratio": row["risk_ratio"],
                "log_risk_ratio": log_rr,
                "residual_association_detected": abs(log_rr) > threshold_abs_log_rr,
                "interpretation": (
                    "Residual association exceeds the configured threshold and suggests possible bias."
                    if abs(log_rr) > threshold_abs_log_rr
                    else "Near-null residual association under this synthetic negative control."
                ),
            }
        )
    return pd.DataFrame(rows)


def empirical_calibration_stub(results: pd.DataFrame) -> pd.DataFrame:
    valid = results.loc[results["analysis"].astype(str).str.contains("adjusted", na=False)].copy()
    if len(valid) < 3:
        return pd.DataFrame(
            [
                {
                    "calibration_status": "not_performed",
                    "reason": "Empirical calibration requires multiple clinically valid negative controls; this demo has fewer than three.",
                }
            ]
        )
    valid["mean_null_log_rr"] = valid["log_risk_ratio"].mean()
    valid["sd_null_log_rr"] = valid["log_risk_ratio"].std(ddof=1)
    return valid[["calibration_status", "mean_null_log_rr", "sd_null_log_rr"]]
