from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .data_model import comparator_label, exposure_label, nested_get


def positivity_table(diagnostics_by_model: dict[str, dict[str, float | bool]]) -> pd.DataFrame:
    rows = []
    for model, diag in diagnostics_by_model.items():
        row = {"model": model}
        row.update(diag)
        rows.append(row)
    return pd.DataFrame(rows)


def design_alternatives_table(config: Mapping[str, Any]) -> pd.DataFrame:
    exposure = exposure_label(config)
    comparator = comparator_label(config)
    outcome = nested_get(config, ["research_question", "outcome", "name"], "outcome")
    rows = [
        {
            "design": "self-controlled case series",
            "compatible": "usually no",
            "reason": f"{exposure} vs {comparator} initiation is a chronic treatment choice, and {outcome} can alter future exposure and observation.",
        },
        {
            "design": "case-crossover",
            "compatible": "usually no",
            "reason": "Best for transient exposures with acute effects; this question concerns sustained initiation strategies.",
        },
        {
            "design": "case-time-control",
            "compatible": "limited",
            "reason": "Could address exposure time trends for acute windows, but does not naturally target the 365-day new-user estimand.",
        },
        {
            "design": "prior event rate ratio adjustment",
            "compatible": "exploratory",
            "reason": "May help with time-stable frailty/utilization differences, but incident-outcome exclusion and treatment switching complicate interpretation.",
        },
        {
            "design": "difference-in-differences",
            "compatible": "only with policy/formulary shock",
            "reason": "Requires a defensible intervention time and parallel-trends evidence.",
        },
        {
            "design": "within-provider or within-site design",
            "compatible": "possibly",
            "reason": "Can reduce site/provider confounding if enough overlap remains and provider preference is not a direct pathway to outcomes.",
        },
    ]
    return pd.DataFrame(rows)


def credibility_grade(
    fragility_flags_by_model: dict[str, dict[str, bool]],
    negative_control_results: pd.DataFrame,
    qba_summary: Mapping[str, object],
    simulation_summary: pd.DataFrame,
) -> tuple[str, str]:
    any_design_failure = any(any(flags.values()) for flags in fragility_flags_by_model.values())
    nc_problem = False
    if not negative_control_results.empty and "residual_association_detected" in negative_control_results:
        adjusted = negative_control_results[negative_control_results["analysis"].astype(str).str.contains("adjusted", na=False)]
        nc_problem = bool(adjusted["residual_association_detected"].any()) if not adjusted.empty else True
    tipping_easy = qba_summary.get("tipping_status") == "found_on_grid"
    false_reassurance = 0.0
    if not simulation_summary.empty and "false_reassurance_rate" in simulation_summary:
        false_reassurance = float(simulation_summary["false_reassurance_rate"].max())

    if any_design_failure and nc_problem:
        return "Low", "Weighted diagnostics and negative controls both suggest vulnerability to residual confounding."
    if any_design_failure:
        return "Moderate", "At least one balance, positivity, or effective-sample-size concern remains."
    if nc_problem:
        return "Low", "Negative-control residual association suggests the design may still carry bias."
    if tipping_easy or false_reassurance > 0.25:
        return "Moderate", "Sensitivity or simulation results leave a material residual-confounding concern."
    return "High", "Design diagnostics are acceptable in this synthetic demo; identification still depends on untestable assumptions."
