from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cohort import assert_valid_cohort_timing, attrition_markdown, build_new_user_active_comparator_cohort
from .covariates import (
    assert_no_post_index_covariates,
    baseline_covariate_columns,
    baseline_table,
    missingness_table,
    table_to_markdown,
)
from .data_model import (
    analysis_config,
    comparator_label,
    exposure_label,
    load_config,
    nested_get,
    project_paths,
    rq,
    write_markdown,
)
from .diagnostics import credibility_grade, design_alternatives_table, positivity_table
from .effect_estimation import bootstrap_effects, estimate_effects, summarize_with_bootstrap
from .hdps import append_hdps_to_cohort, assert_hdps_pre_index, build_hdps_features, hdps_warning
from .instrumental_variables import screen_iv_candidates
from .negative_controls import negative_control_candidates, run_negative_control_outcome_analysis
from .outcomes import outcome_definition_markdown, validate_outcome_columns
from .propensity import balance_diagnostics, estimate_propensity_scores, fragility_flags
from .quantitative_bias_analysis import run_qba, scenario_templates
from .simulation_study import run_simulation_study
from .simple_plots import save_barh, save_histogram_overlay, save_love_plot
from .site import build_gh_pages_site
from .synthetic_data import generate_synthetic_data
from .validation_calibration import run_validation_substudy_example


def run_analysis(config_path: str | Path = "config/research_question.yaml", reports_dir: str | Path = "reports") -> dict[str, Any]:
    config = load_config(config_path)
    paths = project_paths(config_path, reports_dir)
    analysis = analysis_config(config)
    seed = int(analysis.get("random_seed", 20260520))
    thresholds = dict(analysis.get("diagnostic_thresholds", {}))
    ps_cfg = dict(analysis.get("propensity", {}))
    bootstrap_iterations = int(analysis.get("bootstrap_iterations", 0))

    synthetic = generate_synthetic_data(config, seed=seed)
    cohort, attrition = build_new_user_active_comparator_cohort(synthetic.tables, config)
    assert_valid_cohort_timing(cohort)
    validate_outcome_columns(cohort, config)

    measured_covars = baseline_covariate_columns(cohort, include_proxy=False)
    proxy_covars = baseline_covariate_columns(cohort, include_proxy=True)
    assert_no_post_index_covariates(cohort, measured_covars)
    assert_no_post_index_covariates(cohort, proxy_covars)

    baseline = baseline_table(cohort, measured_covars)
    missing = missingness_table(cohort, measured_covars + proxy_covars)
    attrition.to_csv(paths.tables_dir / "attrition.csv", index=False)
    baseline.to_csv(paths.tables_dir / "baseline_table_unweighted.csv", index=False)
    missing.to_csv(paths.tables_dir / "missingness.csv", index=False)

    conventional = estimate_propensity_scores(
        cohort,
        measured_covars,
        ridge_penalty=float(ps_cfg.get("ridge_penalty", 0.1)),
        max_iterations=int(ps_cfg.get("max_iterations", 80)),
        trim_lower=float(ps_cfg.get("ps_trim_lower", 0.025)),
        trim_upper=float(ps_cfg.get("ps_trim_upper", 0.975)),
        truncation_quantiles=tuple(ps_cfg.get("weight_truncation_quantiles", [0.01, 0.99])),
    )
    conventional.balance.to_csv(paths.tables_dir / "balance_conventional_ps.csv", index=False)

    hdps_features, hdps_ranking = build_hdps_features(
        cohort,
        synthetic.tables["code_history"],
        top_k=int(analysis.get("hdps_top_k", 30)),
    )
    assert_hdps_pre_index(hdps_ranking, cohort)
    hdps_ranking.to_csv(paths.tables_dir / "hdps_selected_proxies.csv", index=False)
    hdps_cohort = append_hdps_to_cohort(cohort, hdps_features)
    hdps_covars = proxy_covars + hdps_ranking["feature"].tolist()
    assert_no_post_index_covariates(hdps_cohort, hdps_covars)
    hdps_result = estimate_propensity_scores(
        hdps_cohort,
        hdps_covars,
        ridge_penalty=float(ps_cfg.get("ridge_penalty", 0.1)),
        max_iterations=int(ps_cfg.get("max_iterations", 80)),
        trim_lower=float(ps_cfg.get("ps_trim_lower", 0.025)),
        trim_upper=float(ps_cfg.get("ps_trim_upper", 0.975)),
        truncation_quantiles=tuple(ps_cfg.get("weight_truncation_quantiles", [0.01, 0.99])),
    )
    hdps_result.balance.to_csv(paths.tables_dir / "balance_hdps_proxy_ps.csv", index=False)

    oracle_covars = proxy_covars + ["latent_u"]
    oracle_result = estimate_propensity_scores(hdps_cohort, oracle_covars)

    effects = _estimate_main_effects(
        cohort=cohort,
        conventional=conventional.data,
        hdps=hdps_result.data,
        oracle=oracle_result.data,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    effects.to_csv(paths.tables_dir / "effect_estimates.csv", index=False)

    positivity = positivity_table({"conventional_ps": conventional.diagnostics, "hdps_proxy_ps": hdps_result.diagnostics})
    positivity.to_csv(paths.tables_dir / "positivity_diagnostics.csv", index=False)
    flags = {
        "conventional_ps": fragility_flags(conventional.diagnostics, conventional.balance, thresholds),
        "hdps_proxy_ps": fragility_flags(hdps_result.diagnostics, hdps_result.balance, thresholds),
    }
    pd.DataFrame([{"model": k, **v} for k, v in flags.items()]).to_csv(paths.tables_dir / "analysis_fragility_flags.csv", index=False)

    negative_candidates = negative_control_candidates(config)
    negative_candidates.to_csv(paths.tables_dir / "negative_control_candidates.csv", index=False)
    nc_results = run_negative_control_outcome_analysis(
        conventional.data,
        weight_col="weight_stabilized_iptw",
        threshold_abs_log_rr=float(thresholds.get("negative_control_abs_log_rr", 0.10)),
    )
    nc_results.to_csv(paths.tables_dir / "negative_control_results.csv", index=False)

    primary_rr = float(effects.loc[effects["analysis"].eq("conventional_ps_stabilized_iptw"), "risk_ratio"].iloc[0])
    qba_results, qba_summary = run_qba(
        observed_rr=primary_rr,
        config=config,
        output_dir=paths.tables_dir,
        clinically_unimportant=(
            float(thresholds.get("clinically_unimportant_rr_low", 0.95)),
            float(thresholds.get("clinically_unimportant_rr_high", 1.05)),
        ),
    )
    scenario_templates(config).to_csv(paths.tables_dir / "qba_scenario_templates.csv", index=False)

    validation = run_validation_substudy_example(cohort, measured_covars, seed=seed)
    validation.to_csv(paths.tables_dir / "validation_calibration_examples.csv", index=False)
    iv_screen = screen_iv_candidates(cohort, measured_covars)
    iv_screen.to_csv(paths.tables_dir / "iv_candidate_screening.csv", index=False)
    alternatives = design_alternatives_table(config)
    alternatives.to_csv(paths.tables_dir / "design_alternatives.csv", index=False)

    simulation = run_simulation_study(config, seed=seed + 17)
    simulation.to_csv(paths.tables_dir / "simulation_summary.csv", index=False)
    grade, grade_reason = credibility_grade(flags, nc_results, qba_summary, simulation)

    _plot_ps_overlap(conventional.data, paths.figures_dir / "propensity_overlap.png")
    _plot_balance(conventional.balance, hdps_result.balance, paths.figures_dir / "balance_smd.png")
    _plot_effects(effects, paths.figures_dir / "effect_estimates.png")

    _write_static_reports(
        config=config,
        paths=paths,
        attrition=attrition,
        baseline=baseline,
        missing=missing,
        effects=effects,
        positivity=positivity,
        hdps_ranking=hdps_ranking,
        nc_candidates=negative_candidates,
        nc_results=nc_results,
        qba_summary=qba_summary,
        validation=validation,
        iv_screen=iv_screen,
        alternatives=alternatives,
        simulation=simulation,
        grade=grade,
        grade_reason=grade_reason,
        synthetic_truth=synthetic.truth,
        flags=flags,
    )
    build_gh_pages_site(config, reports_dir=paths.reports_dir, docs_dir=paths.root / "docs")
    return {
        "config": config,
        "cohort": cohort,
        "effects": effects,
        "flags": flags,
        "credibility_grade": grade,
        "reports_dir": paths.reports_dir,
    }


def _estimate_main_effects(
    cohort: pd.DataFrame,
    conventional: pd.DataFrame,
    hdps: pd.DataFrame,
    oracle: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    specs = [
        ("crude", cohort, None),
        ("conventional_ps_stabilized_iptw", conventional, "weight_stabilized_iptw"),
        ("conventional_ps_overlap", conventional, "weight_overlap"),
        ("conventional_ps_truncated_stabilized_iptw", conventional, "weight_truncated_stabilized_iptw"),
        ("hdps_proxy_stabilized_iptw", hdps, "weight_stabilized_iptw"),
        ("hdps_proxy_overlap", hdps, "weight_overlap"),
        ("oracle_adjusts_true_u", oracle, "weight_stabilized_iptw"),
    ]
    rows = []
    for i, (label, df, weight_col) in enumerate(specs):
        point = estimate_effects(df, weight_col=weight_col, label=label)
        boot_n = bootstrap_iterations if label in {"conventional_ps_stabilized_iptw", "hdps_proxy_stabilized_iptw"} else 0
        boot = bootstrap_effects(df, weight_col=weight_col, label=label, iterations=boot_n, seed=seed + i)
        rows.append(summarize_with_bootstrap(point, boot))
    return pd.DataFrame(rows)


def _plot_ps_overlap(df: pd.DataFrame, path: Path) -> None:
    save_histogram_overlay(
        path,
        df.loc[df["A"].eq(1), "propensity_score"],
        df.loc[df["A"].eq(0), "propensity_score"],
        title="Propensity score overlap",
        label_a="Exposure",
        label_b="Comparator",
    )


def _plot_balance(conventional: pd.DataFrame, hdps: pd.DataFrame, path: Path) -> None:
    top = conventional.head(18)[["variable", "abs_smd"]].rename(columns={"abs_smd": "conventional"})
    compare = top.merge(hdps[["variable", "abs_smd"]].rename(columns={"abs_smd": "hdps"}), on="variable", how="left")
    save_love_plot(
        path,
        labels=compare["variable"].astype(str).tolist(),
        conventional=compare["conventional"].astype(float).tolist(),
        hdps=compare["hdps"].astype(float).fillna(np.nan).tolist(),
        title="Weighted balance diagnostics",
    )


def _plot_effects(effects: pd.DataFrame, path: Path) -> None:
    plot_df = effects[["analysis", "risk_ratio"]].copy()
    save_barh(
        path,
        labels=plot_df["analysis"].astype(str).tolist(),
        values=plot_df["risk_ratio"].astype(float).tolist(),
        title="Effect estimates by adjustment strategy",
        x_label="365-day risk ratio",
        vline=1.0,
    )


def _write_static_reports(
    config: Mapping[str, Any],
    paths: Any,
    attrition: pd.DataFrame,
    baseline: pd.DataFrame,
    missing: pd.DataFrame,
    effects: pd.DataFrame,
    positivity: pd.DataFrame,
    hdps_ranking: pd.DataFrame,
    nc_candidates: pd.DataFrame,
    nc_results: pd.DataFrame,
    qba_summary: Mapping[str, object],
    validation: pd.DataFrame,
    iv_screen: pd.DataFrame,
    alternatives: pd.DataFrame,
    simulation: pd.DataFrame,
    grade: str,
    grade_reason: str,
    synthetic_truth: Mapping[str, float],
    flags: Mapping[str, Mapping[str, bool]],
) -> None:
    research = rq(config)
    exposure = exposure_label(config)
    comparator = comparator_label(config)
    outcome_name = nested_get(config, ["research_question", "outcome", "name"], "outcome")
    estimand = nested_get(config, ["research_question", "estimand", "primary_estimand"], "configured estimand")

    write_markdown(
        paths.reports_dir / "target_trial_table.md",
        f"""# Target Trial Specification

| Component | Emulation |
|---|---|
| Eligibility criteria | {'; '.join(research['population']['inclusion_criteria'])}. Exclusions: {'; '.join(research['population']['exclusion_criteria'])}. |
| Treatment strategies | Initiate {exposure} versus initiate {comparator}. |
| Assignment procedure being emulated | Active-comparator new-user design; treatment is observed, not randomized, so exchangeability is pursued by design and adjustment. |
| Time zero | First qualifying dispensing or prescription after baseline washout. |
| Follow-up | Treatment-policy follow-up through {research['outcome']['primary_follow_up_days']} days, event, death/censoring, or data end. |
| Outcome definition | {outcome_name}; {research['outcome']['outcome_ascertainment_notes']} |
| Causal contrast | {estimand}; primary measures: {', '.join(research['estimand']['primary_effect_measures'])}. |
| Analysis plan | Estimate propensity scores, apply IPTW/stabilized IPTW/truncation/overlap weights, check balance and positivity, estimate 365-day risks and time-to-event proxy HR, then stress-test residual confounding. |
| Known deviations from ideal randomized trial | Treatment is not randomized; adherence, treatment switching, death, censoring, unmeasured clinical severity, and clinician preference may remain imperfectly measured. |
""",
    )

    write_markdown(
        paths.reports_dir / "dag.md",
        f"""# DAG

```mermaid
flowchart LR
  U["Suspected unmeasured confounders: glycemic control, renal function, BMI severity, smoking, frailty, SES, adherence, prescribing preference"] --> A["Exposure: {exposure}"]
  U --> Y["Outcome: {outcome_name}"]
  L["Measured confounders: demographics, calendar time, comorbidity, utilization, medications, labs/proxies"] --> A
  L --> Y
  P["Proxy / hdPS code history"] --> A
  P --> Y
  U --> P
  HCU["Healthcare utilization and coding intensity"] --> P
  HCU --> A
  HCU --> Y
  S["Selection and censoring"] --> A
  S --> Y
  A --> Y
  A -. no assumed causal effect .-> NCO["Negative-control outcome"]
  U --> NCO
  L --> NCO
```

This DAG is a working identification aid, not proof of identifiability.
""",
    )

    write_markdown(
        paths.reports_dir / "protocol.md",
        f"""# Protocol

## Research Question
{research['plain_language_question']}

## Estimand
Primary: {estimand}

Secondary: {research['estimand']['secondary_estimand']}

## Data Source Assumption
The demo runs on synthetic claims/EHR-like data because no real data were supplied. Real data must provide the configured required tables or an adapter that produces the same analytic columns.

## Identification Position
The workflow minimizes and diagnoses unmeasured confounding; it does not claim to eliminate it. Any real-data causal claim requires explicit exchangeability, positivity, consistency, measurement, and censoring assumptions.
""",
    )

    write_markdown(
        paths.reports_dir / "analysis_plan.md",
        f"""# Analysis Plan

1. Build a new-user active-comparator cohort using only pre-index baseline information.
2. Estimate conventional propensity scores using investigator-specified measured confounders.
3. Estimate proxy-rich and hdPS-like propensity scores using pre-index code history.
4. Apply stabilized IPTW, truncation, and overlap weighting.
5. Check weighted balance, positivity, effective sample size, missingness, and attrition.
6. Estimate 365-day risk difference, risk ratio, and a weighted incidence-rate proxy for the hazard ratio.
7. Run negative-control outcome analysis for bias detection.
8. Quantify unmeasured-confounding sensitivity with E-value-style and binary-confounder tipping-point analyses.
9. Demonstrate validation calibration and IV screening logic on synthetic data.
10. Run simulations varying latent-confounding strength, proxy quality, negative-control validity stressors, positivity, outcome rarity, and sample size.
""",
    )

    write_markdown(
        paths.reports_dir / "limitations_and_assumptions.md",
        f"""# Limitations and Assumptions

## Untestable Assumptions
- Conditional exchangeability after measured/proxy adjustment is not testable.
- Proxy and hdPS adjustment helps only if observed code history captures latent causes of treatment and outcome.
- Negative controls require no causal exposure effect and a bias structure close enough to the primary outcome.
- QBA scenarios are only as credible as the prevalence and association values supplied.
- Validation calibration assumes the validation subset is representative and measures the latent factor comparably.
- IV analyses require relevance, exclusion restriction, independence, and monotonicity; this workflow screens candidates but does not automatically endorse them.

## Operational Limits
The demo uses synthetic data with known latent U. Real-data adapters, concept-set review, outcome validation, missing-data handling, censoring models, and death competing-risk analyses require domain and data-owner review.
""",
    )

    primary = effects.loc[effects["analysis"].eq("conventional_ps_stabilized_iptw")].iloc[0]
    hdps_effect = effects.loc[effects["analysis"].eq("hdps_proxy_stabilized_iptw")].iloc[0]
    oracle_effect = effects.loc[effects["analysis"].eq("oracle_adjusts_true_u")].iloc[0]
    sim_brief = simulation.sort_values("false_reassurance_rate", ascending=False).head(8)
    write_markdown(
        paths.reports_dir / "results_report.md",
        f"""# Results Report

## Research Question
{research['plain_language_question']}

## Target Trial Specification
See `reports/target_trial_table.md`. The primary estimand is: {estimand}

## Data Source Assumptions
No real data were supplied, so this run used synthetic claims/EHR-like data with known latent U. True synthetic 365-day RR: {synthetic_truth['true_365_risk_ratio']:.3f}; true HR parameter: {synthetic_truth['true_treatment_hazard_ratio']:.3f}.

## Cohort Attrition
{attrition_markdown(attrition)}

## Baseline, Missingness, Balance, and Positivity
Top unweighted baseline imbalances:

{table_to_markdown(baseline, max_rows=8)}

Missingness summary:

{table_to_markdown(missing, max_rows=8)}

Positivity diagnostics:

{table_to_markdown(positivity)}

Fragility flags:

{table_to_markdown(pd.DataFrame([{'model': k, **v} for k, v in flags.items()]))}

## Primary Estimate
Conventional measured-confounder stabilized IPTW estimated RR {primary['risk_ratio']:.3f}, RD {primary['risk_difference']:.3f}, and incidence-rate proxy HR {primary['hazard_ratio_rate_proxy']:.3f}.

## hdPS / Proxy-Adjusted Estimate
hdPS/proxy stabilized IPTW estimated RR {hdps_effect['risk_ratio']:.3f}, RD {hdps_effect['risk_difference']:.3f}, and incidence-rate proxy HR {hdps_effect['hazard_ratio_rate_proxy']:.3f}.

Selected pre-index proxy features:

{table_to_markdown(hdps_ranking[['feature', 'prevalence', 'exposure_association', 'outcome_association', 'bias_potential']] if not hdps_ranking.empty else hdps_ranking, max_rows=10)}

{hdps_warning()}

Oracle adjustment using the synthetic true U estimated RR {oracle_effect['risk_ratio']:.3f}; this is shown only to stress-test bias and is not available in real data.

## Negative-Control Results
Candidate table:

{table_to_markdown(nc_candidates, max_rows=6)}

Bias-detection results:

{table_to_markdown(nc_results)}

## QBA / Tipping Point
E-value-style summary: {float(qba_summary['e_value']):.3f}.

{qba_summary['plain_language']}

Full grid: `reports/tables/qba_results.csv`. Plot: `reports/tables/qba_tipping_point_plot.png`.

## Validation / External Calibration
{table_to_markdown(validation)}

## IV Feasibility Assessment
{table_to_markdown(iv_screen)}

## Optional Design Alternatives
{table_to_markdown(alternatives)}

## Simulation Findings
Summary rows with highest false-reassurance rate:

{table_to_markdown(sim_brief)}

## Residual Confounding Credibility Grade
Grade: **{grade}**.

Reason: {grade_reason}

## Bottom Line
This synthetic demonstration is successful as a reproducible workbench, but it does not prove causal identification for real patients. Real-data use requires concept-set validation, clinical review of negative controls, data-quality checks for time zero and censoring, and defensible assumptions for residual-confounding sensitivity values.
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pharmacoepidemiology unmeasured-confounding workbench.")
    parser.add_argument("--config", default="config/research_question.yaml")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    result = run_analysis(args.config, args.reports_dir)
    print(f"Analysis complete. Reports written to {Path(result['reports_dir']).resolve()}")
    print(f"Residual confounding credibility grade: {result['credibility_grade']}")


if __name__ == "__main__":
    main()
