# Results Report

## Research Question
Among adults with type 2 diabetes initiating an SGLT2 inhibitor versus a GLP-1 receptor agonist, what is the effect on 1-year hospitalization for heart failure?

## Target Trial Specification
See `reports/target_trial_table.md`. The primary estimand is: Average treatment effect among eligible new users under a treatment-policy strategy

## Data Source Assumptions
No real data were supplied, so this run used synthetic claims/EHR-like data with known latent U. True synthetic 365-day RR: 0.818; true HR parameter: 0.780.

## Cohort Attrition
| Step | Excluded | Remaining |
|---|---:|---:|
| Source population | 0 | 5000 |
| Age >= 40 years on index date | 13 | 4987 |
| Evidence of type 2 diabetes before index date | 0 | 4987 |
| At least 365 days continuous enrollment before index date | 0 | 4987 |
| No use of either study drug class in prior 365 days | 231 | 4756 |
| No outcome event in prior 180 days | 241 | 4515 |
| Nonmissing age and sex | 6 | 4509 |
| No data-quality violation making time zero ambiguous | 56 | 4453 |
| Qualifying index dispensing or prescription | 0 | 4453 |
| Mutually exclusive exposure/comparator assignment at time zero | 0 | 4453 |
| Clear time zero | 0 | 4453 |

## Baseline, Missingness, Balance, and Positivity
Top unweighted baseline imbalances:

| variable | treated_mean | comparator_mean | smd |
| --- | --- | --- | --- |
| lab_test_count | 5.693 | 5.133 | 0.227 |
| ckd | 0.353 | 0.272 | 0.176 |
| utilization_visits | 4.712 | 4.361 | 0.152 |
| site_id | 4.626 | 4.319 | 0.133 |
| prior_hf | 0.190 | 0.146 | 0.118 |
| calendar_year_centered | 0.050 | -0.043 | 0.086 |
| bmi | 30.954 | 30.554 | 0.076 |
| sex_female | 0.461 | 0.494 | -0.067 |

Missingness summary:

| variable | missing_n | missing_percent |
| --- | --- | --- |
| age | 0 | 0.000 |
| sex_female | 0 | 0.000 |
| race_black | 0 | 0.000 |
| race_hispanic | 0 | 0.000 |
| race_other | 0 | 0.000 |
| calendar_year_centered | 0 | 0.000 |
| calendar_quarter | 0 | 0.000 |
| diabetes_duration_years | 0 | 0.000 |

Positivity diagnostics:

| model | ps_min | ps_p01 | ps_p50 | ps_p99 | ps_max | common_support_low | common_support_high | fraction_outside_common_support | max_weight | mean_weight | effective_sample_size | effective_sample_size_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conventional_ps | 0.211 | 0.304 | 0.476 | 0.701 | 0.801 | 0.252 | 0.791 | 0.001 | 2.611 | 1.000 | 4302.715 | 0.966 |
| hdps_proxy_ps | 0.137 | 0.237 | 0.475 | 0.772 | 0.866 | 0.185 | 0.844 | 0.001 | 3.320 | 1.000 | 4154.635 | 0.933 |

Fragility flags:

| model | major_positivity_violation | effective_sample_size_collapse | residual_weighted_imbalance |
| --- | --- | --- | --- |
| conventional_ps | False | False | False |
| hdps_proxy_ps | False | False | False |

## Primary Estimate
Conventional measured-confounder stabilized IPTW estimated RR 1.159, RD 0.025, and incidence-rate proxy HR 1.221.

## hdPS / Proxy-Adjusted Estimate
hdPS/proxy stabilized IPTW estimated RR 1.105, RD 0.017, and incidence-rate proxy HR 1.160.

Selected pre-index proxy features:

| feature | prevalence | exposure_association | outcome_association | bias_potential |
| --- | --- | --- | --- | --- |
| hdps_diagnosis__dx_ckd | 0.311 | 0.081 | 0.139 | 0.009 |
| hdps_lab__lab_intense_monitoring | 0.350 | 0.077 | 0.029 | 0.005 |
| hdps_diagnosis__dx_cvd | 0.298 | 0.022 | 0.073 | 0.004 |
| hdps_diagnosis__dx_prior_hf | 0.168 | 0.044 | 0.112 | 0.004 |
| hdps_medication__rx_ras_blocker | 0.397 | 0.043 | 0.016 | 0.004 |
| hdps_visit__visit_high_util | 0.274 | 0.034 | 0.038 | 0.004 |
| hdps_medication__rx_statin | 0.410 | 0.018 | 0.021 | 0.003 |
| hdps_diagnosis__dx_smoking_proxy | 0.236 | 0.015 | 0.055 | 0.003 |
| hdps_diagnosis__dx_htn | 0.649 | 0.013 | 0.021 | 0.003 |
| hdps_procedure__px_frailty_eval | 0.146 | 0.052 | 0.061 | 0.003 |

Proxy and hdPS adjustment can reduce residual confounding only if observed pre-index code history captures information about unmeasured causes of treatment and outcome. It is not proof that latent confounding has been removed.

Oracle adjustment using the synthetic true U estimated RR 0.860; this is shown only to stress-test bias and is not available in real data.

## Negative-Control Results
Candidate table:

| candidate | type | status | rationale | required_assumptions | reasons_to_reject |
| --- | --- | --- | --- | --- | --- |
| acute appendicitis hospitalization | outcome | proposed, requires clinical review | Acute appendicitis should not plausibly be prevented or caused by initiation of either diabetes drug class over one year, but it can share healthcare-seeking and coding intensity bias. | No causal exposure effect, no contraindication-mediated pathway, comparable capture, and shared residual confounding structure. | Reject if clinical review identifies a plausible drug effect, prescribing-channel link, differential capture, or weak overlap with primary-outcome bias sources. |
| traumatic injury encounter | outcome | proposed, requires clinical review | Short-term traumatic injury should not be a direct pharmacologic effect for this comparison; utilization and frailty may still create residual association. | No causal exposure effect, no contraindication-mediated pathway, comparable capture, and shared residual confounding structure. | Reject if clinical review identifies a plausible drug effect, prescribing-channel link, differential capture, or weak overlap with primary-outcome bias sources. |
| cataract procedure | outcome | proposed, requires clinical review | Cataract procedure timing is not expected to be an acute causal consequence of initiation, but access-to-care and health-system factors can bias it. | No causal exposure effect, no contraindication-mediated pathway, comparable capture, and shared residual confounding structure. | Reject if clinical review identifies a plausible drug effect, prescribing-channel link, differential capture, or weak overlap with primary-outcome bias sources. |

Bias-detection results:

| negative_control | analysis | risk_exposed | risk_comparator | risk_ratio | log_risk_ratio | residual_association_detected | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic negative-control outcome | negative_control_crude | 0.071 | 0.070 | 1.023 | 0.023 | False | Near-null residual association under this synthetic negative control. |
| synthetic negative-control outcome | negative_control_adjusted | 0.068 | 0.073 | 0.937 | -0.065 | False | Near-null residual association under this synthetic negative control. |

## QBA / Tipping Point
E-value-style summary: 1.588.

On the configured grid, a latent factor with prevalence 0.60 in the exposure group and 0.10 in the comparator group, plus an outcome RR of 1.25, would move the estimate into the clinically unimportant range.

Full grid: `reports/tables/qba_results.csv`. Plot: `reports/tables/qba_tipping_point_plot.png`.

## Validation / External Calibration
| method | status | validation_n | conventional_rr_subset | oracle_rr_subset | log_rr_calibration_delta | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| propensity_score_calibration_demo | synthetic_example | 1113 | 1.460 | 1.118 | -0.267 | Apply this delta to the full-data log RR only if the validation subset is representative and U is measured comparably. |
| external_adjustment_stub | ready_for_inputs | 0 |  |  |  | Supply external prevalence and U-outcome association estimates, then route them through the QBA grid. |
| two_stage_imputation_stub | ready_for_inputs | 0 |  |  |  | Use when lab/EHR/chart-review data measure otherwise unavailable confounders in a subset. |

## IV Feasibility Assessment
| candidate | variable | relevance_first_stage_difference | first_stage_f_statistic_approx | max_covariate_smd_by_instrument | exclusion_restriction_concerns | independence_concerns | monotonicity_concerns | recommendation | effect_estimation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clinician prescribing preference | clinician_preference | 0.147 | 96.489 | 0.064 | Preference may affect monitoring, adherence support, or coding beyond drug choice. | Preference can cluster by site, patient mix, formulary, and calendar time. | Some clinicians may prefer one class only for selected phenotypes. | plausible with caveats | not_performed by default; requires pre-specified protocol sign-off |
| facility preference | facility_preference_score | 0.066 | 18.356 | 3.453 | Preference may affect monitoring, adherence support, or coding beyond drug choice. | Preference can cluster by site, patient mix, formulary, and calendar time. | Some clinicians may prefer one class only for selected phenotypes. | do not use | not_performed; screening only |

## Optional Design Alternatives
| design | compatible | reason |
| --- | --- | --- |
| self-controlled case series | usually no | SGLT2 inhibitor initiation vs GLP-1 receptor agonist initiation initiation is a chronic treatment choice, and Hospitalization for heart failure can alter future exposure and observation. |
| case-crossover | usually no | Best for transient exposures with acute effects; this question concerns sustained initiation strategies. |
| case-time-control | limited | Could address exposure time trends for acute windows, but does not naturally target the 365-day new-user estimand. |
| prior event rate ratio adjustment | exploratory | May help with time-stable frailty/utilization differences, but incident-outcome exclusion and treatment switching complicate interpretation. |
| difference-in-differences | only with policy/formulary shock | Requires a defensible intervention time and parallel-trends evidence. |
| within-provider or within-site design | possibly | Can reduce site/provider confounding if enough overlap remains and provider preference is not a direct pathway to outcomes. |

## Simulation Findings
Summary rows with highest false-reassurance rate:

| method | u_treatment_strength | u_outcome_strength | proxy_quality | positivity_stress | outcome_baseline_hazard_per_day | mean_rr | true_rr | bias | variance | mean_squared_error | coverage | false_reassurance_rate | repetitions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hdps_proxy | 1.000 | 1.000 | 0.250 | 0.000 | 0.000 | 1.293 | 0.825 | 0.445 | 0.012 | 0.207 | 0.000 | 1.000 | 4 |
| hdps_proxy | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 1.313 | 0.826 | 0.458 | 0.015 | 0.221 | 0.250 | 1.000 | 4 |
| hdps_proxy | 1.000 | 1.000 | 0.750 | 0.800 | 0.000 | 1.194 | 0.824 | 0.367 | 0.008 | 0.141 | 0.250 | 1.000 | 4 |
| hdps_proxy | 0.500 | 0.500 | 0.250 | 0.000 | 0.000 | 0.837 | 0.802 | 0.010 | 0.091 | 0.068 | 1.000 | 1.000 | 4 |
| qba_adjusted_known_u_scenario | 1.000 | 1.000 | 0.250 | 0.000 | 0.000 | 1.057 | 0.825 | 0.244 | 0.011 | 0.067 | 0.750 | 1.000 | 4 |
| qba_adjusted_known_u_scenario | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 1.114 | 0.826 | 0.297 | 0.007 | 0.093 | 0.250 | 1.000 | 4 |
| oracle_u | 1.000 | 0.500 | 0.250 | 0.000 | 0.000 | 0.693 | 0.803 | -0.161 | 0.039 | 0.055 | 1.000 | 1.000 | 4 |
| oracle_u | 0.500 | 0.500 | 0.250 | 0.000 | 0.000 | 0.767 | 0.802 | -0.075 | 0.083 | 0.068 | 0.750 | 1.000 | 4 |

## Residual Confounding Credibility Grade
Grade: **Moderate**.

Reason: Sensitivity or simulation results leave a material residual-confounding concern.

## Bottom Line
This synthetic demonstration is successful as a reproducible workbench, but it does not prove causal identification for real patients. Real-data use requires concept-set validation, clinical review of negative controls, data-quality checks for time zero and censoring, and defensible assumptions for residual-confounding sensitivity values.
