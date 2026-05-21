from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cohort import build_new_user_active_comparator_cohort
from .covariates import baseline_covariate_columns
from .effect_estimation import estimate_effects
from .hdps import append_hdps_to_cohort, build_hdps_features
from .propensity import estimate_propensity_scores, fragility_flags
from .quantitative_bias_analysis import binary_confounder_bias_factor
from .synthetic_data import generate_synthetic_data


def _rough_log_rr_ci(row: dict[str, float | str]) -> tuple[float, float]:
    e1 = max(float(row.get("events_exposed_weighted", 1.0)), 1.0)
    e0 = max(float(row.get("events_comparator_weighted", 1.0)), 1.0)
    rr = max(float(row["risk_ratio"]), 1e-9)
    se = np.sqrt(1.0 / e1 + 1.0 / e0)
    lo = float(np.exp(np.log(rr) - 1.96 * se))
    hi = float(np.exp(np.log(rr) + 1.96 * se))
    return lo, hi


def run_simulation_study(config: Mapping[str, Any], seed: int = 20260520) -> pd.DataFrame:
    sim_cfg = config.get("analysis", {}).get("simulation", {})
    sample_sizes = sim_cfg.get("sample_sizes", [1200])
    repetitions = int(sim_cfg.get("repetitions", 4))
    u_tx_values = sim_cfg.get("u_treatment_strengths", [0.0, 0.5, 1.0])
    u_y_values = sim_cfg.get("u_outcome_strengths", [0.0, 0.5, 1.0])
    proxy_qualities = sim_cfg.get("proxy_qualities", [0.25, 0.75])
    positivity_values = sim_cfg.get("positivity_stress_values", [0.0, 0.8])
    hazard_values = sim_cfg.get("outcome_baseline_hazards_per_day", [0.0003, 0.0007])
    thresholds = config.get("analysis", {}).get("diagnostic_thresholds", {})
    rows = []
    scenario_id = 0
    rng = np.random.default_rng(seed)
    base_scenario = (
        sample_sizes[0],
        u_tx_values[0],
        u_y_values[0],
        proxy_qualities[0],
        positivity_values[0],
        hazard_values[0],
    )
    scenario_set = {base_scenario}
    for n in sample_sizes:
        scenario_set.add((n, base_scenario[1], base_scenario[2], base_scenario[3], base_scenario[4], base_scenario[5]))
    for u_tx in u_tx_values:
        for u_y in u_y_values:
            scenario_set.add((base_scenario[0], u_tx, u_y, base_scenario[3], base_scenario[4], base_scenario[5]))
    for proxy_quality in proxy_qualities:
        scenario_set.add((base_scenario[0], u_tx_values[-1], u_y_values[-1], proxy_quality, base_scenario[4], base_scenario[5]))
    for positivity_stress in positivity_values:
        scenario_set.add((base_scenario[0], u_tx_values[-1], u_y_values[-1], proxy_qualities[-1], positivity_stress, base_scenario[5]))
    for hazard in hazard_values:
        scenario_set.add((base_scenario[0], u_tx_values[-1], u_y_values[-1], proxy_qualities[-1], positivity_values[-1], hazard))

    for n, u_tx, u_y, proxy_quality, positivity_stress, hazard in sorted(scenario_set):
        scenario_id += 1
        for rep in range(repetitions):
            synthetic = generate_synthetic_data(
                config,
                n=int(n),
                seed=int(rng.integers(1, 1_000_000_000)),
                u_treatment_strength=float(u_tx),
                u_outcome_strength=float(u_y),
                proxy_quality=float(proxy_quality),
                positivity_stress=float(positivity_stress),
                outcome_baseline_hazard_per_day=float(hazard),
            )
            cohort, _ = build_new_user_active_comparator_cohort(synthetic.tables, config)
            measured_covars = baseline_covariate_columns(cohort, include_proxy=False)
            proxy_covars = baseline_covariate_columns(cohort, include_proxy=True)
            true_rr = synthetic.truth["true_365_risk_ratio"]

            crude = estimate_effects(cohort, weight_col=None, label="crude")
            rows.append(
                _sim_row(scenario_id, rep, "crude", crude, true_rr, False, n, u_tx, u_y, proxy_quality, positivity_stress, hazard)
            )

            measured_ps = estimate_propensity_scores(cohort, measured_covars)
            measured_eff = estimate_effects(measured_ps.data, weight_col="weight_stabilized_iptw", label="measured_ps")
            measured_flags = fragility_flags(measured_ps.diagnostics, measured_ps.balance, thresholds)
            rows.append(
                _sim_row(
                    scenario_id,
                    rep,
                    "measured_ps",
                    measured_eff,
                    true_rr,
                    _diagnostics_acceptable(measured_flags),
                    n,
                    u_tx,
                    u_y,
                    proxy_quality,
                    positivity_stress,
                    hazard,
                )
            )

            hdps_features, ranking = build_hdps_features(cohort, synthetic.tables["code_history"], top_k=10)
            hdps_cohort = append_hdps_to_cohort(cohort, hdps_features)
            hdps_covars = proxy_covars + ranking["feature"].tolist()
            hdps_ps = estimate_propensity_scores(hdps_cohort, hdps_covars)
            hdps_eff = estimate_effects(hdps_ps.data, weight_col="weight_stabilized_iptw", label="hdps_proxy")
            hdps_flags = fragility_flags(hdps_ps.diagnostics, hdps_ps.balance, thresholds)
            rows.append(
                _sim_row(
                    scenario_id,
                    rep,
                    "hdps_proxy",
                    hdps_eff,
                    true_rr,
                    _diagnostics_acceptable(hdps_flags),
                    n,
                    u_tx,
                    u_y,
                    proxy_quality,
                    positivity_stress,
                    hazard,
                )
            )

            oracle_covars = proxy_covars + ["latent_u"]
            oracle_ps = estimate_propensity_scores(cohort, oracle_covars)
            oracle_eff = estimate_effects(oracle_ps.data, weight_col="weight_stabilized_iptw", label="oracle_u")
            rows.append(
                _sim_row(scenario_id, rep, "oracle_u", oracle_eff, true_rr, True, n, u_tx, u_y, proxy_quality, positivity_stress, hazard)
            )

            u_high = cohort["latent_u"] > 0
            p1 = float(u_high[cohort["A"].eq(1)].mean())
            p0 = float(u_high[cohort["A"].eq(0)].mean())
            rr_u = float(np.exp(u_y))
            bf = binary_confounder_bias_factor(p1, p0, rr_u)
            qba_eff = dict(measured_eff)
            qba_eff["risk_ratio"] = float(qba_eff["risk_ratio"]) / max(bf, 1e-9)
            rows.append(
                _sim_row(
                    scenario_id,
                    rep,
                    "qba_adjusted_known_u_scenario",
                    qba_eff,
                    true_rr,
                    _diagnostics_acceptable(measured_flags),
                    n,
                    u_tx,
                    u_y,
                    proxy_quality,
                    positivity_stress,
                    hazard,
                )
            )
    raw = pd.DataFrame(rows)
    return summarize_simulation(raw)


def _diagnostics_acceptable(flags: dict[str, bool]) -> bool:
    return not any(flags.values())


def _sim_row(
    scenario_id: int,
    repetition: int,
    method: str,
    effect: dict[str, float | str],
    true_rr: float,
    diagnostics_acceptable: bool,
    n: int,
    u_tx: float,
    u_y: float,
    proxy_quality: float,
    positivity_stress: float,
    hazard: float,
) -> dict[str, float | str | bool | int]:
    rr = float(effect["risk_ratio"])
    ci_low, ci_high = _rough_log_rr_ci(effect)
    log_bias = float(np.log(max(rr, 1e-9)) - np.log(max(true_rr, 1e-9)))
    return {
        "scenario_id": scenario_id,
        "repetition": repetition,
        "method": method,
        "n": n,
        "u_treatment_strength": u_tx,
        "u_outcome_strength": u_y,
        "proxy_quality": proxy_quality,
        "positivity_stress": positivity_stress,
        "outcome_baseline_hazard_per_day": hazard,
        "estimate_rr": rr,
        "true_rr": true_rr,
        "log_rr_bias": log_bias,
        "covered": ci_low <= true_rr <= ci_high,
        "diagnostics_acceptable": diagnostics_acceptable,
        "false_reassurance": diagnostics_acceptable and abs(log_bias) > 0.10,
    }


def summarize_simulation(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "u_treatment_strength", "u_outcome_strength", "proxy_quality", "positivity_stress", "outcome_baseline_hazard_per_day"]
    summary = (
        raw.groupby(group_cols)
        .agg(
            mean_rr=("estimate_rr", "mean"),
            true_rr=("true_rr", "mean"),
            bias=("log_rr_bias", "mean"),
            variance=("log_rr_bias", "var"),
            mean_squared_error=("log_rr_bias", lambda s: float(np.mean(np.asarray(s) ** 2))),
            coverage=("covered", "mean"),
            false_reassurance_rate=("false_reassurance", "mean"),
            repetitions=("estimate_rr", "size"),
        )
        .reset_index()
    )
    summary["variance"] = summary["variance"].fillna(0.0)
    return summary
