from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data_model import (
    analysis_config,
    comparator_label,
    exposure_label,
    follow_up_days,
    nested_get,
)


@dataclass
class SyntheticData:
    tables: dict[str, pd.DataFrame]
    truth: dict[str, float]
    parameters: dict[str, float]


def expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def _binary(rng: np.random.Generator, p: np.ndarray | float) -> np.ndarray:
    return rng.binomial(1, np.clip(p, 0.001, 0.999))


def _date_offsets(index_dates: pd.Series, rng: np.random.Generator, low: int, high: int) -> pd.Series:
    return index_dates - pd.to_timedelta(rng.integers(low, high, size=len(index_dates)), unit="D")


def generate_synthetic_data(
    config: Mapping[str, Any],
    n: int | None = None,
    seed: int | None = None,
    true_treatment_hazard_ratio: float | None = None,
    u_treatment_strength: float | None = None,
    u_outcome_strength: float | None = None,
    proxy_quality: float | None = None,
    positivity_stress: float | None = None,
    outcome_baseline_hazard_per_day: float | None = None,
) -> SyntheticData:
    """Generate claims/EHR-like data with a known latent confounder U.

    Treatment is generated after baseline. All generated adjustment covariates
    are pre-index; a deliberately tempting post-index variable is included so
    tests can verify it is excluded.
    """
    analysis = analysis_config(config)
    params = dict(analysis.get("synthetic_parameters", {}))
    n = int(n if n is not None else analysis.get("synthetic_n", 5000))
    seed = int(seed if seed is not None else analysis.get("random_seed", 20260520))
    true_hr = float(
        true_treatment_hazard_ratio
        if true_treatment_hazard_ratio is not None
        else params.get("true_treatment_hazard_ratio", 0.78)
    )
    u_tx = float(u_treatment_strength if u_treatment_strength is not None else params.get("u_treatment_strength", 0.75))
    u_y = float(u_outcome_strength if u_outcome_strength is not None else params.get("u_outcome_strength", 0.70))
    proxy_quality = float(proxy_quality if proxy_quality is not None else params.get("proxy_quality", 0.65))
    positivity_stress = float(positivity_stress if positivity_stress is not None else params.get("positivity_stress", 0.0))
    base_hazard = float(
        outcome_baseline_hazard_per_day
        if outcome_baseline_hazard_per_day is not None
        else params.get("outcome_baseline_hazard_per_day", 0.00042)
    )
    followup = follow_up_days(config)
    rng = np.random.default_rng(seed)

    person_id = np.arange(1, n + 1)
    age = np.clip(rng.normal(63, 10, n), 40, 90)
    sex_female = _binary(rng, np.full(n, 0.48))
    race_draw = rng.choice(["white", "black", "hispanic", "other"], size=n, p=[0.62, 0.17, 0.14, 0.07])
    race_black = (race_draw == "black").astype(int)
    race_hispanic = (race_draw == "hispanic").astype(int)
    race_other = (race_draw == "other").astype(int)
    calendar_year = rng.choice([2019, 2020, 2021, 2022], size=n, p=[0.20, 0.25, 0.27, 0.28])
    calendar_quarter = rng.integers(1, 5, n)
    calendar_year_centered = calendar_year - np.mean(calendar_year)

    socioeconomic_risk = rng.normal(0, 1, n)
    clinician_preference = rng.normal(0, 1, n)
    latent_u = rng.normal(0, 1, n)
    diabetes_duration_years = np.clip(rng.gamma(2.2, 3.0, n) + 0.04 * (age - 60), 0.5, 28)
    a1c_proxy = np.clip(7.3 + 0.25 * latent_u + 0.04 * diabetes_duration_years + rng.normal(0, 0.8, n), 5.4, 12.5)
    egfr_proxy = np.clip(84 - 0.62 * age - 6.0 * latent_u + rng.normal(0, 12, n), 10, 110)
    bmi = np.clip(31 + 1.4 * latent_u - 0.08 * (age - 60) + rng.normal(0, 5, n), 18, 55)
    obesity = (bmi >= 30).astype(int)
    smoking_proxy = _binary(rng, expit(-1.2 + 0.38 * latent_u + 0.25 * socioeconomic_risk))
    frailty_proxy = _binary(rng, expit(-2.0 + 0.04 * (age - 65) + proxy_quality * latent_u))
    ckd = _binary(rng, expit(-1.5 + 0.045 * (age - 60) + 0.65 * (egfr_proxy < 60) + 0.30 * latent_u))
    hypertension = _binary(rng, expit(0.4 + 0.05 * (age - 60) + 0.35 * obesity))
    cvd = _binary(rng, expit(-1.15 + 0.035 * (age - 60) + 0.45 * smoking_proxy + 0.32 * ckd))
    prior_hf = _binary(rng, expit(-2.25 + 0.045 * (age - 60) + 0.70 * ckd + 0.70 * cvd + 0.30 * latent_u))
    evidence_t2d = np.ones(n, dtype=int)
    enrollment_days_prior = rng.integers(365, 1700, n)
    prior_study_drug_use = _binary(rng, np.full(n, 0.045))
    outcome_prior_180 = _binary(rng, expit(-3.25 + 1.15 * prior_hf + 0.30 * ckd))
    data_quality_violation = _binary(rng, np.full(n, 0.012))
    missing_age = _binary(rng, np.full(n, 0.002))
    missing_sex = _binary(rng, np.full(n, 0.002))
    age_with_missing = age.copy()
    age_with_missing[missing_age == 1] = np.nan
    sex_with_missing = sex_female.astype(float)
    sex_with_missing[missing_sex == 1] = np.nan

    utilization_visits = np.maximum(
        0,
        rng.poisson(np.exp(1.4 + 0.012 * (age - 60) + 0.16 * ckd + 0.15 * frailty_proxy + 0.10 * latent_u)),
    )
    emergency_visits = rng.poisson(np.exp(-1.2 + 0.22 * prior_hf + 0.15 * frailty_proxy + 0.08 * latent_u))
    inpatient_visits = rng.poisson(np.exp(-1.8 + 0.38 * prior_hf + 0.18 * ckd + 0.12 * latent_u))
    medication_count = rng.poisson(np.exp(1.9 + 0.09 * diabetes_duration_years / 5 + 0.20 * cvd + 0.12 * prior_hf))
    lab_test_count = rng.poisson(np.exp(1.55 + 0.16 * ckd + 0.08 * utilization_visits / 5 + 0.10 * latent_u))
    site_id = rng.integers(1, 9, n)
    site_preference = (site_id - np.mean(site_id)) / np.std(site_id)

    treatment_logit = (
        -0.05
        + 0.33 * ckd
        + 0.22 * cvd
        - 0.30 * obesity
        - 0.22 * (a1c_proxy > 8.5)
        + 0.14 * calendar_year_centered
        + 0.18 * site_preference
        + 0.35 * clinician_preference
        + u_tx * latent_u
        + positivity_stress * (0.55 * ckd - 0.45 * obesity + 0.40 * prior_hf)
    )
    ps_true = expit(treatment_logit)
    treatment = _binary(rng, ps_true)

    index_base = pd.Timestamp("2019-01-01")
    index_date = index_base + pd.to_timedelta(
        ((calendar_year - 2019) * 365) + ((calendar_quarter - 1) * 91) + rng.integers(0, 80, n),
        unit="D",
    )

    measured_lp = (
        0.015 * (age - 60)
        + 0.14 * (1 - sex_female)
        + 0.20 * ckd
        + 0.35 * prior_hf
        + 0.20 * cvd
        + 0.08 * emergency_visits
        + 0.06 * inpatient_visits
        + 0.10 * (a1c_proxy > 9.0)
    )
    hazard0 = base_hazard * np.exp(measured_lp + u_y * latent_u)
    hazard1 = hazard0 * true_hr
    actual_hazard = np.where(treatment == 1, hazard1, hazard0)
    event_time = rng.exponential(1.0 / np.clip(actual_hazard, 0.000001, None))
    administrative_censor = np.full(n, followup, dtype=float)
    random_dropout = rng.exponential(900, n)
    followup_days_observed = np.minimum.reduce([event_time, administrative_censor, random_dropout])
    event_365 = ((event_time <= followup) & (event_time <= random_dropout)).astype(int)
    death_event = _binary(rng, expit(-4.2 + 0.04 * (age - 65) + 0.5 * prior_hf + 0.35 * latent_u))
    negative_control_event_365 = _binary(
        rng,
        expit(-3.0 + 0.25 * (age - 60) / 10 + 0.25 * utilization_visits / 6 + 0.50 * latent_u),
    )

    # Tempting post-index information. It should never be selected as a baseline covariate.
    post_index_healthcare_visits = rng.poisson(np.exp(1.2 + 0.75 * event_365 + 0.15 * treatment))

    persons = pd.DataFrame(
        {
            "person_id": person_id,
            "age": age_with_missing,
            "sex_female": sex_with_missing,
            "race_black": race_black,
            "race_hispanic": race_hispanic,
            "race_other": race_other,
            "calendar_year": calendar_year,
            "calendar_quarter": calendar_quarter,
            "calendar_year_centered": calendar_year_centered,
            "evidence_t2d": evidence_t2d,
            "enrollment_days_prior": enrollment_days_prior,
            "prior_study_drug_use": prior_study_drug_use,
            "outcome_prior_180": outcome_prior_180,
            "data_quality_violation": data_quality_violation,
            "diabetes_duration_years": diabetes_duration_years,
            "cvd": cvd,
            "prior_hf": prior_hf,
            "ckd": ckd,
            "hypertension": hypertension,
            "bmi": bmi,
            "obesity": obesity,
            "smoking_proxy": smoking_proxy,
            "frailty_proxy": frailty_proxy,
            "socioeconomic_risk_proxy": socioeconomic_risk,
            "utilization_visits": utilization_visits,
            "emergency_visits": emergency_visits,
            "inpatient_visits": inpatient_visits,
            "medication_count": medication_count,
            "lab_test_count": lab_test_count,
            "egfr_proxy": egfr_proxy,
            "a1c_proxy": a1c_proxy,
            "site_id": site_id,
            "clinician_preference": clinician_preference,
            "latent_u": latent_u,
            "A": treatment,
            "ps_true": ps_true,
            "index_date": index_date,
            "event_365": event_365,
            "followup_days": np.maximum(1.0, followup_days_observed),
            "death_event": death_event,
            "negative_control_event_365": negative_control_event_365,
            "post_index_healthcare_visits": post_index_healthcare_visits,
        }
    )

    drug_rows = []
    exposure_name = exposure_label(config)
    comparator_name = comparator_label(config)
    for pid, a, idx in zip(person_id, treatment, index_date):
        drug_rows.append(
            {
                "person_id": pid,
                "drug_class": exposure_name if a == 1 else comparator_name,
                "treatment_role": "exposure" if a == 1 else "comparator",
                "drug_start_date": idx,
                "is_index_exposure": 1,
            }
        )
    drug_exposure = pd.DataFrame(drug_rows)

    enrollment = pd.DataFrame(
        {
            "person_id": person_id,
            "observation_start_date": index_date - pd.to_timedelta(enrollment_days_prior, unit="D"),
            "observation_end_date": index_date + pd.to_timedelta(followup, unit="D"),
        }
    )

    code_history = _generate_code_history(persons, rng, proxy_quality)
    conditions = code_history[code_history["dimension"] == "diagnosis"].copy()
    procedures = code_history[code_history["dimension"] == "procedure"].copy()
    visits = code_history[code_history["dimension"] == "visit"].copy()
    measurements = code_history[code_history["dimension"] == "lab"].copy()
    death = pd.DataFrame(
        {
            "person_id": person_id[death_event == 1],
            "death_date": index_date[death_event == 1] + pd.to_timedelta(rng.integers(5, followup + 1, death_event.sum()), unit="D"),
        }
    )

    true_risk0 = 1.0 - np.exp(-hazard0 * followup)
    true_risk1 = 1.0 - np.exp(-hazard1 * followup)
    truth = {
        "true_treatment_hazard_ratio": true_hr,
        "true_365_risk_exposed": float(np.mean(true_risk1)),
        "true_365_risk_comparator": float(np.mean(true_risk0)),
        "true_365_risk_difference": float(np.mean(true_risk1) - np.mean(true_risk0)),
        "true_365_risk_ratio": float(np.mean(true_risk1) / np.mean(true_risk0)),
        "latent_u_treatment_strength": u_tx,
        "latent_u_outcome_strength": u_y,
    }
    tables = {
        "person": persons,
        "continuous_enrollment": enrollment,
        "drug_exposure": drug_exposure,
        "condition_occurrence": conditions,
        "procedure_occurrence": procedures,
        "visit_occurrence": visits,
        "measurement": measurements,
        "death": death,
        "code_history": code_history,
    }
    return SyntheticData(tables=tables, truth=truth, parameters=params | {
        "true_treatment_hazard_ratio": true_hr,
        "u_treatment_strength": u_tx,
        "u_outcome_strength": u_y,
        "proxy_quality": proxy_quality,
        "positivity_stress": positivity_stress,
        "outcome_baseline_hazard_per_day": base_hazard,
    })


def _generate_code_history(persons: pd.DataFrame, rng: np.random.Generator, proxy_quality: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    code_specs = [
        ("diagnosis", "DX_T2D", np.ones(len(persons))),
        ("diagnosis", "DX_CKD", persons["ckd"].to_numpy()),
        ("diagnosis", "DX_HTN", persons["hypertension"].to_numpy()),
        ("diagnosis", "DX_CVD", persons["cvd"].to_numpy()),
        ("diagnosis", "DX_PRIOR_HF", persons["prior_hf"].to_numpy()),
        ("diagnosis", "DX_SMOKING_PROXY", persons["smoking_proxy"].to_numpy()),
        ("diagnosis", "DX_FRAILTY_PROXY", persons["frailty_proxy"].to_numpy()),
        ("diagnosis", "DX_SOCIAL_RISK_PROXY", expit(-1.0 + 0.65 * persons["socioeconomic_risk_proxy"].to_numpy())),
        ("procedure", "PX_CARDIAC_TEST", expit(-1.3 + 0.35 * persons["cvd"].to_numpy() + 0.05 * persons["utilization_visits"].to_numpy())),
        ("procedure", "PX_KIDNEY_MONITORING", expit(-1.0 + 0.45 * persons["ckd"].to_numpy() + 0.04 * persons["lab_test_count"].to_numpy())),
        ("procedure", "PX_FRAILTY_EVAL", expit(-2.0 + proxy_quality * persons["latent_u"].to_numpy() + 0.30 * persons["frailty_proxy"].to_numpy())),
        ("medication", "RX_INSULIN", expit(-1.4 + 0.18 * persons["diabetes_duration_years"].to_numpy() + 0.35 * (persons["a1c_proxy"].to_numpy() > 8.5))),
        ("medication", "RX_LOOP_DIURETIC", expit(-2.1 + 1.00 * persons["prior_hf"].to_numpy() + 0.25 * persons["ckd"].to_numpy())),
        ("medication", "RX_RAS_BLOCKER", expit(-0.8 + 0.45 * persons["hypertension"].to_numpy() + 0.30 * persons["ckd"].to_numpy())),
        ("medication", "RX_STATIN", expit(-0.7 + 0.55 * persons["cvd"].to_numpy() + 0.25 * persons["diabetes_duration_years"].to_numpy() / 10)),
        ("visit", "VISIT_HIGH_UTIL", expit(-1.4 + 0.08 * persons["utilization_visits"].to_numpy() + 0.25 * persons["latent_u"].to_numpy())),
        ("visit", "VISIT_ED", expit(-1.7 + 0.26 * persons["emergency_visits"].to_numpy() + 0.18 * persons["latent_u"].to_numpy())),
        ("lab", "LAB_A1C_HIGH", (persons["a1c_proxy"].to_numpy() > 8.5).astype(float)),
        ("lab", "LAB_EGFR_LOW", (persons["egfr_proxy"].to_numpy() < 60).astype(float)),
        ("lab", "LAB_INTENSE_MONITORING", expit(-1.2 + 0.10 * persons["lab_test_count"].to_numpy() + 0.22 * persons["latent_u"].to_numpy())),
    ]
    index_dates = persons["index_date"].reset_index(drop=True)
    person_ids = persons["person_id"].to_numpy()
    for dimension, code, prob in code_specs:
        prob_array = np.asarray(prob, dtype=float)
        if prob_array.min() < 0 or prob_array.max() > 1:
            prob_array = np.clip(prob_array, 0.001, 0.999)
        present = rng.binomial(1, prob_array)
        date_values = _date_offsets(index_dates, rng, 1, 365).to_numpy()
        for pid, is_present, date_value in zip(person_ids, present, date_values):
            if is_present:
                rows.append(
                    {
                        "person_id": int(pid),
                        "dimension": dimension,
                        "code": code,
                        "code_date": pd.Timestamp(date_value),
                        "count": int(rng.integers(1, 4)),
                    }
                )

    # A small amount of post-index code history is present in source data to
    # ensure the hdPS module proves it is restricting to pre-index information.
    sampled = persons.sample(frac=0.08, random_state=int(rng.integers(1, 1_000_000)))
    for _, row in sampled.iterrows():
        rows.append(
            {
                "person_id": int(row["person_id"]),
                "dimension": "diagnosis",
                "code": "DX_POST_INDEX_OUTCOME_RELATED",
                "code_date": row["index_date"] + pd.Timedelta(days=int(rng.integers(1, 120))),
                "count": 1,
            }
        )
    return pd.DataFrame(rows)


def summarize_synthetic_truth(synthetic: SyntheticData) -> pd.DataFrame:
    return pd.DataFrame([synthetic.truth])
