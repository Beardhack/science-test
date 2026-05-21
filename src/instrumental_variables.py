from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .propensity import balance_diagnostics


def screen_iv_candidates(df: pd.DataFrame, measured_covariates: Iterable[str]) -> pd.DataFrame:
    candidates = []
    if "clinician_preference" in df.columns:
        candidates.append(("clinician prescribing preference", "clinician_preference"))
    if "site_id" in df.columns:
        site_rates = df.groupby("site_id")["A"].mean()
        df = df.copy()
        df["facility_preference_score"] = df["site_id"].map(site_rates)
        candidates.append(("facility preference", "facility_preference_score"))
    rows = []
    for label, col in candidates:
        z = (df[col] >= df[col].median()).astype(int)
        tmp = df.copy()
        tmp["Z"] = z
        first_stage = float(tmp.loc[tmp["Z"].eq(1), "A"].mean() - tmp.loc[tmp["Z"].eq(0), "A"].mean())
        p = float(tmp["A"].mean())
        n = len(tmp)
        se = np.sqrt(max(p * (1 - p), 1e-6) * (1 / max((tmp["Z"] == 1).sum(), 1) + 1 / max((tmp["Z"] == 0).sum(), 1)))
        first_stage_f = float((first_stage / max(se, 1e-6)) ** 2)
        balance = balance_diagnostics(tmp, measured_covariates, treatment_col="Z", weight_col=None)
        max_smd = float(balance["abs_smd"].max()) if not balance.empty else 0.0
        if abs(first_stage) < 0.05:
            recommendation = "do not use"
        elif max_smd > 0.20:
            recommendation = "do not use"
        elif max_smd > 0.10:
            recommendation = "exploratory only"
        else:
            recommendation = "plausible with caveats"
        rows.append(
            {
                "candidate": label,
                "variable": col,
                "relevance_first_stage_difference": first_stage,
                "first_stage_f_statistic_approx": first_stage_f,
                "max_covariate_smd_by_instrument": max_smd,
                "exclusion_restriction_concerns": "Preference may affect monitoring, adherence support, or coding beyond drug choice.",
                "independence_concerns": "Preference can cluster by site, patient mix, formulary, and calendar time.",
                "monotonicity_concerns": "Some clinicians may prefer one class only for selected phenotypes.",
                "recommendation": recommendation,
                "effect_estimation_status": (
                    "not_performed; screening only" if recommendation != "plausible with caveats" else "not_performed by default; requires pre-specified protocol sign-off"
                ),
            }
        )
    if not rows:
        rows.append(
            {
                "candidate": "none_available",
                "variable": "",
                "relevance_first_stage_difference": np.nan,
                "first_stage_f_statistic_approx": np.nan,
                "max_covariate_smd_by_instrument": np.nan,
                "exclusion_restriction_concerns": "No candidate variable available.",
                "independence_concerns": "No candidate variable available.",
                "monotonicity_concerns": "No candidate variable available.",
                "recommendation": "do not use",
                "effect_estimation_status": "not_performed",
            }
        )
    return pd.DataFrame(rows)
