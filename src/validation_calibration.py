from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .effect_estimation import estimate_effects
from .propensity import estimate_propensity_scores


def run_validation_substudy_example(
    df: pd.DataFrame,
    measured_covariates: Iterable[str],
    validation_fraction: float = 0.25,
    seed: int = 20260520,
) -> pd.DataFrame:
    """Synthetic validation-substudy example where latent U is observed in a subset."""
    if "latent_u" not in df.columns:
        return pd.DataFrame(
            [
                {
                    "method": "validation_substudy",
                    "status": "not_available",
                    "interpretation": "No validation variable for the latent confounder was available.",
                }
            ]
        )
    rng = np.random.default_rng(seed)
    validation_ids = rng.choice(df.index.to_numpy(), size=max(20, int(len(df) * validation_fraction)), replace=False)
    subset = df.loc[validation_ids].copy()
    conventional = estimate_propensity_scores(subset, measured_covariates)
    oracle_covars = list(measured_covariates) + ["latent_u"]
    oracle = estimate_propensity_scores(subset, oracle_covars)
    conventional_rr = float(
        estimate_effects(conventional.data, weight_col="weight_stabilized_iptw", label="validation_conventional")[
            "risk_ratio"
        ]
    )
    oracle_rr = float(estimate_effects(oracle.data, weight_col="weight_stabilized_iptw", label="validation_oracle")["risk_ratio"])
    log_delta = float(np.log(max(oracle_rr, 1e-9)) - np.log(max(conventional_rr, 1e-9)))
    return pd.DataFrame(
        [
            {
                "method": "propensity_score_calibration_demo",
                "status": "synthetic_example",
                "validation_n": len(subset),
                "conventional_rr_subset": conventional_rr,
                "oracle_rr_subset": oracle_rr,
                "log_rr_calibration_delta": log_delta,
                "interpretation": "Apply this delta to the full-data log RR only if the validation subset is representative and U is measured comparably.",
            },
            {
                "method": "external_adjustment_stub",
                "status": "ready_for_inputs",
                "validation_n": 0,
                "conventional_rr_subset": np.nan,
                "oracle_rr_subset": np.nan,
                "log_rr_calibration_delta": np.nan,
                "interpretation": "Supply external prevalence and U-outcome association estimates, then route them through the QBA grid.",
            },
            {
                "method": "two_stage_imputation_stub",
                "status": "ready_for_inputs",
                "validation_n": 0,
                "conventional_rr_subset": np.nan,
                "oracle_rr_subset": np.nan,
                "log_rr_calibration_delta": np.nan,
                "interpretation": "Use when lab/EHR/chart-review data measure otherwise unavailable confounders in a subset.",
            },
        ]
    )
