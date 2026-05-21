from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def build_hdps_features(
    cohort: pd.DataFrame,
    code_history: pd.DataFrame,
    top_k: int = 30,
    min_prevalence: float = 0.01,
    max_prevalence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create and rank hdPS-like pre-index code proxies.

    The implementation follows the spirit of hdPS: scan code dimensions, form
    prevalent pre-index binary features, rank on treatment association, outcome
    association, and prevalence, then add the top-ranked proxies to the PS model.
    """
    if code_history.empty:
        return cohort[["person_id"]].copy(), pd.DataFrame()

    index_dates = cohort[["person_id", "time_zero", "A", "event_365"]].copy()
    codes = code_history.merge(index_dates, on="person_id", how="inner")
    codes["code_date"] = pd.to_datetime(codes["code_date"])
    codes = codes[codes["code_date"] < codes["time_zero"]].copy()
    if codes.empty:
        return cohort[["person_id"]].copy(), pd.DataFrame()
    codes["days_before_index"] = (pd.to_datetime(codes["time_zero"]) - codes["code_date"]).dt.days

    codes["feature"] = (
        "hdps_"
        + codes["dimension"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
        + "__"
        + codes["code"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    wide = (
        codes.assign(value=1)
        .pivot_table(index="person_id", columns="feature", values="value", aggfunc="max", fill_value=0)
        .reset_index()
    )
    features = cohort[["person_id", "A", "event_365"]].merge(wide, on="person_id", how="left").fillna(0)
    candidate_cols = [col for col in features.columns if col.startswith("hdps_")]
    selected_rows: list[dict[str, object]] = []
    for col in candidate_cols:
        x = features[col].astype(float)
        prevalence = float(x.mean())
        if prevalence < min_prevalence or prevalence > max_prevalence:
            continue
        p_a1 = float(x[features["A"].eq(1)].mean())
        p_a0 = float(x[features["A"].eq(0)].mean())
        p_y1 = float(x[features["event_365"].eq(1)].mean()) if features["event_365"].sum() else 0.0
        p_y0 = float(x[features["event_365"].eq(0)].mean())
        exposure_assoc = abs(p_a1 - p_a0)
        outcome_assoc = abs(p_y1 - p_y0)
        prevalence_score = prevalence * (1.0 - prevalence)
        bias_potential = prevalence_score * (0.10 + exposure_assoc) * (0.10 + outcome_assoc)
        max_code_date = codes.loc[codes["feature"].eq(col), "code_date"].max()
        min_days_before_index = int(codes.loc[codes["feature"].eq(col), "days_before_index"].min())
        selected_rows.append(
            {
                "feature": col,
                "dimension": col.split("__", 1)[0].replace("hdps_", ""),
                "prevalence": prevalence,
                "exposure_association": exposure_assoc,
                "outcome_association": outcome_assoc,
                "bias_potential": bias_potential,
                "max_code_date": max_code_date,
                "min_days_before_index": min_days_before_index,
            }
        )
    ranking = pd.DataFrame(selected_rows)
    if ranking.empty:
        return cohort[["person_id"]].copy(), ranking
    ranking = ranking.sort_values("bias_potential", ascending=False).head(top_k).reset_index(drop=True)
    feature_frame = cohort[["person_id"]].merge(wide[["person_id"] + ranking["feature"].tolist()], on="person_id", how="left").fillna(0)
    return feature_frame, ranking


def append_hdps_to_cohort(cohort: pd.DataFrame, hdps_features: pd.DataFrame) -> pd.DataFrame:
    out = cohort.merge(hdps_features, on="person_id", how="left")
    hdps_cols = [col for col in out.columns if col.startswith("hdps_")]
    out[hdps_cols] = out[hdps_cols].fillna(0).astype(int)
    return out


def assert_hdps_pre_index(selected_features: pd.DataFrame, cohort: pd.DataFrame) -> None:
    if selected_features.empty:
        return
    if "min_days_before_index" in selected_features and (selected_features["min_days_before_index"] <= 0).any():
        raise AssertionError("At least one selected hdPS feature uses post-index information.")
    if selected_features["feature"].astype(str).str.contains("post_index", case=False, regex=False).any():
        raise AssertionError("Post-index-looking code labels must not enter selected hdPS features.")


def hdps_warning() -> str:
    return (
        "Proxy and hdPS adjustment can reduce residual confounding only if observed pre-index "
        "code history captures information about unmeasured causes of treatment and outcome. "
        "It is not proof that latent confounding has been removed."
    )
