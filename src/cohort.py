from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .data_model import age_threshold_from_config, baseline_days, comparator_label, exposure_label, follow_up_days, washout_days


def build_new_user_active_comparator_cohort(
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a mutually exclusive new-user active-comparator cohort."""
    persons = tables["person"].copy()
    drug_exposure = tables["drug_exposure"].copy()
    required_baseline = baseline_days(config)
    required_washout = washout_days(config)
    min_age = age_threshold_from_config(config)
    attrition_rows: list[dict[str, object]] = []

    def apply_step(df: pd.DataFrame, label: str, keep_mask: pd.Series) -> pd.DataFrame:
        before = len(df)
        out = df.loc[keep_mask].copy()
        attrition_rows.append({"step": label, "excluded": before - len(out), "remaining": len(out)})
        return out

    cohort = persons.copy()
    attrition_rows.append({"step": "Source population", "excluded": 0, "remaining": len(cohort)})
    cohort = apply_step(cohort, f"Age >= {min_age} years on index date", cohort["age"].ge(min_age).fillna(False))
    cohort = apply_step(cohort, "Evidence of type 2 diabetes before index date", cohort["evidence_t2d"].eq(1))
    cohort = apply_step(
        cohort,
        f"At least {required_baseline} days continuous enrollment before index date",
        cohort["enrollment_days_prior"].ge(required_baseline),
    )
    cohort = apply_step(
        cohort,
        f"No use of either study drug class in prior {required_washout} days",
        cohort["prior_study_drug_use"].eq(0),
    )
    cohort = apply_step(cohort, "No outcome event in prior 180 days", cohort["outcome_prior_180"].eq(0))
    cohort = apply_step(cohort, "Nonmissing age and sex", cohort["age"].notna() & cohort["sex_female"].notna())
    cohort = apply_step(cohort, "No data-quality violation making time zero ambiguous", cohort["data_quality_violation"].eq(0))

    exposure_name = exposure_label(config)
    comparator_name = comparator_label(config)
    index_rows = drug_exposure.loc[drug_exposure["is_index_exposure"].eq(1)].copy()
    index_rows = index_rows[index_rows["drug_class"].isin([exposure_name, comparator_name])]
    counts = index_rows.groupby("person_id")["drug_class"].nunique().rename("n_index_classes")
    first_index = index_rows.sort_values(["person_id", "drug_start_date"]).drop_duplicates("person_id")
    first_index = first_index.merge(counts, on="person_id", how="left")
    cohort = cohort.merge(
        first_index[["person_id", "drug_class", "treatment_role", "drug_start_date", "n_index_classes"]],
        on="person_id",
        how="left",
    )
    cohort = apply_step(cohort, "Qualifying index dispensing or prescription", cohort["drug_start_date"].notna())
    cohort = apply_step(cohort, "Mutually exclusive exposure/comparator assignment at time zero", cohort["n_index_classes"].eq(1))
    cohort["A"] = cohort["drug_class"].eq(exposure_name).astype(int)
    cohort["treatment_label"] = cohort["A"].map({1: exposure_name, 0: comparator_name})
    cohort["time_zero"] = pd.to_datetime(cohort["drug_start_date"])
    cohort["follow_up_end"] = cohort["time_zero"] + pd.to_timedelta(follow_up_days(config), unit="D")
    cohort["covariate_window_start"] = cohort["time_zero"] - pd.to_timedelta(required_baseline, unit="D")
    cohort["covariate_window_end"] = cohort["time_zero"] - pd.to_timedelta(1, unit="D")
    cohort["clear_time_zero"] = cohort["time_zero"].notna()
    cohort = apply_step(cohort, "Clear time zero", cohort["clear_time_zero"])

    attrition = pd.DataFrame(attrition_rows)
    return cohort.reset_index(drop=True), attrition


def assert_valid_cohort_timing(cohort: pd.DataFrame) -> None:
    if cohort["time_zero"].isna().any():
        raise AssertionError("All cohort members must have a nonmissing time zero.")
    if (cohort["covariate_window_end"] >= cohort["time_zero"]).any():
        raise AssertionError("Baseline covariate windows must end before time zero.")
    if cohort.groupby("person_id")["A"].nunique().max() > 1:
        raise AssertionError("Treatment assignment is not mutually exclusive.")
    if cohort["person_id"].duplicated().any():
        raise AssertionError("Each person must enter the cohort once.")


def attrition_markdown(attrition: pd.DataFrame) -> str:
    lines = ["| Step | Excluded | Remaining |", "|---|---:|---:|"]
    for _, row in attrition.iterrows():
        lines.append(f"| {row['step']} | {int(row['excluded'])} | {int(row['remaining'])} |")
    return "\n".join(lines)
