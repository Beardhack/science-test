from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CORE_BASELINE_COVARIATES = [
    "age",
    "sex_female",
    "race_black",
    "race_hispanic",
    "race_other",
    "calendar_year_centered",
    "calendar_quarter",
    "diabetes_duration_years",
    "cvd",
    "prior_hf",
    "ckd",
    "hypertension",
    "obesity",
    "bmi",
    "utilization_visits",
    "emergency_visits",
    "inpatient_visits",
    "medication_count",
    "lab_test_count",
]

PROXY_COVARIATES = [
    "smoking_proxy",
    "frailty_proxy",
    "socioeconomic_risk_proxy",
    "egfr_proxy",
    "a1c_proxy",
]

OPTIONAL_SYSTEM_COVARIATES = [
    "site_id",
]


def available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def baseline_covariate_columns(
    cohort: pd.DataFrame,
    include_proxy: bool = False,
    include_system: bool = True,
    include_oracle_u: bool = False,
) -> list[str]:
    cols = available_columns(cohort, CORE_BASELINE_COVARIATES)
    if include_proxy:
        cols.extend(available_columns(cohort, PROXY_COVARIATES))
    if include_system:
        cols.extend(available_columns(cohort, OPTIONAL_SYSTEM_COVARIATES))
    if include_oracle_u and "latent_u" in cohort.columns:
        cols.append("latent_u")
    return [col for col in dict.fromkeys(cols) if not is_post_index_column(col)]


def is_post_index_column(column: str) -> bool:
    lowered = column.lower()
    return lowered.startswith("post_index") or lowered.endswith("_post") or "post_index" in lowered


def assert_no_post_index_covariates(cohort: pd.DataFrame, covariate_columns: Iterable[str]) -> None:
    bad = [col for col in covariate_columns if is_post_index_column(col)]
    if bad:
        raise AssertionError(f"Post-index columns selected for baseline adjustment: {bad}")
    if "covariate_window_end" in cohort.columns and "time_zero" in cohort.columns:
        if (pd.to_datetime(cohort["covariate_window_end"]) >= pd.to_datetime(cohort["time_zero"])).any():
            raise AssertionError("Covariate window end must precede time zero for every patient.")


def missingness_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in columns:
        missing = int(df[col].isna().sum())
        rows.append({"variable": col, "missing_n": missing, "missing_percent": 100.0 * missing / max(n, 1)})
    return pd.DataFrame(rows)


def _smd(x_treated: pd.Series, x_comp: pd.Series) -> float:
    m1 = float(x_treated.mean())
    m0 = float(x_comp.mean())
    v1 = float(x_treated.var(ddof=0))
    v0 = float(x_comp.var(ddof=0))
    denom = np.sqrt((v1 + v0) / 2.0)
    if denom <= 1e-12:
        return 0.0
    return (m1 - m0) / denom


def baseline_table(df: pd.DataFrame, covariate_columns: Iterable[str], treatment_col: str = "A") -> pd.DataFrame:
    rows = []
    for col in covariate_columns:
        treated = pd.to_numeric(df.loc[df[treatment_col].eq(1), col], errors="coerce")
        comp = pd.to_numeric(df.loc[df[treatment_col].eq(0), col], errors="coerce")
        rows.append(
            {
                "variable": col,
                "treated_mean": treated.mean(),
                "comparator_mean": comp.mean(),
                "smd": _smd(treated.fillna(treated.mean()), comp.fillna(comp.mean())),
            }
        )
    return pd.DataFrame(rows).sort_values("smd", key=lambda s: s.abs(), ascending=False)


def table_to_markdown(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 3) -> str:
    show = df if max_rows is None else df.head(max_rows)
    if show.empty:
        return "_No rows._"
    fmt = show.copy()
    for col in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[col]):
            fmt[col] = fmt[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    headers = [str(col) for col in fmt.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in fmt.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in fmt.columns) + " |")
    return "\n".join(lines)
