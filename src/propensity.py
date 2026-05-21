from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PropensityResult:
    data: pd.DataFrame
    coefficients: pd.Series
    covariates: list[str]
    diagnostics: dict[str, float | bool]
    balance: pd.DataFrame


def _design_matrix(df: pd.DataFrame, covariates: Iterable[str]) -> tuple[np.ndarray, list[str], dict[str, tuple[float, float]]]:
    cols = list(covariates)
    x = df[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        median = float(x[col].median()) if x[col].notna().any() else 0.0
        x[col] = x[col].fillna(median)
    means_sds: dict[str, tuple[float, float]] = {}
    arr_cols = []
    for col in cols:
        values = x[col].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std())
        if sd < 1e-8:
            sd = 1.0
        arr_cols.append((values - mean) / sd)
        means_sds[col] = (mean, sd)
    if arr_cols:
        x_arr = np.column_stack(arr_cols)
    else:
        x_arr = np.zeros((len(df), 0))
    x_arr = np.column_stack([np.ones(len(df)), x_arr])
    return x_arr, ["intercept"] + cols, means_sds


def _fit_logistic_ridge(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float = 0.1,
    max_iter: int = 80,
) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    penalty_matrix = np.eye(x.shape[1]) * penalty
    penalty_matrix[0, 0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(x @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = x.T @ (p - y) + penalty_matrix @ beta
        w = p * (1.0 - p)
        hess = (x.T * w) @ x + penalty_matrix
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return beta


def estimate_propensity_scores(
    df: pd.DataFrame,
    covariates: Iterable[str],
    treatment_col: str = "A",
    ridge_penalty: float = 0.1,
    max_iterations: int = 80,
    trim_lower: float = 0.025,
    trim_upper: float = 0.975,
    truncation_quantiles: tuple[float, float] = (0.01, 0.99),
) -> PropensityResult:
    covariate_list = list(covariates)
    x, names, _ = _design_matrix(df, covariate_list)
    y = df[treatment_col].to_numpy(dtype=float)
    beta = _fit_logistic_ridge(x, y, penalty=ridge_penalty, max_iter=max_iterations)
    ps = np.clip(1.0 / (1.0 + np.exp(-np.clip(x @ beta, -35, 35))), 0.001, 0.999)
    out = df.copy()
    out["propensity_score"] = ps
    p_treat = float(y.mean())
    out["weight_iptw"] = np.where(y == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    out["weight_stabilized_iptw"] = np.where(y == 1, p_treat / ps, (1.0 - p_treat) / (1.0 - ps))
    out["weight_overlap"] = np.where(y == 1, 1.0 - ps, ps)
    lower_q, upper_q = truncation_quantiles
    lo = float(out["weight_stabilized_iptw"].quantile(lower_q))
    hi = float(out["weight_stabilized_iptw"].quantile(upper_q))
    out["weight_truncated_stabilized_iptw"] = out["weight_stabilized_iptw"].clip(lo, hi)
    out["in_ps_trim"] = out["propensity_score"].between(trim_lower, trim_upper)
    diagnostics = positivity_diagnostics(out, treatment_col=treatment_col, weight_col="weight_stabilized_iptw")
    balance = balance_diagnostics(out, covariate_list, treatment_col=treatment_col, weight_col="weight_stabilized_iptw")
    coefficients = pd.Series(beta, index=names, name="coefficient")
    return PropensityResult(data=out, coefficients=coefficients, covariates=covariate_list, diagnostics=diagnostics, balance=balance)


def effective_sample_size(weights: pd.Series | np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if np.sum(w * w) <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / np.sum(w * w))


def positivity_diagnostics(
    df: pd.DataFrame,
    treatment_col: str = "A",
    weight_col: str = "weight_stabilized_iptw",
) -> dict[str, float | bool]:
    ps = df["propensity_score"]
    treated = df[treatment_col].eq(1)
    comp = df[treatment_col].eq(0)
    common_low = max(float(ps[treated].min()), float(ps[comp].min()))
    common_high = min(float(ps[treated].max()), float(ps[comp].max()))
    outside_common = float((~ps.between(common_low, common_high)).mean())
    w = df[weight_col]
    ess = effective_sample_size(w)
    return {
        "ps_min": float(ps.min()),
        "ps_p01": float(ps.quantile(0.01)),
        "ps_p50": float(ps.quantile(0.50)),
        "ps_p99": float(ps.quantile(0.99)),
        "ps_max": float(ps.max()),
        "common_support_low": common_low,
        "common_support_high": common_high,
        "fraction_outside_common_support": outside_common,
        "max_weight": float(w.max()),
        "mean_weight": float(w.mean()),
        "effective_sample_size": ess,
        "effective_sample_size_ratio": float(ess / max(len(df), 1)),
    }


def _weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return 0.0, 0.0
    mean = float(np.sum(w * x) / np.sum(w))
    var = float(np.sum(w * (x - mean) ** 2) / np.sum(w))
    return mean, var


def balance_diagnostics(
    df: pd.DataFrame,
    covariates: Iterable[str],
    treatment_col: str = "A",
    weight_col: str | None = None,
) -> pd.DataFrame:
    rows = []
    if weight_col is None:
        weights = pd.Series(1.0, index=df.index)
    else:
        weights = df[weight_col].astype(float)
    for col in covariates:
        x = pd.to_numeric(df[col], errors="coerce")
        mask_t = df[treatment_col].eq(1)
        mask_c = df[treatment_col].eq(0)
        m1, v1 = _weighted_mean_var(x[mask_t].to_numpy(dtype=float), weights[mask_t].to_numpy(dtype=float))
        m0, v0 = _weighted_mean_var(x[mask_c].to_numpy(dtype=float), weights[mask_c].to_numpy(dtype=float))
        denom = np.sqrt((v1 + v0) / 2.0)
        smd = 0.0 if denom < 1e-12 else (m1 - m0) / denom
        rows.append(
            {
                "variable": col,
                "treated_mean": m1,
                "comparator_mean": m0,
                "smd": float(smd),
                "abs_smd": float(abs(smd)),
                "weighted": weight_col is not None,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def max_abs_smd(balance: pd.DataFrame) -> float:
    if balance.empty:
        return 0.0
    return float(balance["abs_smd"].max())


def fragility_flags(
    diagnostics: dict[str, float | bool],
    balance: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, bool]:
    return {
        "major_positivity_violation": float(diagnostics["fraction_outside_common_support"]) > thresholds.get(
            "major_positivity_tail_fraction", 0.05
        ),
        "effective_sample_size_collapse": float(diagnostics["effective_sample_size_ratio"]) < thresholds.get(
            "minimum_ess_ratio", 0.35
        ),
        "residual_weighted_imbalance": max_abs_smd(balance) > thresholds.get("max_weighted_smd", 0.10),
    }
