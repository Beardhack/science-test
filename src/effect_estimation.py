from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w)
    if mask.sum() == 0 or w[mask].sum() <= 0:
        return float("nan")
    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))


def estimate_effects(
    df: pd.DataFrame,
    treatment_col: str = "A",
    outcome_col: str = "event_365",
    time_col: str = "followup_days",
    weight_col: str | None = None,
    label: str = "analysis",
) -> dict[str, float | str]:
    weights = pd.Series(1.0, index=df.index) if weight_col is None else df[weight_col].astype(float)
    treated = df[treatment_col].eq(1)
    comp = df[treatment_col].eq(0)
    risk1 = _weighted_mean(df.loc[treated, outcome_col], weights.loc[treated])
    risk0 = _weighted_mean(df.loc[comp, outcome_col], weights.loc[comp])
    person_years1 = _weighted_mean(df.loc[treated, time_col] / 365.25, weights.loc[treated]) * float(weights.loc[treated].sum())
    person_years0 = _weighted_mean(df.loc[comp, time_col] / 365.25, weights.loc[comp]) * float(weights.loc[comp].sum())
    events1 = float(np.sum(df.loc[treated, outcome_col] * weights.loc[treated]))
    events0 = float(np.sum(df.loc[comp, outcome_col] * weights.loc[comp]))
    rate1 = events1 / max(person_years1, 1e-9)
    rate0 = events0 / max(person_years0, 1e-9)
    rr = risk1 / risk0 if risk0 > 0 else float("nan")
    hr = rate1 / rate0 if rate0 > 0 else float("nan")
    return {
        "analysis": label,
        "weight_col": weight_col or "none",
        "risk_exposed": risk1,
        "risk_comparator": risk0,
        "risk_difference": risk1 - risk0,
        "risk_ratio": rr,
        "hazard_ratio_rate_proxy": hr,
        "events_exposed_weighted": events1,
        "events_comparator_weighted": events0,
        "n": float(len(df)),
    }


def bootstrap_effects(
    df: pd.DataFrame,
    weight_col: str | None,
    label: str,
    iterations: int = 100,
    seed: int = 20260520,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    if iterations <= 0:
        return pd.DataFrame()
    indices = np.arange(len(df))
    for i in range(iterations):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        sampled = df.iloc[sample_idx].reset_index(drop=True)
        try:
            row = estimate_effects(sampled, weight_col=weight_col, label=label)
            row["bootstrap_iteration"] = i
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def summarize_with_bootstrap(point: dict[str, float | str], boot: pd.DataFrame) -> dict[str, float | str]:
    out = dict(point)
    for measure in ["risk_difference", "risk_ratio", "hazard_ratio_rate_proxy"]:
        if measure in boot and boot[measure].notna().any():
            out[f"{measure}_ci_low"] = float(boot[measure].quantile(0.025))
            out[f"{measure}_ci_high"] = float(boot[measure].quantile(0.975))
        else:
            out[f"{measure}_ci_low"] = float("nan")
            out[f"{measure}_ci_high"] = float("nan")
    return out


def estimate_suite(
    df: pd.DataFrame,
    analyses: Iterable[tuple[str, str | None]],
    bootstrap_iterations: int = 0,
    seed: int = 20260520,
) -> pd.DataFrame:
    rows = []
    for i, (label, weight_col) in enumerate(analyses):
        point = estimate_effects(df, weight_col=weight_col, label=label)
        boot = bootstrap_effects(df, weight_col=weight_col, label=label, iterations=bootstrap_iterations, seed=seed + i)
        rows.append(summarize_with_bootstrap(point, boot))
    return pd.DataFrame(rows)
