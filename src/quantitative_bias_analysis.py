from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data_model import nested_get
from .simple_plots import save_line_plot


def e_value(rr: float) -> float:
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    rr_for_calc = rr if rr >= 1.0 else 1.0 / rr
    return float(rr_for_calc + np.sqrt(rr_for_calc * (rr_for_calc - 1.0)))


def binary_confounder_bias_factor(p_exposed: float, p_comparator: float, rr_u_outcome: float) -> float:
    numerator = p_exposed * rr_u_outcome + (1.0 - p_exposed)
    denominator = p_comparator * rr_u_outcome + (1.0 - p_comparator)
    return float(numerator / denominator)


def prevalence_odds_ratio(p_exposed: float, p_comparator: float) -> float:
    p1 = np.clip(p_exposed, 1e-6, 1.0 - 1e-6)
    p0 = np.clip(p_comparator, 1e-6, 1.0 - 1e-6)
    return float((p1 / (1.0 - p1)) / (p0 / (1.0 - p0)))


def run_qba(
    observed_rr: float,
    config: Mapping[str, Any],
    output_dir: str | Path,
    clinically_unimportant: tuple[float, float] = (0.95, 1.05),
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    qba_cfg = nested_get(config, ["analysis", "qba"], {})
    p1_grid = qba_cfg.get("u_prevalence_exposed_grid", [0.1, 0.2, 0.3, 0.4, 0.5])
    p0_grid = qba_cfg.get("u_prevalence_comparator_grid", [0.1, 0.2, 0.3, 0.4, 0.5])
    rru_grid = qba_cfg.get("u_outcome_rr_grid", [1.1, 1.25, 1.5, 2.0, 3.0])
    rows = []
    target_low, target_high = clinically_unimportant
    for p1 in p1_grid:
        for p0 in p0_grid:
            for rru in rru_grid:
                bf = binary_confounder_bias_factor(float(p1), float(p0), float(rru))
                adjusted_rr = float(observed_rr / bf)
                moved_to_null = (observed_rr < 1.0 and adjusted_rr >= 1.0) or (observed_rr > 1.0 and adjusted_rr <= 1.0)
                moved_to_clinically_unimportant = target_low <= adjusted_rr <= target_high
                rows.append(
                    {
                        "observed_rr": observed_rr,
                        "u_prevalence_exposed": p1,
                        "u_prevalence_comparator": p0,
                        "u_treatment_prevalence_or": prevalence_odds_ratio(float(p1), float(p0)),
                        "u_outcome_rr": rru,
                        "bias_factor": bf,
                        "bias_adjusted_rr": adjusted_rr,
                        "moves_to_null": moved_to_null,
                        "moves_to_clinically_unimportant": moved_to_clinically_unimportant,
                    }
                )
    results = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "qba_results.csv", index=False)
    _plot_tipping(results, output_dir / "qba_tipping_point_plot.png")
    tipping = results.loc[results["moves_to_clinically_unimportant"]].copy()
    if tipping.empty:
        summary = {
            "e_value": e_value(observed_rr),
            "tipping_status": "not_found_on_grid",
            "plain_language": "No configured binary-confounder scenario moved the estimate into the clinically unimportant range.",
        }
    else:
        tipping = tipping.sort_values(["u_outcome_rr", "u_treatment_prevalence_or"], key=lambda s: s.abs())
        first = tipping.iloc[0]
        summary = {
            "e_value": e_value(observed_rr),
            "tipping_status": "found_on_grid",
            "plain_language": (
                "On the configured grid, a latent factor with prevalence "
                f"{first['u_prevalence_exposed']:.2f} in the exposure group and "
                f"{first['u_prevalence_comparator']:.2f} in the comparator group, plus an outcome RR of "
                f"{first['u_outcome_rr']:.2f}, would move the estimate into the clinically unimportant range."
            ),
        }
    return results, summary


def _plot_tipping(results: pd.DataFrame, path: Path) -> None:
    pivot = (
        results.groupby(["u_outcome_rr", "u_treatment_prevalence_or"])["bias_adjusted_rr"]
        .median()
        .reset_index()
        .sort_values("u_treatment_prevalence_or")
    )
    series = []
    for rru, group in pivot.groupby("u_outcome_rr"):
        series.append(
            (
                f"U-outcome RR {rru:g}",
                group["u_treatment_prevalence_or"].astype(float).tolist(),
                group["bias_adjusted_rr"].astype(float).tolist(),
            )
        )
    save_line_plot(
        path,
        series=series,
        title="QBA tipping-point grid",
        x_label="U-treatment prevalence odds ratio",
        y_label="Bias-adjusted risk ratio",
        hline=1.0,
    )


def scenario_templates(config: Mapping[str, Any]) -> pd.DataFrame:
    suspected = nested_get(config, ["research_question", "suspected_unmeasured_confounders"], [])
    rows = []
    for item in suspected:
        label = str(item)
        rows.append(
            {
                "suspected_confounder": label,
                "low_scenario": "p_exposed=0.20, p_comparator=0.30, U-outcome RR=1.25",
                "moderate_scenario": "p_exposed=0.20, p_comparator=0.40, U-outcome RR=1.75",
                "strong_scenario": "p_exposed=0.10, p_comparator=0.50, U-outcome RR=2.50",
                "review_note": "Replace these placeholders with literature, chart-review, lab-rich subset, or expert-elicited values.",
            }
        )
    return pd.DataFrame(rows)
