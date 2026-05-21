from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .data_model import comparator_label, exposure_label, nested_get, rq


def build_gh_pages_site(config: Mapping[str, Any], reports_dir: str | Path = "reports", docs_dir: str | Path = "docs") -> None:
    reports = Path(reports_dir)
    docs = Path(docs_dir)
    assets = docs / "assets"
    docs.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    for src in [
        reports / "figures" / "propensity_overlap.png",
        reports / "figures" / "balance_smd.png",
        reports / "figures" / "effect_estimates.png",
        reports / "tables" / "qba_tipping_point_plot.png",
    ]:
        if src.exists():
            shutil.copy2(src, assets / src.name)

    effects = _read_csv(reports / "tables" / "effect_estimates.csv")
    attrition = _read_csv(reports / "tables" / "attrition.csv")
    flags = _read_csv(reports / "tables" / "analysis_fragility_flags.csv")
    nc = _read_csv(reports / "tables" / "negative_control_results.csv")
    qba = _read_csv(reports / "tables" / "qba_results.csv")
    simulation = _read_csv(reports / "tables" / "simulation_summary.csv")
    iv = _read_csv(reports / "tables" / "iv_candidate_screening.csv")
    alternatives = _read_csv(reports / "tables" / "design_alternatives.csv")

    research = rq(config)
    question = research["plain_language_question"]
    exposure = exposure_label(config)
    comparator = comparator_label(config)
    outcome = nested_get(config, ["research_question", "outcome", "name"], "outcome")
    estimand = nested_get(config, ["research_question", "estimand", "primary_estimand"], "configured estimand")
    grade, grade_reason = _extract_grade(reports / "results_report.md")

    primary = _effect_row(effects, "conventional_ps_stabilized_iptw")
    hdps = _effect_row(effects, "hdps_proxy_stabilized_iptw")
    oracle = _effect_row(effects, "oracle_adjusts_true_u")
    final_n = int(attrition.iloc[-1]["remaining"]) if not attrition.empty else 0
    nc_adjusted = nc[nc["analysis"].astype(str).str.contains("adjusted", na=False)] if not nc.empty else pd.DataFrame()
    nc_text = "Not available"
    if not nc_adjusted.empty:
        nc_row = nc_adjusted.iloc[0]
        nc_text = f"Adjusted negative-control RR {float(nc_row['risk_ratio']):.2f}; residual association detected: {nc_row['residual_association_detected']}"
    tipping_count = int(qba["moves_to_clinically_unimportant"].sum()) if "moves_to_clinically_unimportant" in qba else 0
    max_false_reassurance = float(simulation["false_reassurance_rate"].max()) if "false_reassurance_rate" in simulation else float("nan")

    (docs / "index.html").write_text(
        _page(
            title="Science Test Results",
            active="results",
            body=f"""
<section class="hero">
  <div>
    <p class="eyebrow">Observational Drug Safety / Effectiveness Workbench</p>
    <h1>SGLT2 vs GLP-1: 1-Year Heart-Failure Hospitalization</h1>
    <p>{html.escape(question)}</p>
  </div>
  <div class="grade">
    <span>Residual Confounding Credibility</span>
    <strong>{html.escape(grade)}</strong>
    <p>{html.escape(grade_reason)}</p>
  </div>
</section>

<section class="metrics">
  {_metric("Cohort", f"{final_n:,}", "eligible synthetic new users after exclusions")}
  {_metric("Primary RR", _fmt(primary, "risk_ratio"), "conventional measured-confounder stabilized IPTW")}
  {_metric("hdPS / Proxy RR", _fmt(hdps, "risk_ratio"), "pre-index proxy-rich adjustment")}
  {_metric("Oracle RR", _fmt(oracle, "risk_ratio"), "synthetic-only adjustment for true latent U")}
</section>

<section class="grid two">
  <article>
    <h2>Effect Estimates</h2>
    <img src="assets/effect_estimates.png" alt="Effect estimates by adjustment strategy">
  </article>
  <article>
    <h2>Balance</h2>
    <img src="assets/balance_smd.png" alt="Weighted balance standardized mean differences">
  </article>
</section>

<section class="grid two">
  <article>
    <h2>Overlap</h2>
    <img src="assets/propensity_overlap.png" alt="Propensity score overlap">
  </article>
  <article>
    <h2>QBA Tipping Grid</h2>
    <img src="assets/qba_tipping_point_plot.png" alt="Quantitative bias analysis tipping point grid">
  </article>
</section>

<section class="panel">
  <h2>What The Synthetic Demo Found</h2>
  <div class="summary-list">
    <p><strong>Primary analysis:</strong> conventional measured-confounder IPTW estimated RR {_fmt(primary, "risk_ratio")} and RD {_fmt(primary, "risk_difference")}.</p>
    <p><strong>Proxy-rich analysis:</strong> hdPS/proxy IPTW shifted the RR to {_fmt(hdps, "risk_ratio")}, moving toward the oracle estimate but not proving latent confounding was removed.</p>
    <p><strong>Negative control:</strong> {html.escape(nc_text)}.</p>
    <p><strong>QBA:</strong> {tipping_count} configured grid scenarios moved the estimate into the clinically unimportant range.</p>
    <p><strong>Simulation:</strong> maximum false-reassurance rate across configured stress scenarios was {_nan_fmt(max_false_reassurance)}.</p>
  </div>
</section>

<section class="panel compact">
  <h2>Public Interpretation</h2>
  <p>This is a reproducible methods workbench, not a real-data treatment recommendation. The demo uses synthetic data with known latent confounding to show where a standard propensity-score design can look healthy while still being biased.</p>
  <a class="button" href="methods.html">Methods, assumptions, and review needs</a>
</section>
""",
        ),
        encoding="utf-8",
    )

    (docs / "methods.html").write_text(
        _page(
            title="Methods and Assumptions",
            active="methods",
            body=f"""
<section class="hero slim">
  <div>
    <p class="eyebrow">Target Trial and Causal Assumptions</p>
    <h1>What Must Be True For The Result To Be Credible</h1>
    <p>Exposure: {html.escape(exposure)}. Comparator: {html.escape(comparator)}. Outcome: {html.escape(outcome)}.</p>
  </div>
</section>

<section class="panel">
  <h2>Estimand</h2>
  <p>{html.escape(estimand)}</p>
</section>

<section class="grid two">
  <article>
    <h2>Design Guardrails</h2>
    <ul>
      <li>New-user active-comparator design.</li>
      <li>Clear time zero at first qualifying dispensing or prescription.</li>
      <li>Baseline covariates restricted to pre-index information.</li>
      <li>Continuous baseline enrollment and washout checks.</li>
      <li>Balance, overlap, and effective-sample-size diagnostics.</li>
    </ul>
  </article>
  <article>
    <h2>Unmeasured-Confounding Workbench</h2>
    <ul>
      <li>Proxy-rich and hdPS-like adjustment.</li>
      <li>Negative-control outcome framework.</li>
      <li>Binary-confounder tipping-point QBA.</li>
      <li>Validation-substudy and external-calibration examples.</li>
      <li>IV candidate screening without automatic endorsement.</li>
    </ul>
  </article>
</section>

<section class="panel">
  <h2>Fragility Flags</h2>
  {_html_table(flags)}
</section>

<section class="panel">
  <h2>IV Screening</h2>
  {_html_table(iv)}
</section>

<section class="panel">
  <h2>Alternative Designs</h2>
  {_html_table(alternatives)}
</section>

<section class="panel">
  <h2>Requires Clinical Review Before Real-Data Use</h2>
  <ul>
    <li>Drug and outcome concept sets.</li>
    <li>Incident heart-failure hospitalization definition and inpatient-position rules.</li>
    <li>Negative-control candidates and rejection reasons.</li>
    <li>QBA scenario values for glycemic control, eGFR, BMI, smoking, frailty, SES, adherence, and prescribing preference.</li>
    <li>Death, competing-risk, censoring, treatment switching, and as-treated definitions.</li>
  </ul>
</section>

<section class="panel compact">
  <h2>Untestable Assumptions</h2>
  <p>Conditional exchangeability, proxy validity, negative-control validity, QBA scenario credibility, validation-substudy representativeness, and IV exclusion/independence/monotonicity remain untestable or only partly testable.</p>
  <a class="button" href="index.html">Back to results</a>
</section>
""",
        ),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _effect_row(effects: pd.DataFrame, analysis: str) -> pd.Series:
    rows = effects[effects["analysis"].eq(analysis)] if not effects.empty else pd.DataFrame()
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def _fmt(row: pd.Series, column: str) -> str:
    if row.empty or column not in row or pd.isna(row[column]):
        return "NA"
    return f"{float(row[column]):.2f}"


def _nan_fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2f}"


def _extract_grade(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "Not available", "Report has not been generated."
    text = path.read_text(encoding="utf-8")
    grade = "Not available"
    reason = "See the full report for details."
    for line in text.splitlines():
        if line.startswith("Grade:"):
            grade = line.replace("Grade:", "").replace("*", "").strip().strip(".")
        if line.startswith("Reason:"):
            reason = line.replace("Reason:", "").strip()
    return grade, reason


def _metric(label: str, value: str, note: str) -> str:
    return f"""<article class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong><p>{html.escape(note)}</p></article>"""


def _html_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    show = df.head(max_rows).copy()
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in show.columns)
    rows = []
    for _, row in show.iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(_cell(value))}</td>" for value in row) + "</tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _page(title: str, active: str, body: str) -> str:
    results_active = "active" if active == "results" else ""
    methods_active = "active" if active == "methods" else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #1b1f23;
      --muted: #5f6b7a;
      --line: #d8dee7;
      --paper: #f6f8fb;
      --panel: #ffffff;
      --blue: #245f9f;
      --green: #2e7d5b;
      --gold: #9a6a12;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.5;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px clamp(18px, 4vw, 52px);
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    .brand {{ font-weight: 700; color: var(--blue); }}
    nav a {{
      color: var(--muted);
      text-decoration: none;
      margin-left: 18px;
      font-weight: 600;
      font-size: 14px;
    }}
    nav a.active {{ color: var(--ink); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px clamp(18px, 4vw, 42px) 52px; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 24px;
      align-items: stretch;
      padding: 30px 0 18px;
    }}
    .hero.slim {{ grid-template-columns: 1fr; }}
    h1 {{ font-size: clamp(34px, 5vw, 58px); line-height: 1.02; margin: 8px 0 14px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    p {{ margin: 0 0 12px; color: var(--muted); }}
    .eyebrow {{ color: var(--green); font-weight: 700; text-transform: uppercase; font-size: 12px; letter-spacing: 0; }}
    .grade, .metric, article, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .grade {{ padding: 22px; }}
    .grade span, .metric span {{ display: block; color: var(--muted); font-size: 13px; font-weight: 700; }}
    .grade strong {{ display: block; font-size: 42px; margin: 8px 0; color: var(--gold); }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 18px 0 26px; }}
    .metric {{ padding: 18px; min-height: 130px; }}
    .metric strong {{ display: block; font-size: 34px; margin: 8px 0; color: var(--blue); }}
    .grid {{ display: grid; gap: 18px; margin: 18px 0; }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    article, .panel {{ padding: 18px; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid var(--line); }}
    .panel {{ margin: 18px 0; }}
    .compact {{ max-width: 860px; }}
    .summary-list p {{ border-top: 1px solid var(--line); padding-top: 12px; }}
    .button {{
      display: inline-block;
      padding: 10px 14px;
      border-radius: 6px;
      background: var(--blue);
      color: #fff;
      text-decoration: none;
      font-weight: 700;
    }}
    ul {{ margin: 0; padding-left: 20px; color: var(--muted); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); }}
    footer {{ padding: 22px clamp(18px, 4vw, 52px); color: var(--muted); border-top: 1px solid var(--line); }}
    @media (max-width: 860px) {{
      .hero, .grid.two, .metrics {{ grid-template-columns: 1fr; }}
      nav a {{ margin-left: 10px; }}
      h1 {{ font-size: 36px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">science-test</div>
    <nav>
      <a class="{results_active}" href="index.html">Results</a>
      <a class="{methods_active}" href="methods.html">Methods</a>
      <a href="https://github.com/Beardhack/science-test">GitHub</a>
    </nav>
  </header>
  <main>
    {body}
  </main>
  <footer>
    Synthetic demonstration only. Do not interpret as real-world comparative effectiveness evidence.
  </footer>
</body>
</html>
"""
