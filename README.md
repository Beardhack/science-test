# SGLT2 vs GLP-1 Heart-Failure Unmeasured-Confounding Workbench

This repository is a config-driven pharmacoepidemiology research workbench for:

> Among adults with type 2 diabetes initiating an SGLT2 inhibitor versus a GLP-1 receptor agonist, what is the effect on 1-year hospitalization for heart failure?

The source of truth is `config/research_question.yaml`. The default run uses synthetic claims/EHR-like data with a known latent confounder `U`, proxy variables, pre-index code history, negative-control outcomes, and clinician/site preference variables.

The workbench is designed to minimize, detect, quantify, and stress-test residual confounding. It does not claim unmeasured confounding is eliminated.

## Quick Start

Using an environment with the dependencies installed:

```bash
python run_analysis.py --config config/research_question.yaml --reports-dir reports
python -m unittest discover -s tests
```

With `make`:

```bash
make demo
make test
```

In this Codex workspace, the bundled runtime already has NumPy, Pandas, and Pillow:

```powershell
& 'C:\Users\locka\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_analysis.py --config config\research_question.yaml --reports-dir reports
& 'C:\Users\locka\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests
```

## What The Pipeline Produces

- `reports/protocol.md`
- `reports/target_trial_table.md`
- `reports/dag.md`
- `reports/analysis_plan.md`
- `reports/results_report.md`
- `reports/limitations_and_assumptions.md`
- `reports/tables/*.csv`
- `reports/figures/*.png`

Key generated tables include attrition, baseline characteristics, missingness, balance, positivity, effect estimates, hdPS-selected proxies, negative-control results, QBA results, IV screening, design alternatives, and simulation summaries.

## GitHub Pages

The pipeline also writes a compact two-page public summary to `docs/`:

- `docs/index.html`: executive results dashboard
- `docs/methods.html`: target-trial, diagnostics, assumptions, and review needs

To publish it, configure GitHub Pages for the repository to serve from the `docs/` folder on the `main` branch.

## Minimal Fields To Change For Another Question

Edit only `config/research_question.yaml` when changing the pharmacoepidemiologic question:

- `project_slug`
- `plain_language_question`
- `population.inclusion_criteria`
- `population.exclusion_criteria`
- `exposure.name`
- `exposure.drug_class_or_concept_set`
- `exposure.index_date_definition`
- `comparator.name`
- `comparator.drug_class_or_concept_set`
- `outcome.name`
- `outcome.primary_follow_up_days`
- `estimand.primary_estimand`
- `measured_confounder_domains`
- `suspected_unmeasured_confounders`
- optional negative controls and validation sources

Synthetic parameters and diagnostic thresholds can also be changed under `analysis`.

## Real-Data Adapter Expectations

Real data should be adapted into tables equivalent to:

- `person`
- `continuous_enrollment`
- `drug_exposure`
- `condition_occurrence`
- `procedure_occurrence`
- `visit_occurrence`
- `measurement`
- `death`
- `code_history`

All baseline covariates and hdPS features must be measured before `time_zero`. The tests include explicit guards against post-index adjustment.

## Residual-Confounding Position

This workbench operationalizes "solve unmeasured confounding" as:

1. Minimize confounding by target-trial design.
2. State the estimand explicitly.
3. Use rich measured and proxy adjustment.
4. Detect residual bias with negative controls and diagnostics.
5. Quantify required unmeasured-confounding strength with QBA.
6. Use validation subsets and instruments only with stated assumptions.
7. Stress-test the workflow on synthetic data with known latent `U`.
8. Avoid claiming unmeasured confounding has been eliminated.

## Clinical Review Needed Before Real Use

- Drug and outcome concept sets
- Incident outcome washout and inpatient-position rules
- Negative-control candidates
- Plausible QBA scenario values
- Validation-substudy representativeness
- IV assumptions for clinician/facility/formulary/calendar candidates
- Death and competing-risk handling
- As-treated censoring and treatment switching definitions
