# Target Trial Specification

| Component | Emulation |
|---|---|
| Eligibility criteria | Age >= 40 years on index date; Evidence of type 2 diabetes before index date; At least 365 days continuous enrollment before index date; New user of either exposure or comparator drug class. Exclusions: Use of either study drug class in the 365 days before index date; Outcome event in the 180 days before index date, for incident-outcome analysis; Missing age or sex; Any data-quality violation that makes time zero ambiguous. |
| Treatment strategies | Initiate SGLT2 inhibitor initiation versus initiate GLP-1 receptor agonist initiation. |
| Assignment procedure being emulated | Active-comparator new-user design; treatment is observed, not randomized, so exchangeability is pursued by design and adjustment. |
| Time zero | First qualifying dispensing or prescription after baseline washout. |
| Follow-up | Treatment-policy follow-up through 365 days, event, death/censoring, or data end. |
| Outcome definition | Hospitalization for heart failure; Prefer inpatient diagnosis codes in primary or discharge position. Treat death as a competing event in sensitivity analyses if death data exist. |
| Causal contrast | Average treatment effect among eligible new users under a treatment-policy strategy; primary measures: 365-day risk difference, 365-day risk ratio, hazard ratio. |
| Analysis plan | Estimate propensity scores, apply IPTW/stabilized IPTW/truncation/overlap weights, check balance and positivity, estimate 365-day risks and time-to-event proxy HR, then stress-test residual confounding. |
| Known deviations from ideal randomized trial | Treatment is not randomized; adherence, treatment switching, death, censoring, unmeasured clinical severity, and clinician preference may remain imperfectly measured. |
