# Analysis Plan

1. Build a new-user active-comparator cohort using only pre-index baseline information.
2. Estimate conventional propensity scores using investigator-specified measured confounders.
3. Estimate proxy-rich and hdPS-like propensity scores using pre-index code history.
4. Apply stabilized IPTW, truncation, and overlap weighting.
5. Check weighted balance, positivity, effective sample size, missingness, and attrition.
6. Estimate 365-day risk difference, risk ratio, and a weighted incidence-rate proxy for the hazard ratio.
7. Run negative-control outcome analysis for bias detection.
8. Quantify unmeasured-confounding sensitivity with E-value-style and binary-confounder tipping-point analyses.
9. Demonstrate validation calibration and IV screening logic on synthetic data.
10. Run simulations varying latent-confounding strength, proxy quality, negative-control validity stressors, positivity, outcome rarity, and sample size.
