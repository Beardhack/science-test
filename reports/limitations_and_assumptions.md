# Limitations and Assumptions

## Untestable Assumptions
- Conditional exchangeability after measured/proxy adjustment is not testable.
- Proxy and hdPS adjustment helps only if observed code history captures latent causes of treatment and outcome.
- Negative controls require no causal exposure effect and a bias structure close enough to the primary outcome.
- QBA scenarios are only as credible as the prevalence and association values supplied.
- Validation calibration assumes the validation subset is representative and measures the latent factor comparably.
- IV analyses require relevance, exclusion restriction, independence, and monotonicity; this workflow screens candidates but does not automatically endorse them.

## Operational Limits
The demo uses synthetic data with known latent U. Real-data adapters, concept-set review, outcome validation, missing-data handling, censoring models, and death competing-risk analyses require domain and data-owner review.
