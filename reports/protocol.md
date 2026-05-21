# Protocol

## Research Question
Among adults with type 2 diabetes initiating an SGLT2 inhibitor versus a GLP-1 receptor agonist, what is the effect on 1-year hospitalization for heart failure?

## Estimand
Primary: Average treatment effect among eligible new users under a treatment-policy strategy

Secondary: Overlap-weighted treatment effect among patients with clinical equipoise

## Data Source Assumption
The demo runs on synthetic claims/EHR-like data because no real data were supplied. Real data must provide the configured required tables or an adapter that produces the same analytic columns.

## Identification Position
The workflow minimizes and diagnoses unmeasured confounding; it does not claim to eliminate it. Any real-data causal claim requires explicit exchangeability, positivity, consistency, measurement, and censoring assumptions.
