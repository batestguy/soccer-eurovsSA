---
title: Bayesian World Cup Prediction
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
python_version: 3.12
---

# Bayesian World Cup Prediction

Eight-lens decision-support app for a retrospective pre-tournament replay of the
2026 World Cup. Serving is static and lightweight: the Space only loads
precomputed NetCDF/CSV/PNG artifacts and never runs MCMC or model fitting.

## Lenses

- Continental Strength: hierarchical panel spline trends, A/B differences, and
  90% bands at monthly, quarterly, or annual resolution
- Posterior forest plot for hierarchical effects
- 50,000-simulation Monte Carlo Oracle with uncertainty intervals
- `do()` What-If counterfactual simulation (not an estimated causal effect)
- Prior-predictive versus posterior comparison
- Causal DAG and identification explanation
- Four DAG assumption checks
- Ranking Dynamics: deterministic descriptive SES fits and a 12-month scenario
  forecast for top-5/top-10/top-20 composition

## Limitations

- The retained Continental Strength fit has 5 divergences, maximum R-hat 1.081,
  and minimum ESS 39.7; it is exploratory and structural/associational.
- Ranking Dynamics has no probabilistic uncertainty claim: its fitted curves and
  12-month extension are deterministic SES descriptions.
- Latent team strength prevents identification of a causal continental-champion
  effect from 22 World Cups. The app labels structural, associational, and
  counterfactual quantities accordingly.
