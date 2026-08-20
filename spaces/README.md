---
title: Bayesian World Cup Prediction
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
python_version: 3.11
---

# Bayesian World Cup Prediction

**Live demo:** https://bayesian-world-cup-prediction.onrender.com

**Source:** https://github.com/batestguy/soccer-eurovsSA

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

## Release

This bundle is deployed on Render Free with Python 3.11.11. Render runs
`python spaces/app.py`, binds the assigned `$PORT` on `0.0.0.0`, and serves only
the precomputed files in `spaces/data/`. It never runs MCMC or model fitting.

The startup dependency `requests==2.32.3` is pinned because Gradio 5.23.1 imports
`requests` through its CLI module during application startup. The public release
was fixed in commit `d6db8b7`. Documentation is maintained on `main` and deploys
automatically with the service.

For deployment commands and failure diagnosis, see the repository
[`docs/DEPLOYMENT.md`](https://github.com/batestguy/soccer-eurovsSA/blob/main/docs/DEPLOYMENT.md).

## Limitations

- The retained Continental Strength fit has 5 divergences, maximum R-hat 1.081,
  and minimum ESS 39.7; it is exploratory and structural/associational.
- Ranking Dynamics has no probabilistic uncertainty claim: its fitted curves and
  12-month extension are deterministic SES descriptions.
- Latent team strength prevents identification of a causal continental-champion
  effect from 22 World Cups. The app labels structural, associational, and
  counterfactual quantities accordingly.
