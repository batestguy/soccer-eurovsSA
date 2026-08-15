# PRIOR PREDICTIVE REPORT — stage 02

- Generated: 2026-08-15 05:00 UTC
- Model frame: 489 team-World Cup rows across 22 editions
- Outcome likelihood: one categorical World Cup winner per edition (softmax)
- Prior draws: 600 winner-model draws; 400 GP prior draws

## Priors
- `beta_elo ~ Normal(1.0, 0.5)` on standardized pre-tournament Elo
- `mu_cc ~ Normal(0.20, 0.50)`; `sigma_cc ~ HalfNormal(0.50)`
- `cc_effect[confederation] ~ Normal(mu_cc, sigma_cc)`
- `sigma_conf ~ HalfNormal(0.50)`; confederation offsets are hierarchical
- GP amplitude `~ HalfNormal(0.70)`; length scale `~ Gamma(2, 1)`

## Sanity summary
- beta_elo prior mean/90% interval: 1.008 [0.174, 1.787]
- mean continental effect prior mean/90% interval: 0.194 [-0.657, 0.981]
- Prior probabilities are shown as intervals, never point predictions.
- The prior model is deliberately weak enough that Elo informs ranking without
  making the champion feature deterministic.

## Artifacts
- `prior.nc`: complete prior/prior-predictive InferenceData
- `gp_prior.nc`: GP prior InferenceData
- `prior_predictive_probabilities.png`: 2022 prior winner probabilities
- `prior_continental_effects.png`: hierarchical champion-effect priors
- `gp_prior.png`: prior GP trajectories
- `prior_model_frame.csv`: exact frame used by the prior model

## Gate notes
- AFCON 2025 and Gold Cup 2025 remain UNKNOWN in the alignment table; they do
  not affect prior sampling, but must be resolved before posterior fitting.
