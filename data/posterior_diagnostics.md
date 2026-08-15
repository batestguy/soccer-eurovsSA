# POSTERIOR DIAGNOSTICS — stage 03

- Generated: 2026-08-15 07:48 UTC
- Maximum R-hat: 1.009
- Minimum bulk ESS: 308.8
- Divergences after tuning: 0

## Summary
|                       |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:----------------------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| beta_elo              |  1.461 | 0.288 |    0.859 |     1.945 |       0.008 |     0.009 |   1217.84  |    730.023 |   1.001 |
| mu_cc                 | -0.106 | 0.405 |   -0.91  |     0.605 |       0.012 |     0.014 |   1201.02  |    573.874 |   1     |
| sigma_cc              |  0.369 | 0.282 |    0.002 |     0.884 |       0.011 |     0.007 |    481.946 |    305.266 |   1.003 |
| sigma_conf            |  0.42  | 0.298 |    0.003 |     0.967 |       0.013 |     0.008 |    388.088 |    360.395 |   1.003 |
| cc_effect[UEFA]       | -0.074 | 0.51  |   -1.074 |     0.907 |       0.015 |     0.021 |   1173.49  |    638.041 |   1     |
| cc_effect[CONMEBOL]   | -0.313 | 0.542 |   -1.319 |     0.579 |       0.018 |     0.022 |    912.337 |    671.588 |   1.004 |
| cc_effect[CAF]        | -0.116 | 0.658 |   -1.279 |     1.114 |       0.022 |     0.04  |    983.146 |    505.478 |   1.001 |
| cc_effect[CONCACAF]   | -0.162 | 0.596 |   -1.312 |     0.917 |       0.019 |     0.022 |   1045.07  |    710.411 |   0.999 |
| cc_effect[OFC]        | -0.082 | 0.63  |   -1.312 |     1.058 |       0.02  |     0.025 |   1017.84  |    694.043 |   1     |
| cc_effect[AFC]        | -0.151 | 0.608 |   -1.25  |     0.9   |       0.021 |     0.032 |    966.957 |    647.844 |   1.003 |
| conf_offset[UEFA]     |  0.012 | 0.367 |   -0.686 |     0.748 |       0.013 |     0.017 |    831.762 |    606.876 |   1.003 |
| conf_offset[CONMEBOL] |  0.359 | 0.466 |   -0.265 |     1.363 |       0.023 |     0.018 |    308.793 |    733.957 |   1.009 |
| conf_offset[CAF]      | -0.099 | 0.469 |   -0.95  |     0.907 |       0.017 |     0.024 |    858.687 |    616.386 |   1     |
| conf_offset[CONCACAF] | -0.184 | 0.489 |   -1.104 |     0.841 |       0.016 |     0.023 |   1049.12  |    719.772 |   0.998 |
| conf_offset[OFC]      |  0.005 | 0.476 |   -1.049 |     0.977 |       0.018 |     0.026 |    744.203 |    596.497 |   0.999 |
| conf_offset[AFC]      | -0.103 | 0.467 |   -1.145 |     0.715 |       0.017 |     0.022 |    823.173 |    518.414 |   1.005 |

## Interpretation
- R-hat near 1.00 indicates chains mixed adequately for the reported parameters.
- Intervals are posterior uncertainty, not point certainty.
- The continental champion effect remains associational because latent strength
  is not observed; causal claims remain prohibited by the Stage 01 DAG.
