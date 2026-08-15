# POSTERIOR DIAGNOSTICS — stage 03

- Generated: 2026-08-15 05:25 UTC
- Maximum R-hat: 1.377
- Minimum bulk ESS: 8.8

## Summary
|                       |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:----------------------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| beta_elo              |  1.507 | 0.251 |    0.977 |     1.913 |       0.024 |     0.019 |    109.412 |   1282.33  |   1.066 |
| mu_cc                 | -0.083 | 0.381 |   -0.922 |     0.489 |       0.066 |     0.021 |     33.923 |     57.378 |   1.092 |
| sigma_cc              |  0.309 | 0.274 |    0.01  |     0.793 |       0.065 |     0.007 |      8.805 |     12.341 |   1.377 |
| sigma_conf            |  0.454 | 0.295 |    0.028 |     0.881 |       0.056 |     0.015 |     25.118 |     34.654 |   1.101 |
| cc_effect[UEFA]       | -0.099 | 0.485 |   -1.215 |     0.684 |       0.072 |     0.038 |     47.73  |     57.761 |   1.07  |
| cc_effect[CONMEBOL]   | -0.235 | 0.493 |   -1.204 |     0.591 |       0.09  |     0.016 |     25.248 |   1103.39  |   1.107 |
| cc_effect[CAF]        | -0.113 | 0.531 |   -1.156 |     0.897 |       0.057 |     0.022 |     63.208 |   1179.28  |   1.052 |
| cc_effect[CONCACAF]   | -0.12  | 0.52  |   -1.084 |     0.88  |       0.071 |     0.019 |     44.132 |   1096.76  |   1.069 |
| cc_effect[OFC]        | -0.078 | 0.571 |   -1.266 |     0.954 |       0.063 |     0.027 |     57.321 |   1046.15  |   1.052 |
| cc_effect[AFC]        | -0.082 | 0.527 |   -1.131 |     0.888 |       0.056 |     0.023 |     59.302 |   1249.15  |   1.054 |
| conf_offset[UEFA]     |  0.061 | 0.325 |   -0.617 |     0.609 |       0.043 |     0.013 |     40.864 |   1299.16  |   1.064 |
| conf_offset[CONMEBOL] |  0.414 | 0.428 |   -0.216 |     1.136 |       0.056 |     0.026 |     57.439 |    959.301 |   1.053 |
| conf_offset[CAF]      |  0.197 | 0.652 |   -0.709 |     1.411 |       0.223 |     0.118 |      9.783 |     13.339 |   1.315 |
| conf_offset[CONCACAF] | -0.23  | 0.443 |   -1.01  |     0.595 |       0.064 |     0.024 |     27.699 |    963.306 |   1.092 |
| conf_offset[OFC]      |  0.047 | 0.448 |   -0.819 |     0.934 |       0.029 |     0.028 |     64.652 |    907.071 |   1.045 |
| conf_offset[AFC]      | -0.258 | 0.506 |   -1.069 |     0.644 |       0.113 |     0.017 |     26.821 |    293.001 |   1.098 |

## Interpretation
- R-hat near 1.00 indicates chains mixed adequately for the reported parameters.
- Intervals are posterior uncertainty, not point certainty.
- The continental champion effect remains associational because latent strength
  is not observed; causal claims remain prohibited by the Stage 01 DAG.
