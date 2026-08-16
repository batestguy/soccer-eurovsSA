# CONTINENTAL STRENGTH — STAGE 05 BUILD REPORT

- Generated: 2026-08-16 12:33 UTC
- Panel: 32,432 team-month rows, 226 teams, 416 months, log(monthly Elo).
- Model: log_elo ~ N(alpha_0 + alpha_c + u_i + S(t).theta_c, sigma); spline df=8.
- Sampling: 4 chains x 400 draws; divergences=5; max R-hat=1.081; min ESS=39.7.
- Historical membership overrides applied (Australia, Israel, Kazakhstan).

## Continental strength ranking (log-Elo random intercepts)
| Continent | effect (mean) | 90% HDI |
|---|---|---|
| CONMEBOL | 0.130 | [0.027, 0.245] |
| AFC | 0.023 | [-0.049, 0.083] |
| OFC | -0.005 | [-0.078, 0.058] |
| CAF | -0.033 | [-0.110, 0.033] |
| UEFA | -0.033 | [-0.103, 0.027] |
| CONCACAF | -0.065 | [-0.144, 0.006] |

## Pairwise: which continent is stronger (overall average difference)
A ahead of B if avg_log_diff > 0; P(A>B) = P(avg difference > 0).
| Pair | avg log-diff | 90% HDI | avg Elo-pts diff | P(A>B) |
|---|---|---|---|---|
| CONMEBOL vs CONCACAF | +0.195 | [+0.069, +0.314] | +308.1 | 1.00 |
| CONMEBOL vs CAF | +0.163 | [+0.052, +0.284] | +262.0 | 0.99 |
| CONMEBOL vs OFC | +0.135 | [+0.030, +0.248] | +220.4 | 0.99 |
| CONMEBOL vs AFC | +0.107 | [+0.002, +0.223] | +179.5 | 0.95 |
| CAF vs CONCACAF | +0.032 | [-0.022, +0.089] | +46.1 | 0.82 |
| UEFA vs CONCACAF | +0.032 | [-0.013, +0.072] | +45.9 | 0.89 |

## Caveats
- 'Effect' is structural/associational (continent is a fixed attribute); NOT a causal claim.
- Differences are per the whole 1992-2026 window; the dynamic gap is in strength_diff_curves.csv.
- Elo is the documented FIFA-rankings proxy; log-Elo difference translates to %-higher, and
  Elo-points translation is exp() of the fitted log-trends (reference-level dependent).
