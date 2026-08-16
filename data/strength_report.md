# CONTINENTAL STRENGTH — STAGE 05 BUILD REPORT

- Generated: 2026-08-16 09:31 UTC
- Panel: 32,432 team-month rows, 226 teams, 416 months, log(monthly Elo).
- Model: log_elo ~ N(alpha_0 + alpha_c + u_i + S(t).theta_c, sigma); spline df=8.
- Sampling: 2 chains x 300 draws; divergences=3; max R-hat=1.488; min ESS=4.0.
- Historical membership overrides applied (Australia, Israel, Kazakhstan).

## Continental strength ranking (log-Elo random intercepts)
| Continent | effect (mean) | 90% HDI |
|---|---|---|
| CONMEBOL | 0.112 | [0.014, 0.228] |
| AFC | 0.015 | [-0.046, 0.066] |
| OFC | -0.015 | [-0.078, 0.039] |
| CAF | -0.026 | [-0.094, 0.032] |
| UEFA | -0.035 | [-0.100, 0.019] |
| CONCACAF | -0.055 | [-0.128, 0.010] |

## Pairwise: which continent is stronger (overall average difference)
A ahead of B if avg_log_diff > 0; P(A>B) = P(avg difference > 0).
| Pair | avg log-diff | 90% HDI | avg Elo-pts diff | P(A>B) |
|---|---|---|---|---|
| CONMEBOL vs CONCACAF | +0.169 | [+0.060, +0.286] | +268.2 | 0.99 |
| CONMEBOL vs CAF | +0.146 | [+0.038, +0.260] | +234.7 | 0.99 |
| CONMEBOL vs OFC | +0.128 | [+0.018, +0.247] | +208.4 | 0.97 |
| CONMEBOL vs AFC | +0.101 | [-0.013, +0.222] | +167.5 | 0.93 |
| CAF vs CONCACAF | +0.023 | [-0.017, +0.065] | +33.5 | 0.81 |
| UEFA vs CONCACAF | +0.015 | [-0.025, +0.055] | +22.1 | 0.70 |

## Caveats
- 'Effect' is structural/associational (continent is a fixed attribute); NOT a causal claim.
- Differences are per the whole 1992-2026 window; the dynamic gap is in strength_diff_curves.csv.
- Elo is the documented FIFA-rankings proxy; log-Elo difference translates to %-higher, and
  Elo-points translation is exp() of the fitted log-trends (reference-level dependent).
