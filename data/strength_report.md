# CONTINENTAL STRENGTH — STAGE 05 BUILD REPORT

- Generated: 2026-08-16 11:47 UTC
- Panel: 32,432 team-month rows, 226 teams, 416 months, log(monthly Elo).
- Model: log_elo ~ N(alpha_0 + alpha_c + u_i + S(t).theta_c, sigma); spline df=8.
- Sampling: 2 chains x 1000 draws; divergences=17; max R-hat=1.104; min ESS=19.5.
- Historical membership overrides applied (Australia, Israel, Kazakhstan).

## Continental strength ranking (log-Elo random intercepts)
| Continent | effect (mean) | 90% HDI |
|---|---|---|
| CONMEBOL | 0.123 | [0.029, 0.240] |
| AFC | 0.014 | [-0.059, 0.075] |
| OFC | -0.013 | [-0.088, 0.050] |
| CAF | -0.034 | [-0.113, 0.038] |
| UEFA | -0.041 | [-0.114, 0.020] |
| CONCACAF | -0.064 | [-0.148, 0.010] |

## Pairwise: which continent is stronger (overall average difference)
A ahead of B if avg_log_diff > 0; P(A>B) = P(avg difference > 0).
| Pair | avg log-diff | 90% HDI | avg Elo-pts diff | P(A>B) |
|---|---|---|---|---|
| CONMEBOL vs CONCACAF | +0.188 | [+0.074, +0.314] | +297.6 | 1.00 |
| CONMEBOL vs CAF | +0.158 | [+0.051, +0.276] | +254.5 | 1.00 |
| CONMEBOL vs OFC | +0.137 | [+0.032, +0.250] | +222.6 | 0.99 |
| CONMEBOL vs AFC | +0.110 | [+0.008, +0.223] | +182.3 | 0.97 |
| CAF vs CONCACAF | +0.030 | [-0.024, +0.085] | +43.1 | 0.81 |
| UEFA vs CONCACAF | +0.024 | [-0.023, +0.072] | +34.2 | 0.81 |

## Caveats
- 'Effect' is structural/associational (continent is a fixed attribute); NOT a causal claim.
- Differences are per the whole 1992-2026 window; the dynamic gap is in strength_diff_curves.csv.
- Elo is the documented FIFA-rankings proxy; log-Elo difference translates to %-higher, and
  Elo-points translation is exp() of the fitted log-trends (reference-level dependent).
