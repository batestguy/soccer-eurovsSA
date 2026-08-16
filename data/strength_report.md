# CONTINENTAL STRENGTH — STAGE 05 BUILD REPORT

- Generated: 2026-08-16 09:59 UTC
- Panel: 32,432 team-month rows, 226 teams, 416 months, log(monthly Elo).
- Model: log_elo ~ N(alpha_0 + alpha_c + u_i + S(t).theta_c, sigma); spline df=8.
- Sampling: 2 chains x 400 draws; divergences=4; max R-hat=1.122; min ESS=13.5.
- Historical membership overrides applied (Australia, Israel, Kazakhstan).

## Continental strength ranking (log-Elo random intercepts)
| Continent | effect (mean) | 90% HDI |
|---|---|---|
| CONMEBOL | 0.123 | [0.029, 0.241] |
| AFC | 0.019 | [-0.047, 0.078] |
| OFC | -0.008 | [-0.076, 0.052] |
| CAF | -0.029 | [-0.108, 0.033] |
| UEFA | -0.037 | [-0.103, 0.022] |
| CONCACAF | -0.060 | [-0.139, 0.011] |

## Pairwise: which continent is stronger (overall average difference)
A ahead of B if avg_log_diff > 0; P(A>B) = P(avg difference > 0).
| Pair | avg log-diff | 90% HDI | avg Elo-pts diff | P(A>B) |
|---|---|---|---|---|
| CONMEBOL vs CONCACAF | +0.184 | [+0.066, +0.322] | +291.4 | 1.00 |
| CONMEBOL vs CAF | +0.152 | [+0.045, +0.285] | +245.8 | 0.99 |
| CONMEBOL vs OFC | +0.131 | [+0.029, +0.255] | +214.6 | 0.99 |
| CONMEBOL vs AFC | +0.104 | [+0.001, +0.227] | +173.7 | 0.95 |
| CAF vs CONCACAF | +0.032 | [-0.028, +0.101] | +45.5 | 0.77 |
| UEFA vs CONCACAF | +0.024 | [-0.018, +0.070] | +34.3 | 0.81 |

## Caveats
- 'Effect' is structural/associational (continent is a fixed attribute); NOT a causal claim.
- Differences are per the whole 1992-2026 window; the dynamic gap is in strength_diff_curves.csv.
- Elo is the documented FIFA-rankings proxy; log-Elo difference translates to %-higher, and
  Elo-points translation is exp() of the fitted log-trends (reference-level dependent).
