# DAG ASSUMPTION TESTS — STAGE 05 BUILD REPORT

- Generated: 2026-08-16 13:11 UTC
- Frame: 489 team-WC rows, 22 winners, reused from prior_model_frame.csv; Elo snapshot month 2026-06 (150 teams, latest full month).
- Checks probe DAG edges with the observed proxy Elo for latent team_strength.
- Honesty: these are edge checks, NOT causal-effect estimates (22 WCs can't identify).

## C1 — Confounding: continental_champion vs Elo (strength -> champion)
| group | mean log-Elo | n |
|---|---|---|
| champions | 7.536 | 55 |
| non-champions | 7.490 | 434 |

Mean diff (champion - non-champion) = **+0.045** log-Elo, 90% bootstrap CI [+0.027, +0.066], Welch t p=0.0003.
Interpretation: champions are on average stronger, supporting the strength->champion edge, but distributions overlap (the small-data reason we never claim a point causal effect).

## C2 — Independence proxy: does champion add win-info beyond Elo?
| model | OR(champion) | 90% CI | p | n |
|---|---|---|---|---|
| winner ~ elo_z + champion | 0.82 | [0.27, 2.45] | 0.7598 | 487 |
| winner ~ elo_z + champion + conf | 0.87 | [0.27, 2.77] | 0.8446 | 487 |
| LR test of conf terms | stat=15.30 (df=4) | — | 0.0041 | 487 |
Interpretation: with 22 winners the champion odds ratio is noisy — point estimate slightly below 1 with a 90% CI that easily includes 1, so there is **no clear evidence** champion adds win-information beyond Elo (consistent with the winner model's mu_cc ~ -0.11). The confederation terms are jointly strong (LR p small).

## C3 — Sensitivity: continent effects with/without conditioning
| confederation | logit M0 (marginal) | logit M2 (elo+champ) | shift | posterior ref (90% HDI) |
|---|---|---|---|---|
| CONMEBOL | +1.037 | +0.984 | -0.052 | +0.36 [-0.17, +1.26] |
| CAF | -12.183 | -11.848 | +0.334 | -0.10 [-0.95, +0.57] |
| CONCACAF | -12.238 | -11.965 | +0.273 | -0.18 [-1.10, +0.45] |
| AFC | -12.083 | -11.805 | +0.278 | -0.10 [-0.98, +0.61] |

Max |shift| M0->M2 = **0.334** (largest mover: CAF). CAF/CONCACAF/AFC have never won a World Cup, so their logits saturate near-separation; only the CONMEBOL-vs-UEFA contrast is well-identified, and it moves little (Δ≈-0.05). Continent log-odds are sensitive to conditioning — exactly why the winner model reports distributions and never labels conf effects 'causal'.

## C4 — Balance: confederation predicts Elo level (conf -> strength)
ANOVA on log-Elo, 150 teams at 2026-06 (latest full month): F = **5.9**, p = 0.0001, eta² = **0.171**.
| confederation | mean log-Elo | 90% CI | n |
|---|---|---|---|
| UEFA | 7.425 | [7.383, 7.467] | 55 |
| CONMEBOL | 7.571 | [7.517, 7.625] | 8 |
| CAF | 7.368 | [7.324, 7.412] | 31 |
| CONCACAF | 7.332 | [7.258, 7.406] | 22 |
| OFC | 7.178 | [6.986, 7.369] | 4 |
| AFC | 7.257 | [7.192, 7.322] | 30 |

Interpretation: confederation explains a large share of Elo variance, justifying the conf->strength hierarchical prior.
