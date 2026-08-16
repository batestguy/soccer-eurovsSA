# MONTE CARLO ORACLE — STAGE 04 REPORT

- Generated: 2026-08-16 03:36 UTC
- Retrospective pre-tournament replay of the 2026 World Cup.
- Field: 48 teams from match participation; Elo frozen before 2026-06-11.
- Simulations: 50,000 per scenario (natural, do(champion=1), do(champion=0)).
- Posterior draws sampled with replacement from posterior.nc (800 draws).

## No-leakage statement
- Team list uses participation only (no scores, no results).
- Elo recomputed from full match history and frozen strictly before 2026-06-11.
- 2026 continental champions (resolved): UEFA=Spain, CONMEBOL=Argentina, CAF=Morocco, CONCACAF=Mexico, OFC=New Zealand, AFC=Qatar

## Headline — P(win) with 90% interval and empirical winner frequency
| # | Team | Conf | P(win) | 90% interval | Win freq (50k sims) |
|---|---|---|---|---|---|
| 1 | Spain | UEFA | 20.8% | 8.3%–39.3% | 20.8% |
| 2 | Argentina | CONMEBOL | 14.2% | 5.5%–25.4% | 14.1% |
| 3 | France | UEFA | 6.9% | 4.4%–9.8% | 7.0% |
| 4 | Brazil | CONMEBOL | 5.9% | 3.4%–9.5% | 5.9% |
| 5 | Ecuador | CONMEBOL | 4.9% | 2.8%–8.0% | 5.0% |
| 6 | England | UEFA | 3.8% | 2.6%–5.1% | 3.7% |
| 7 | Portugal | UEFA | 3.8% | 2.6%–5.1% | 3.8% |
| 8 | Colombia | CONCACAF | 3.1% | 0.8%–5.5% | 3.0% |
| 9 | Germany | UEFA | 2.9% | 2.0%–3.9% | 3.1% |
| 10 | Turkey | UEFA | 2.6% | 1.7%–3.5% | 2.7% |
| 11 | Morocco | CAF | 2.5% | 0.4%–5.7% | 2.4% |
| 12 | Japan | AFC | 2.2% | 0.6%–4.0% | 2.2% |

## Confederation win probability (sum of team probs) and winner frequency
| Confederation | P(a team from conf wins) | Empirical winner conf freq |
|---|---|---|
| UEFA | 52.2% | 52.4% |
| CONMEBOL | 28.9% | 28.9% |
| CONCACAF | 6.8% | 6.7% |
| CAF | 6.5% | 6.4% |
| AFC | 5.4% | 5.4% |
| OFC | 0.2% | 0.1% |

## do()-contrast (counterfactual simulation, NOT causal)
Δ = P(win | do(champion=1)) − P(win | do(champion=0)) for every team, 50k sims each.
Largest positive:
| Team | Conf | Δ mean | 90% interval |
|---|---|---|---|
| Spain | UEFA | +1.11pp | -5.43 to +9.96pp |
| France | UEFA | +0.36pp | -1.80 to +3.19pp |
| Morocco | CAF | +0.31pp | -1.30 to +2.57pp |
| England | UEFA | +0.20pp | -1.02 to +1.83pp |
| Portugal | UEFA | +0.20pp | -1.02 to +1.83pp |
Most negative:
| Argentina | CONMEBOL | -2.26pp | -13.74 to +6.04pp |
| Brazil | CONMEBOL | -0.71pp | -4.33 to +1.86pp |
| Ecuador | CONMEBOL | -0.59pp | -3.61 to +1.58pp |
| Uruguay | CONMEBOL | -0.27pp | -1.59 to +0.69pp |
| Paraguay | CONMEBOL | -0.21pp | -1.22 to +0.53pp |

## Honesty statement
These are posterior-predictive simulations from the Stage 03 softmax model. Because the
Stage 01 DAG shows `continental_champion` is confounded by latent team strength and every
back-door adjustment set requires the unmeasured variable, the do()-contrast is a
**counterfactual simulation**, not an estimated causal effect. All quantities are
distributions with credible intervals — never point predictions.

## Artifacts
- `monte_carlo_results.csv`, `do_contrast.csv`, `sim_field_2026.csv`
- `monte_carlo_barchart.png`, `do_contrast.png`
- `simulation_config.json`, `stage04_report.md`
