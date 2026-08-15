# DAG VALIDATION — stage 01
Generated: 2026-08-15 04:51 UTC

## Model
Nodes: `team_strength, confederation, fifa_ranking, continental_champion, wc_outcome`

## Edge rationale
- confederation -> team_strength: hierarchical pooling — region sets the prior base.
- team_strength -> fifa_ranking: ratings measure strength (measurement model).
- team_strength -> continental_champion: strong teams win continental trophies.
- team_strength -> wc_outcome: strength is the dominant direct driver.
- continental_champion -> wc_outcome: the feature of interest.
- fifa_ranking -> wc_outcome: current form carries predictive information
  (the GP 'macro-trend' lens).
- confederation -> continental_champion: the aligned trophy's identity and
  competitiveness depend on the region (OFC vs UEFA).

## Implied conditional independencies (global Markov property, verified)
- `fifa_ranking _||_ continental_champion | {confederation, team_strength}` 
- `confederation _||_ fifa_ranking | {team_strength}` 
- `continental_champion _||_ fifa_ranking | {team_strength}` 
- `confederation _||_ wc_outcome | {continental_champion, fifa_ranking, team_strength}` 

Every statement conditions on `team_strength` (latent), so none is directly
testable with observed data. The observable proxy check that remains: within a
confederation, champion and non-champion Elo distributions overlap — the reason
a point estimate would be dishonest.

## Back-door paths: continental_champion -> wc_outcome
 - continental_champion <- team_strength -> fifa_ranking -> wc_outcome
 - continental_champion <- team_strength -> wc_outcome
 - continental_champion <- confederation -> team_strength -> fifa_ranking -> wc_outcome
 - continental_champion <- confederation -> team_strength -> wc_outcome

## Causal identification — the honest statement
Valid back-door adjustment sets (enumerated):
 - `{team_strength}`
 - `{confederation, team_strength}`
 - `{fifa_ranking, team_strength}`
 - `{confederation, fifa_ranking, team_strength}`

Every valid set includes the **latent** `team_strength` — confirmed: no observable-only set exists.
With 22 World Cups we therefore CANNOT identify a causal effect of
continental_champion from observational data. Claims about it are (1) a
*descriptive association* from the hierarchical regression, and (2)
*counterfactual simulations* from the Monte Carlo Oracle via `do()` on this DAG
(run the generative model with continental_champion forced to 0 vs 1).
Nothing is ever labeled an 'estimated causal effect'.

## Data gaps feeding this stage
- AFCON 2025 and Gold Cup 2025 champions are UNKNOWN (post-curation cutoff);
  they sit inside the 18-month window for 2026 and must be filled before
  building the 2026 `continental_champion` feature.
