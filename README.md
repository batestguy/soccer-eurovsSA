# Soccer Deep Learning — Bayesian World Cup Prediction

Bayesian decision-support app for football (PyMC + Monte Carlo + Gradio).
Stage 00 = data foundation. Artifacts:
- `data/historical_canon.csv`      World Cup finals 1930-2026 (2026 = prediction target)
- `data/continental_finals.csv`    all continental finals (>= 1930)
- `data/ranking_chronology.csv`    monthly Elo per team since 1992-01 (self-computed proxy)
- `data/latest_elo.csv`            most recent Elo per team
- `data/alignment_engine.csv`      WC x confederation aligned continental tournament
- `data/team_confederations.csv`   team -> confederation map
- `data/source/`                   pinned raw match data (reproducibility)

See `data/data_manifest.md` for provenance and known gaps.
