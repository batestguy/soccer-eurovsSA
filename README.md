# Soccer Deep Learning - Bayesian World Cup Prediction

Bayesian decision-support app for football. It combines hierarchical PyMC inference,
precomputed posterior artifacts, a 50,000-simulation Monte Carlo Oracle, counterfactual
`do()` simulations, and an eight-tab Gradio interface.

## Live Release

- **App:** https://bayesian-world-cup-prediction.onrender.com
- **Source:** https://github.com/batestguy/soccer-eurovsSA
- **Render service:** https://dashboard.render.com/web/srv-da17uktbedkc73c99sd0
- **Application release commit:** `d6db8b7`; documentation updates deploy automatically from `main`
- **Runtime:** Render Free, Python 3.11.11, static precomputed artifacts, no runtime MCMC

The first Render deploy built successfully but failed at startup because Gradio imported
`requests` and it was not included in the release requirements. After that fix, interactive
NetCDF-backed tabs also required the `h5py` backend. The release pins `requests==2.32.3`
and `h5py==3.12.1`; the deployment is being reverified across all tabs.

## What the App Shows

| Tab | Purpose |
|---|---|
| Continental Strength | Hierarchical log-Elo trends, A/B comparison, 90% bands, and differences |
| Forest Plot | Posterior distributions for champion effect, confederation offset, and Elo coefficient |
| Monte Carlo | 50,000 simulated 2026 World Cups with winner probabilities and intervals |
| do()-What-If | Champion-status counterfactual simulation, not a causal estimate |
| Prior Predictive | Comparison of model beliefs before and after the observed data |
| Causal: Continent to Winner | DAG, structural effects, and identification caveat |
| DAG Assumption Tests | Four checks for confounding, independence, sensitivity, and balance |
| Ranking Dynamics | Descriptive top-5/top-10/top-20 composition with SES and 12-month scenario forecast |

## Modeling Guardrails

- Outputs are distributions with intervals or error bars, never point-only predictions.
- The winner model uses hierarchical pooling across 22 World Cups.
- The latent team-strength path prevents identification of a causal champion effect.
- `do()` results are counterfactual simulations, not estimated causal effects.
- Ranking Dynamics is deterministic descriptive SES, not a Bayesian forecast.
- Inference is decoupled from serving. Render never runs MCMC or model fitting.

Headline 2026 retrospective replay: Spain 20.8%, Argentina 14.2%, and UEFA 52.2%
confederation win probability. These are simulation summaries, not certainties.

## Run Locally

The serving bundle is self-contained under `spaces/` and reads only the 19 static artifacts
under `spaces/data/`.

```powershell
conda run -n causality-handbook python spaces\app.py
```

The app serves on `http://127.0.0.1:7860` by default. To use another port in PowerShell:

```powershell
$env:PORT = "7861"
conda run -n causality-handbook python spaces\app.py
```

## Release Validation

Run the no-refit release gate from the Bayesian analysis environment:

```powershell
conda run -n causality-handbook python stages\05_app\validate_release.py --report output\stage05_release_validation.json
```

The gate checks schemas, probability sums, interval ordering, 416 monthly strength periods,
139 quarterly periods, 35 annual periods, 15 pairwise comparisons, all eight tabs, 11 rendered
figures, app parity, and absence of runtime fitting code.

## Deployment

Render is configured by `render.yaml`:

- Build: `pip install -r spaces/requirements.txt`
- Start: `python spaces/app.py`
- Health check: `/`
- Port: the app binds `0.0.0.0` and Render's assigned `$PORT`

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the CLI workflow, failure diagnosis,
verification evidence, and recovery commands.

HF Spaces remains unavailable for this account because hosted Gradio/Docker Spaces on the
free `cpu-basic` plan returned HTTP 402 and require PRO.

## Project Documentation

- `SESSION_HANDOFF.md` - current stage status, model state, and gotchas
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) - live deployment runbook
- `stages/05_app/README_SPACES.md` - serving bundle documentation
- `spaces/README.md` - published serving-bundle metadata and limitations

Local-only workspace guides are not included in the public release: `Briefing.txt`,
`ENVIRONMENTS.md`, `AGENTS.md`, and `CLAUDE.md`.

## Stage Pipeline

| Stage | Output |
|---|---|
| 00 | Data foundation: canon, chronology, and alignment engine |
| 01 | Causal DAG and validation |
| 02 | Priors and prior predictive checks |
| 03 | Hierarchical posterior and GP trend |
| 04 | Monte Carlo Oracle and `do()` contrast |
| 05 | Eight-tab static Gradio release, Render deployment, and public browser verification |

Stages 00-04 and the model-building jobs run on Google Colab VMs driven from the terminal
through the official `colab` CLI in WSL. The public GitHub repository is the source of truth;
Colab VMs are ephemeral. See `SESSION_HANDOFF.md` for the published stage status and release
notes; local execution and security rules remain in the workspace-only guides.
