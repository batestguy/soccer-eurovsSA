# Soccer Deep Learning — Bayesian World Cup Prediction

Bayesian decision-support app for football: hierarchical PyMC models + 50k-sim Monte Carlo
Oracle + Gradio UI. Stages run on **Google Colab VMs driven from the terminal** via the official
`colab` CLI (in WSL); the source of truth is online (public GitHub `batestguy/soccer-eurovsSA`),
never this local folder.

## Docs
- `Briefing.txt` — product spec (read first)
- `ENVIRONMENTS.md` — machine reference (local envs are NOT the runtime)
- `AGENTS.md` — working contract for agent sessions (stage pipeline, gate protocol, gotchas)
- `SESSION_HANDOFF.md` — current status + the **locked Stage-05 final plan** (8 tabs) + gotchas

## Stage pipeline
| Stage | Output |
|---|---|
| 00 | Data foundation (canon, chronology, alignment engine) |
| 01 | Causal DAG + validation |
| 02 | Priors + prior predictive checks |
| 03 | Hierarchical posterior + GP trend |
| 04 | Monte Carlo Oracle + do()-contrast |
| 05 | Eight-tab Gradio release validated locally and live on Render at https://bayesian-world-cup-prediction.onrender.com; HF hosting was rejected with HTTP 402 because `JBZABC` is not on PRO |

## How a stage runs
Each stage is a self-contained script the agent executes on a Colab VM via
`wsl.exe -d Ubuntu -- bash -lc "colab exec -s soccerdl -f <local.py> --timeout N"`; it pushes
artifacts to GitHub from inside the VM. The agent inspects artifacts on GitHub before the next stage.
See `AGENTS.md` + `SESSION_HANDOFF.md` §3 for the full command reference and gotchas.
