# SESSION HANDOFF — SoccerDL (Bayesian World Cup Prediction)

Prepared: 2026-08-15. **Last handoff update: 2026-08-20 (Render deployment live; startup dependency fixed; public URL verified).**
**Read this first in a new session.** Repo: `D:\SoccerDeepLearning`.
Source of truth: public GitHub `batestguy/soccer-eurovsSA`. Runtime: Google Colab driven by the
official `colab` CLI from WSL Ubuntu.

---

## 1. Status dashboard

| Stage | Status | GitHub commit (latest) | Key artifacts |
|---|---|---|---|
| 00 Data foundation | DONE | `bf8eef3` | `data/historical_canon.csv`, `continental_finals.csv`, `ranking_chronology.csv`, `alignment_engine.csv`, `team_confederations.csv`, `latest_elo.csv`, `source/results_65d212a.csv`, `data_manifest.md` |
| 01 Causal DAG | DONE | `279aef8` | `data/dag.dot`, `dag.png`, `dag_validation.md` |
| 02 Priors + prior predictive | DONE | `9bb5c48` | `data/prior.nc`, `gp_prior.nc`, `prior_model_frame.csv`, `prior_predictive_*.png`, `prior_predictive_report.md` |
| 03 Posterior + GP trend | DONE | `4850e00` | `data/posterior.nc`, `gp_posterior_{conf}.nc` (×6), `gp_posterior_trends.png`, `posterior_trace.png`, `posterior_report.md`, `posterior_diagnostics.{md,csv}` |
| 04 Monte Carlo Oracle + do() | DONE | `fc4ca47` | `data/monte_carlo_results.csv`, `do_contrast.csv`, `sim_field_2026.csv`, `monte_carlo_barchart.png`, `do_contrast.png`, `simulation_config.json`, `stage04_report.md` |
| 05 Gradio app | **LIVE on Render.** Final 8-tab app, static serving bundle, no-refit release gate, Render `$PORT` binding, and pinned startup dependencies are complete. | `d6db8b7` | `stages/05_app/app.py`, `validate_release.py`, `release_stage05.py`, `render.yaml`; byte-identical `spaces/app.py`; 19 static serving artifacts. Public URL: https://bayesian-world-cup-prediction.onrender.com. HF create-Space returned HTTP 402 because hosted Gradio/Docker on `cpu-basic` requires PRO. |

Stage scripts are preserved under `stages/<NN>_<name>/<script>.py` **on GitHub** (provenance copies
pushed by each stage; sizes are nonzero).

## 2. Model state (what the posterior actually is)

- **Winner model**: one categorical (softmax) winner per World Cup edition, 22 editions (1930–2022).
  Per-team logit = `beta_elo * elo_z + conf_offset[conf] + cc_effect[conf] * continental_champion`.
- **Priors** (stage 02): `beta_elo ~ Normal(1,0.5)`, `mu_cc ~ Normal(0.20,0.50)`,
  `sigma_cc, sigma_conf ~ HalfNormal(0.50)`, **non-centered** hierarchy (`cc_raw`, `conf_raw`).
- **Posterior** (stage 03): 2 chains × 400 draws, `target_accept=0.96`, **0 divergences**,
  max R-hat 1.009, min bulk ESS 308.8. `beta_elo` ≈ 1.46±0.29. `mu_cc` ≈ −0.11±0.41 (≈ no effect).
- **GP trend** (stage 03): per-confederation Marginal GP on annual mean Elo 1992–2022, ExpQuad
  kernel, 90% bands, 150 draws/chain × 2 chains.
- **Honesty rule (from Stage 01 DAG)**: `{team_strength}` is the minimal back-door adjustment set
  and it is latent → the continental-champion effect is **NOT identifiable**. Every output is a
  distribution; nothing is ever labeled an "estimated causal effect". `do()`-simulations are
  counterfactual model runs, not estimates.

## 3. Execution model — how stages actually run

Everything is driven through the official Google `colab` CLI installed in **WSL Ubuntu**
(`wsl.exe -d Ubuntu -- bash -lc "colab ..."`). Windows itself is NOT supported by the CLI.

Quick reference:
```
wsl.exe -d Ubuntu -- bash -lc "colab new -s soccerdl"                          # provision CPU VM
wsl.exe -d Ubuntu -- bash -lc "colab upload -s soccerdl <winpath> <vm_path>"  # upload (winpath = /mnt/c/...)
wsl.exe -d Ubuntu -- bash -lc "colab exec -s soccerdl -f <LOCAL .py> --timeout N"  # run local file ON the VM
wsl.exe -d Ubuntu -- bash -lc "colab ls/status/log -s soccerdl ..."
wsl.exe -d Ubuntu -- bash -lc "colab stop -s soccerdl"
```
- `-f FILE` reads the file **locally in WSL** and ships it to the VM (so `/mnt/c/...` paths work;
  the VM's `/content/...` copy is separate).
- **D: drive is not mounted in WSL** (only C/E/G). Stage files live in
  `C:\Users\TOSHIBA\AppData\Local\Temp\opencode` = `/mnt/c/Users/TOSHIBA/AppData/Local/Temp/opencode`.
  Copy stage scripts + `Githubtoken.txt` there before every run.
- **Every stage script self-detects CLI mode** when `/tmp/github_token` exists on the VM: it
  writes artifacts to `/content/soccerdl_out`, then clones + commits + pushes to GitHub from inside
  the VM. Artifacts are inspected on GitHub before the next stage (gate protocol).
- GitHub auth inside the VM: `git -c http.extraheader="AUTHORIZATION: basic <b64(x-access-token:TOK)>"`
  via `GIT_CONFIG_COUNT` env (git ≥2.31). Token read from `/tmp/github_token`, **first line only**.
- Push runs from inside the VM. If the exec client times out/disconnects, the VM keeps running and
  still pushes at the end — verify via the GitHub API (`/repos/.../commits?per_page=1`) or
  `colab ls -s soccerdl /content/soccerdl_out/data` to watch progress (posterior saved first, then
  GPs one by one).

## 4. Stage 04 plan — Monte Carlo Oracle + do()-contrast (LOCKED + EXECUTED)

**Decision (user-confirmed): retrospective pre-tournament replay.** Executed at commit `fc4ca47`.
Headline 2026 replay (50k sims): Spain 20.8% [8.3–39.3%], Argentina 14.2%, France 6.9%, Brazil 5.9%,
Ecuador 4.9%; confederation P(win): UEFA 52.2%, CONMEBOL 28.9%, CONCACAF 6.8%, CAF 6.5%, AFC 5.4%,
OFC 0.2%. do()-contrast: Spain +1.1pp, Argentina −2.3pp; every 90% interval crosses zero → honest
"counterfactual simulation, not causal". 2026 field = 48 teams, Elo frozen before 2026-06-11.

Spec as executed:

1. **Field**: derive the 2026 World Cup teams from match participation in the pinned source
   (`FIFA World Cup` rows dated 2026-06-11..2026-07-19). Do NOT read scores/outcomes.
2. **No leakage**: Elo frozen strictly before `2026-06-11` (recompute full-history Elo, take last
   rating per team with `date < 2026-06-11`). `continental_champion` uses the **resolved** 2026
   aligned champions: UEFA=Spain, CONMEBOL=Argentina, CAF=**Morocco**, CONCACAF=**Mexico**,
   OFC=New Zealand, AFC=Qatar (all confirmed; AFCON/Gold Cup 2025 now filled).
3. **Simulation**: draw posterior parameter samples with replacement → for each of 50,000 draws,
   compute softmax logits for the 2026 field → sample one winner. (Pure NumPy vectorized — seconds.)
4. **do()-contrast**: same 50k, but force `continental_champion=1` vs `=0` for every team
   (compare P(win | champion) − P(win | not champion), per team & confederation). Label it a
   **counterfactual simulation**, never a causal estimate.
5. **Outputs** (mirror to `data/`): `monte_carlo_results.csv` (team × P(win), P(advance optional),
   90% intervals), `do_contrast.csv`, `monte_carlo_barchart.png` (ranked, error bars),
   `do_contrast.png`, `simulation_config.json`, `stage04_report.md`. Push via the same
   `push_to_github` pattern. Save stage script provenance.
6. Reuse pattern from `stages/02_priors/stage02_priors.py` (fetch CSVs from GitHub raw, recompute
   Elo) and `posterior.nc` (fetch to temp file, load with `az.from_netcdf`, see gotchas).

## 5. Gotchas / lessons (do not rediscover these)

1. **`argparse` breaks in Colab cells**: the kernel injects `-f /root/.local/share/jupyter/.../kernel.json`
   into `sys.argv`. Parse args manually (ignore unknown), or use `parse_known_args`.
2. **`colab exec -f` runs the file without `__file__`** → provenance copies must read from the
   uploaded path (`/content/<script>.py`) with a fallback to `__file__`.
3. **PyMC 5.28 renamed** `gp.marginal_likelihood(noise=...)` → `sigma=...`. Local `causality-handbook`
   has pymc 5.25 (old API works); the Colab VM has 5.28. Use the NEW name.
4. **Divergences on small-data hierarchy**: use non-centered parameterization
   (`param = mu + sigma * raw`, `raw ~ N(0,1)`) + `target_accept ≈ 0.96`. The old centered model
   gave 800 divergences.
5. **`az.from_netcdf(io.BytesIO(...))` fails** with `RuntimeError: file is closed` (xarray closes
   the byte file). Always write the bytes to a temp `.nc` on disk, then `az.from_netcdf(path)`.
6. **`Githubtoken.txt` has a note line after the token**: always `.splitlines()[0].strip()`.
   Never echo/commit the token. File is NOT pushed (only `data/`+`stages/`+README are).
7. **Free-tier VMs are flaky**: sessions get recycled (404/401) mid-session; `colab exec` clients
   time out while the VM keeps working; keep-alive errors in `colab log` are normal. Always make
   stages self-contained (fetch inputs from GitHub, push at the end) so a recycled VM doesn't lose
   work. Run bounded workloads (< ~30 min) or watch progress via `colab ls` + GitHub API.
8. **Local machine is NOT the runtime**: pytensor on this Windows box has no proper BLAS (slow GP
   fits). Use the Colab VM for all PyMC work. Local runs are smoke-only.
9. **`colab install`** can fail silently on the VM (uv/pip resolver); prefer dependencies already
   present (Colab ships pymc 5.28, arviz, numpy, pandas, matplotlib, networkx).
10. **Quoting through WSL**: avoid nested quotes in `bash -lc "..."`; for multi-line/sensitive
    Python, write a file and use `colab exec -f`.
11. **MCMC low-ESS fix (session 08-16): chains > draws.** numpyro runs of 300/400/1000 draws × 2
    chains all stuck at min ESS 13–20 / R-hat 1.10–1.12 on `sigma_alpha` (6 continents) no matter
    the draw count or `target_accept`. **4 chains × 400 draws fixed it** (alpha_c ESS 109–294,
    R-hat ≤1.05). The spline columns must be centered (subtract `spline_col_means`) or alpha_c and
    the spline confound. If it looks stuck again, try more chains before more draws.
12. **`ranking_chronology.csv` has sparse/partial months** — it ends at **2026-07**, and its last
    month has only 25 teams. Use 404 observed months; for Elo-level snapshots pick the latest month
    with ≥150 teams (**2026-06**, 154 teams). Gap months (Covid 2020-03..08, 2019-04, 2021-04,
    2023-05, 2025-04, 2026-04, 2026-08) exist — `ffill`/carry-forward for continuity.
13. **Fitted-α SES by 1-step SSE always → α≈1** on sticky series (even a smooth sine; the last
    value is the best 1-step forecast). Fit α by **rolling-origin +h-month forecast SSE** (h = the
    app's forecast horizon = 12). Grid-search α in numpy (no scipy needed).
14. **`posterior.nc` `conf_offset` coord is named `confederation`** (not `continent`); `cc_effect` is
    per-confederation too. `.sel(confederation=...)`.
15. **Local tooling quirks**: reportlab ≥5 dropped the `Body` style alias (use `Normal`);
    `pip install reportlab pypdf pdfplumber` into `causality-handbook` bumped pillow to 12.3 which
    **conflicts with gradio's `pillow<12` pin** — gradio may need a reinstall before serving if it
    breaks. `pd.period_range(...) + int` shifts a 1-element index — build forecast ranges with
    `pd.period_range(last+1, periods=12, freq="M")`. A stale `spaces\app.py` python process was
    holding port 7860 (old v1) — check `Get-NetTCPConnection -LocalPort 7860` before relaunching.
16. **Model cannot view images** (deepseek-v4-flash): render-gate = script exits clean + pdfplumber
    text extraction; hand the PNG/PDF to the user for visual confirmation.
17. **The `write` tool truncates very long content** (JSON parse error mid-string) — write large
    files in parts using a `# __NEXT__` sentinel + `edit` to append.

## 6. Stage 05 — Gradio app: FINAL PLAN (LOCKED 2026-08-16)

**v1 status:** built locally (6 tabs, not deployed). GP machinery **DROPPED** in the final plan
(`build_gp_grid.py` and `gp_lens_grid.csv` removed). HF bundle in `spaces/`. HF user `JBZABC`
(authenticated; hosted Gradio Space creation blocked by the account's non-PRO plan).

**Session 2026-08-16 — everything done so far (detail in the checklist below):**
1. ✅ `build_strength.py` pushed `c1347cb` (sampling settled on **numpyro 4 chains × 400 draws**,
   R-hat ≤1.08, `alpha_c` ESS ≥109, 5 divergences; earlier 2-chain runs failed ESS ≈13–20 no matter
   the draw count — more **chains** fixed it). Spline columns are centered (subtract
   `spline_col_means`) or `alpha_c` and the spline confound. Result: CONMEBOL +0.130 tops the
   typical-team ranking; trend/diff curves monthly 1992–2026.
2. ✅ `build_elite.py` pushed `df59b00` (Metric-A shares top 5/10/20 monthly 1992-01..2026-07;
   SES α fitted by **rolling-origin +12-month forecast SSE** — 1-step SSE collapses to α≈1; +12m
   anchored forecast; running-leader summary). UEFA led top-5/10/20 in 68%/70%/65% of months.
3. ✅ `build_dag_checks.py` pushed `2fb3ad2` (C1 champions +0.045 log-Elo, p=0.0003; C2 OR(champion)
   0.82 [0.27,2.45] → no clear extra info beyond Elo; C3 max |conf shift| 0.334; C4 ANOVA η²=0.17,
   n=150 at 2026-06 latest FULL month).
4. ✅ `app.py` + `spaces/` pushed at `c32e742`. Aggregation is corrected to 416 monthly /
   139 quarterly / 35 annual periods; strength diagnostics are disclosed; Ranking Dynamics is
   explicitly deterministic/descriptive; obsolete GP-lens packaging is absent.
5. ✅ `Analysis_Report.pdf` (9 pages, layman-friendly, table-heavy) written to the repo root — local
   deliverable, not pushed.
6. ⚠️ Authenticated as `JBZABC`, but HF rejected Gradio Space creation with HTTP 402: hosted
   Gradio/Docker on free `cpu-basic` requires PRO. The identical app is running locally at
   `http://127.0.0.1:7860` (PID 18380 at handoff time).

### Final app layout — 8 tabs
| # | Tab | Status |
|---|---|---|
| 1 | **Continental Strength** (replaces GP Trend) | locked |
| 2 | Forest Plot | kept |
| 3 | Monte Carlo (future cast) | kept |
| 4 | do()-What-If | kept |
| 5 | Prior Predictive | kept |
| 6 | **Causal: Continent → Winner** (plain DAG merged in) | new |
| 7 | **DAG Assumption Tests** (one tab, check selector, 4 checks) | new |
| 8 | **Ranking Dynamics — elite composition** (top 5/10/20, Metric A, SES) | new |

### Tab 1 — Continental Strength (locked)
- Model: **log(monthly Elo)** hierarchical panel, entity = team, group = continent, time = month;
  all teams (≥10-match filter); natural-cubic spline (df ≈ 6–10) × continent; non-centered;
  `mu_it = alpha_0 + alpha_c[i,t] + u_i + S(t)·theta_c[i,t]`; `alpha_c ~ N(0,sigma_continent)`,
  `u_i ~ N(0,sigma_team)`, `sigma ~ HalfNormal`, `target_accept≈0.96`.
- Historical membership via `data/conf_membership.csv`: Australia OFC→AFC **2006-01**, Israel →UEFA
  **1994-07**, Kazakhstan →UEFA **2002-01**; default = `team_confederations.csv`.
- App: two-continent selectors + period aggregation (monthly/quarterly/annual) for trend lines;
  **faint-all + highlight-two** team log-Elo series; A/B spline trends + 90% bands; separate
  **difference panel** (A−B, band, zero line); **annotation in non-overlapping sections** reporting
  BOTH overall-average difference (90% HDI, P(A>B)) and dynamic difference (periods band excludes
  zero), units in **both log-points and Elo-points**; side pairwise table (15 pairs).

### Tab 6 — Causal: Continent → Winner
- DAG diagram (continent→winner pathways highlighted) + `dag_validation` content.
- Posterior continent effects on P(win) from the winner model `conf_offset` (distributions,
  identification caveat: structural/associational, NOT "causal effect").

### Tab 7 — DAG Assumption Tests (one tab, check selector)
1. Confounding check — `continental_champion` vs Elo (confirms `strength→champion` edge).
2. Independence proxies — within confederation, does champion add win-info beyond Elo.
3. Sensitivity — continent effects with/without conditioning on champion + Elo.
4. Balance — confederation predicts Elo level (justifies `conf→strength` prior).

### Tab 8 — Ranking Dynamics (elite composition) (locked)
- Metric **A** (share of top-N slice, sums to 100%), thresholds **top 5/10/20**, monthly
  **1992-01 → 2026-07** (2026-07 = latest observed ranking month in `ranking_chronology.csv`),
  Elo ranking + `conf_membership` overrides.
- Model: **simple exponential smoothing (SES)** per continent×threshold on proportions directly
  (interpretable, flexible, α fitted); fitted line through monthly markers; **+12-month forecast**
  with the last observed values anchored; **"which confederation line is ahead over time"** summary
  (running leader, % window led, final ranking).

### Build scripts (each a bounded Colab-VM job → push → gate-inspect)
| Script | Produces |
|---|---|
| `stages/05_app/build_strength.py` | `data/conf_membership.csv`; `strength_posterior.nc`, `strength_pairwise.csv`, `strength_ranking.png`, `strength_report.md`, `strength_model_meta.json`, `strength_trends.csv`, `strength_diff_curves.csv` |
| `stages/05_app/build_elite.py` | `data/elite_composition.csv`, `elite_fit.csv`, `elite_summary.csv` |
| `stages/05_app/build_dag_checks.py` | `data/dag_checks.csv`, `data/dag_checks_report.md` (reuses posterior.nc, prior_model_frame.csv, ranking_chronology.csv) |
| `stages/05_app/app.py` update | 8-tab UI; drop GP Trend; rebuild `spaces/` bundle |

### Implementation order
1. ✅ `build_strength.py` → **DONE** (`c1347cb`): 4 chains × 400 draws (numpyro), R-hat ≤1.08, `alpha_c` ESS ≥109. Ranking (typical-team log-Elo): CONMEBOL +0.130 > AFC +0.023 > OFC −0.005 > CAF ≈ UEFA −0.033 > CONCACAF −0.065. Caveat: "typical team strength" — UEFA's big pool drags its mean below CONMEBOL's tight, strong pool; explain this in the app. Artifacts pushed: `conf_membership.csv`, `strength_posterior.nc`, `strength_pairwise.csv`, `strength_ranking.png`, `strength_report.md`, `strength_model_meta.json`, `strength_trends.csv`, `strength_diff_curves.csv`.
2. ✅ `build_elite.py` → **DONE** (`df59b00`): monthly Metric-A shares (top 5/10/20, 1992-01..2026-07 = latest observed ranking month; Covid gaps carried forward), SES per conf×threshold with α fitted by **rolling-origin +12-month forecast SSE** (1-step SSE collapses to α≈1 for persistent series — do not "fix" this), +12-month anchored forecast, running-leader summary. Top-5/10/20 leader over time: UEFA (68%/70%/65% of window); top-20 CAF 2nd (0.20). Artifacts pushed: `elite_composition.csv`, `elite_fit.csv`, `elite_summary.csv`, `elite_summary_report.md`.
3. ✅ `build_dag_checks.py` → **DONE** (`2fb3ad2`): 4 DAG edge checks on the observed Elo proxy. C1 confounding: champions +0.045 log-Elo stronger (90% CI [+0.027,+0.066], p=0.0003) → supports `strength→champion`; C2 independence proxy: OR(champion) 0.82 [0.27,2.45] → **no clear evidence** champion adds win-info beyond Elo (consistent with `mu_cc≈−0.11`); C3 sensitivity: max |conf shift| M0→M2 = 0.334 (CAF saturates — never won a WC; CONMEBOL contrast well-identified, Δ≈−0.05); C4 balance: ANOVA F=5.9, p=0.0001, η²=0.17 (n=150, **2026-06 latest FULL month** — 2026-07 is a 25-team partial snapshot, don't use it) → justifies `conf→strength` prior. Artifacts pushed: `dag_checks.csv`, `dag_checks_report.md`.
4. ✅ `app.py` + `spaces/` release → **PUSHED `c32e742`**. The release validator exercised
   35 selector calls, confirmed all eight tabs, schemas, probability sums, byte parity, no runtime
   fitting, and rendered 11 nonempty PNGs. Immutable Git blob SHA-256:
   `74f8641f795c31a009bc945f3af0642c5f64907a648e26989d732dec6e01f26e`.
5. ✅ Local browser gate → all eight tabs opened; period, continent, parameter, region, Top-N,
   prior/posterior, C1-C4, and ranking-threshold controls exercised. Evidence is under
   `output/playwright/` locally. Gradio produced only two missing-local-font 404s; no backend errors.
6. ✅ Render Free deployment → `render.yaml` builds with `pip install -r spaces/requirements.txt` and
   starts `python spaces/app.py`; both app copies honor Render's assigned `$PORT` and bind to `0.0.0.0`.
   The first deploy exposed a missing transitive `requests` import in Gradio; pinned `requests==2.32.3`
   was added in `d6db8b7`. Render deploy `dep-da366c9srm7s7390mf4g` is **LIVE** and the public URL
   returned HTTP 200 with all eight tabs. Browser console has only the known optional `manifest.json`
   and font 404s plus harmless Gradio argument warnings. HF Spaces remains blocked by the account plan.
   No refit or artifact rebuild is needed.

### Guardrails (unchanged)
Distributions-not-points everywhere; DAG labels "associational / counterfactual, not causal";
inference decoupled from serving (no MCMC at runtime); SESSION_HANDOFF §5 gotchas apply (non-centered,
`target_accept≈0.96`, temp-file `.nc` loads, token first line, bounded self-contained jobs).

## 7. Data gap status

- AFCON 2025 = **Morocco** (beat Senegal 3-0, 2026-01-18), Gold Cup 2025 = **Mexico**
  (beat USA 2-1, 2025-07-06) — both resolved from the pinned match source and already in
  `alignment_engine.csv`. No remaining UNKNOWN champions for the 2026 window.
