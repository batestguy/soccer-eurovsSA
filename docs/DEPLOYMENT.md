# Deployment Runbook

## Current Release

- Live app: https://bayesian-world-cup-prediction.onrender.com
- GitHub source: https://github.com/batestguy/soccer-eurovsSA
- Render dashboard: https://dashboard.render.com/web/srv-da17uktbedkc73c99sd0
- Render service ID: `srv-da17uktbedkc73c99sd0`
- Workspace: `My Workspace` (`tea-da17r17lk1mc739dlkb0`)
- Branch: `main`
- Live documentation commit: `d43e021`
- Application dependency fix: `d6db8b7`
- Live Render deploy: `dep-da369hj7uimc73bd6e9g`

The service is a Render Free Python web service. It serves the static Gradio bundle
under `spaces/`, including precomputed CSV, NetCDF, Markdown, and PNG artifacts.
Runtime serving does not fit a model, run MCMC, or regenerate the analysis.

## Render Configuration

`render.yaml` is the source configuration:

```yaml
buildCommand: pip install -r spaces/requirements.txt
startCommand: python spaces/app.py
healthCheckPath: /
```

The service uses Python 3.11.11 through `PYTHON_VERSION`. `spaces/app.py` binds
`0.0.0.0` and reads Render's assigned `$PORT`, falling back to `7860` locally.

## CLI Authentication

Install the official Render CLI for the current platform, then authenticate through
the browser. Do not place API keys or CLI tokens in the repository.

```powershell
render login
render workspaces -o json --confirm
render workspace set tea-da17r17lk1mc739dlkb0
render services -o json --confirm
```

Useful service and deploy commands:

```powershell
render deploys list srv-da17uktbedkc73c99sd0 -o json --confirm
render logs --resources srv-da17uktbedkc73c99sd0 --output text --limit 100 --confirm
render deploys create srv-da17uktbedkc73c99sd0 --commit <COMMIT_SHA> --wait --confirm
```

Use `--clear-cache` on `render deploys create` only when a dependency cache is
actually suspected. Automatic deploys are enabled for pushes to `main`.

## First Deploy Failure

The first deployment of commit `35b0d03` completed dependency installation and
uploaded the build successfully. It then crashed during application startup:

```text
File "/opt/render/project/src/.venv/lib/python3.11/site-packages/gradio/cli/commands/components/docs.py"
  import requests
ModuleNotFoundError: No module named 'requests'
```

The root cause was Gradio 5.23.1 importing its CLI module at `import gradio`, while
`requests` was not installed by the original pinned requirements. The minimal fix
added this line to both requirements files:

```text
requests==2.32.3
```

Commit `d6db8b7` deployed successfully. The later documentation commit `d43e021`
also deployed successfully and is the current live commit.

## Local Validation

Run the release gate before changing the serving bundle:

```powershell
conda run -n causality-handbook python stages\05_app\validate_release.py --report output\stage05_release_validation.json
```

The gate verifies:

- CSV schemas and required static artifacts
- 48-team Monte Carlo field and probability sums
- interval ordering for posterior and simulation outputs
- 416 monthly, 139 quarterly, and 35 annual strength periods
- 15 pairwise continental comparisons
- 12-month SES forecasts for all 18 ranking series
- DAG checks C1-C4
- posterior and prior variables
- byte parity between the Stage 05 and Space app copies
- absence of PyMC imports and runtime sampling
- all eight tabs and 11 canonical rendered figures

The final gate passed with app SHA-256:
`29af327845d2755b70531a44799cf1095e0e249b24163e18d4fff610800544de`.

## Public Verification

The live URL returned HTTP 200 and the browser verified the title
`Bayesian World Cup Prediction`, all eight tabs, the Ranking Dynamics controls,
and successful Gradio queue requests. Expected non-blocking browser messages are:

- `/manifest.json` returns 404 because the app does not provide a PWA manifest.
- The optional Gradio UI font returns 404 and falls back to a system font.
- Gradio reports harmless "Too many arguments provided for the endpoint" warnings.

These warnings do not prevent the application or any tab from loading.

## Recovery Checklist

1. Check the current deploy with `render deploys list`.
2. Query service logs with `render logs` and inspect the final traceback.
3. Confirm `spaces/requirements.txt` and `stages/05_app/requirements.txt` remain identical.
4. Run the local release gate.
5. Push the smallest fix to `main`.
6. Wait for the automatic deploy to become `live`.
7. Verify the public URL returns HTTP 200 and exercise the tabs in a browser.

Never commit `Githubtoken.txt`, Render credentials, API keys, or local CLI tokens.

## Hosting Alternatives

Hugging Face Spaces is not the active host. The authenticated `JBZABC` account was
rejected with HTTP 402 because hosted Gradio/Docker Spaces on the free `cpu-basic`
plan require PRO. Render is the working public deployment target.
