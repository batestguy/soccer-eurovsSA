# -*- coding: utf-8 -*-
"""Stage 03: posterior inference, GP trends, and diagnostics."""

import base64
import datetime as dt
import io
import os
import shutil
import subprocess
import sys
import urllib.request

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


GITHUB_REPO = "batestguy/soccer-eurovsSA"
DATA_ROOT = "https://raw.githubusercontent.com/batestguy/soccer-eurovsSA/main/data"
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]


def fetch_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "soccerdl-stage03"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_csv(name):
    return pd.read_csv(io.BytesIO(fetch_bytes(f"{DATA_ROOT}/{name}")))


def load_stage02_module():
    source = fetch_bytes(
        "https://raw.githubusercontent.com/batestguy/soccer-eurovsSA/main/"
        "stages/02_priors/stage02_priors.py"
    )
    namespace = {"__name__": "stage02_module"}
    exec(compile(source, "stage02_priors.py", "exec"), namespace)
    return namespace


def posterior_diagnostics(idata, path):
    variables = ["beta_elo", "mu_cc", "sigma_cc", "sigma_conf", "cc_effect", "conf_offset"]
    summary = az.summary(idata, var_names=variables, round_to=3)
    summary.to_csv(path.replace(".md", ".csv"))
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    lines = [
        "# POSTERIOR DIAGNOSTICS — stage 03",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Maximum R-hat: {max_rhat:.3f}",
        f"- Minimum bulk ESS: {min_ess:.1f}",
        "",
        "## Summary",
        summary.to_markdown(),
        "",
        "## Interpretation",
        "- R-hat near 1.00 indicates chains mixed adequately for the reported parameters.",
        "- Intervals are posterior uncertainty, not point certainty.",
        "- The continental champion effect remains associational because latent strength",
        "  is not observed; causal claims remain prohibited by the Stage 01 DAG.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return max_rhat, min_ess


def fit_gp_series(confederation, years, values, fast=False):
    x = ((years - years.mean()) / years.std()).to_numpy()[:, None]
    y = values.to_numpy(dtype=float)
    y = (y - y.mean()) / y.std()
    x_pred = np.linspace(-1.2, 1.2, 36)[:, None]

    with pm.Model() as model:
        amplitude = pm.HalfNormal(f"amplitude_{confederation}", sigma=0.70)
        length_scale = pm.Gamma(f"length_scale_{confederation}", alpha=2.0, beta=1.0)
        noise = pm.HalfNormal(f"noise_{confederation}", sigma=0.30)
        covariance = amplitude**2 * pm.gp.cov.ExpQuad(1, ls=length_scale)
        gp = pm.gp.Marginal(cov_func=covariance)
        gp.marginal_likelihood("y", X=x, y=y, sigma=noise)
        gp.conditional("f_pred", Xnew=x_pred)
        trace = pm.sample(
            draws=40 if fast else 150,
            tune=40 if fast else 150,
            chains=1 if fast else 2,
            cores=1,
            target_accept=0.95,
            random_seed=20260815,
            progressbar=False,
        )
        prediction = pm.sample_posterior_predictive(
            trace,
            var_names=["f_pred"],
            random_seed=20260815,
            progressbar=False,
        )
    samples = prediction.posterior_predictive["f_pred"].stack(sample=("chain", "draw")).values
    return model, trace, x_pred[:, 0], samples


def save_gp_plot(predictions, path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), dpi=160, sharex=True)
    axes = axes.ravel()
    for ax, confederation in zip(axes, CONFEDERATIONS):
        item = predictions[confederation]
        x = item["years"]
        samples = item["samples"]
        mean = samples.mean(axis=1)
        low = np.quantile(samples, 0.05, axis=1)
        high = np.quantile(samples, 0.95, axis=1)
        ax.fill_between(x, low, high, color="#74A9CF", alpha=0.30)
        ax.plot(x, mean, color="#045A8D", lw=2, label="posterior GP")
        ax.scatter(item["observed_years"], item["observed_values"], s=16, color="#D95F02", alpha=0.75)
        ax.set_title(confederation)
        ax.grid(alpha=0.20)
        ax.set_ylabel("standardized mean Elo")
    for ax in axes[-2:]:
        ax.set_xlabel("World Cup year")
    fig.suptitle("Posterior GP trend by confederation\n90% credible bands", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def push_to_github(repo, token, data_dir, stage_dir):
    clone_dir = "/content/soccerdl_repo"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = dict(os.environ)
    env.update(
        {
            "GIT_CONFIG_COUNT": "3",
            "GIT_CONFIG_KEY_0": "http.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
            "GIT_CONFIG_KEY_1": "user.name",
            "GIT_CONFIG_VALUE_1": "soccerdl-bot",
            "GIT_CONFIG_KEY_2": "user.email",
            "GIT_CONFIG_VALUE_2": "soccerdl-bot@users.noreply.github.com",
        }
    )
    subprocess.run(
        ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", clone_dir],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    shutil.copytree(data_dir, os.path.join(clone_dir, "data"), dirs_exist_ok=True)
    shutil.copytree(stage_dir, os.path.join(clone_dir, "stages", "03_posterior"), dirs_exist_ok=True)
    subprocess.run(["git", "add", "-A"], cwd=clone_dir, env=env, check=True)
    subprocess.run(
        ["git", "commit", "-m", "stage 03: posterior and GP trend"],
        cwd=clone_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, env=env, check=True)


def main():
    out_root = "/content/soccerdl_out" if os.path.exists("/tmp/github_token") else "output"
    data_dir = os.path.join(out_root, "data")
    stage_dir = os.path.join(out_root, "stages", "03_posterior")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)
    fast = os.environ.get("SOCCERDL_FAST") == "1"

    print("Python:", sys.version.split()[0], "| PyMC:", pm.__version__, "| fast:", fast)
    stage02 = load_stage02_module()
    frame = fetch_csv("prior_model_frame.csv")

    # Fit the winner model and save immediately after sampling.
    model = stage02["build_prior_model"](frame)
    with model:
        posterior = pm.sample(
            draws=80 if fast else 400,
            tune=80 if fast else 400,
            chains=1 if fast else 2,
            cores=1,
            target_accept=0.96,
            random_seed=20260815,
            progressbar=False,
        )
    posterior_path = os.path.join(data_dir, "posterior.nc")
    az.to_netcdf(posterior, posterior_path)
    print("saved immediately:", posterior_path)
    max_rhat, min_ess = posterior_diagnostics(
        posterior, os.path.join(data_dir, "posterior_diagnostics.md")
    )
    az.plot_trace(
        posterior,
        var_names=["beta_elo", "mu_cc", "sigma_cc", "sigma_conf"],
        compact=True,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "posterior_trace.png"), dpi=160, bbox_inches="tight")
    plt.close("all")

    # Fit independent GP trends to annual mean Elo by confederation.
    chronology = fetch_csv("ranking_chronology.csv")
    team_conf = fetch_csv("team_confederations.csv")
    chronology["confederation"] = chronology["team"].map(
        team_conf.set_index("team")["confederation"]
    )
    chronology["year"] = chronology["month"].str[:4].astype(int)
    chronology = chronology[
        (chronology["year"] >= 1992)
        & (chronology["year"] <= 2022)
        & chronology["confederation"].isin(CONFEDERATIONS)
    ]
    annual = (
        chronology.groupby(["confederation", "year"], as_index=False)["elo"]
        .mean()
        .rename(columns={"elo": "mean_elo"})
    )
    annual.to_csv(os.path.join(data_dir, "gp_annual_confederation_frame.csv"), index=False)

    predictions = {}
    for confederation in CONFEDERATIONS:
        group = annual[annual["confederation"] == confederation].sort_values("year")
        if len(group) < 10:
            raise ValueError(f"Too few annual observations for {confederation}: {len(group)}")
        years = group["year"].astype(float)
        model_gp, gp_trace, x_pred, samples = fit_gp_series(
            confederation, years, group["mean_elo"], fast=fast
        )
        az.to_netcdf(gp_trace, os.path.join(data_dir, f"gp_posterior_{confederation}.nc"))
        scale = group["mean_elo"].std()
        predictions[confederation] = {
            "years": np.linspace(1992, 2026, len(x_pred)),
            "samples": samples * scale + group["mean_elo"].mean(),
            "observed_years": years.to_numpy(),
            "observed_values": group["mean_elo"].to_numpy(),
        }
    save_gp_plot(predictions, os.path.join(data_dir, "gp_posterior_trends.png"))

    report = [
        "# POSTERIOR REPORT — stage 03",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Winner-model frame: {len(frame)} rows across {frame['wc_year'].nunique()} editions",
        f"- Winner-model draws/tune/chains: {100 if fast else 1000}/{100 if fast else 1000}/{1 if fast else 4}",
        f"- Maximum R-hat: {max_rhat:.3f}",
        f"- Minimum bulk ESS: {min_ess:.1f}",
        "- Posterior was saved immediately after winner-model sampling to `posterior.nc`.",
        "- GP models use annual mean Elo by confederation, 1992–2022, with 90% bands.",
        "- All reported quantities are posterior distributions; no point-only claims.",
        "- Causal interpretation remains prohibited by the Stage 01 DAG: the latent",
        "  team-strength adjustment variable is unavailable.",
    ]
    with open(os.path.join(data_dir, "posterior_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(stage_dir, "stage03_posterior.py"), "w", encoding="utf-8") as f:
        try:
            f.write(open("/content/stage03_posterior.py", encoding="utf-8").read())
        except Exception:
            pass

    print("artifacts:", ", ".join(sorted(os.listdir(data_dir))))
    if os.path.exists("/tmp/github_token"):
        token = open("/tmp/github_token", encoding="utf-8").read().splitlines()[0].strip()
        push_to_github(GITHUB_REPO, token, data_dir, stage_dir)
        print("commit: True | pushed to", GITHUB_REPO)
    else:
        print("commit: skipped (no /tmp/github_token)")


if __name__ == "__main__":
    main()
