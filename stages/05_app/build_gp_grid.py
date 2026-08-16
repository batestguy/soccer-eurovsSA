# -*- coding: utf-8 -*-
# =====================================================================
# STAGE 05 BUILD — GP LENS GRID
#
# Precomputes the GP "temporal lens" for the Gradio app: for each
# confederation, the posterior GP hyperparameters (amplitude, length-scale,
# noise) come from gp_posterior_{conf}.nc (Stage 03). The slider selects a
# MULTIPLIER on the posterior length-scale; the GP predictive is evaluated
# for each slider level WITHOUT any MCMC. Result -> gp_lens_grid.csv.
#
# This keeps inference fully decoupled from serving: the app only reads the
# precomputed grid, never runs a sampler.
# =====================================================================

import base64
import datetime as dt
import io
import os
import shutil
import subprocess
import tempfile
import urllib.request

import arviz as az
import numpy as np
import pandas as pd

GITHUB_REPO = "batestguy/soccer-eurovsSA"
DATA_ROOT = "https://raw.githubusercontent.com/batestguy/soccer-eurovsSA/main/data"
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]
SLIDER_FACTORS = [0.4, 0.6, 0.8, 1.0, 1.5, 2.5, 5.0]
PRED_YEARS = np.arange(1992, 2027, 1.0)
N_SAMPLES = 150


def fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "soccerdl-stage05"})
    return urllib.request.urlopen(req, timeout=180).read()


def exp_quad_kernel(x, y, length_scale, amplitude):
    d2 = ((x[:, None] - y[None, :]) ** 2)
    return amplitude**2 * np.exp(-0.5 * d2 / length_scale**2)


def gp_predictive(x, y, x_pred, length_scale, amplitude, noise):
    """Analytic GP predictive (mean + marginal variance) for fixed hyperparameters."""
    k = exp_quad_kernel(x, x, length_scale, amplitude) + noise**2 * np.eye(len(x))
    kp = exp_quad_kernel(x_pred, x, length_scale, amplitude)
    kpp = exp_quad_kernel(x_pred, x_pred, length_scale, amplitude)
    try:
        chol = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        k += 1e-6 * np.eye(len(x))
        chol = np.linalg.cholesky(k)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    mean = kp @ alpha
    v = np.diag(kpp) - np.einsum("ij,ij->i", kp, np.linalg.solve(chol.T, np.linalg.solve(chol, kp.T)).T)
    v = np.clip(v, 0, None)
    return mean, v


def push_to_github(repo, token, data_dir, stage_dir):
    clone_dir = "/content/soccerdl_repo"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = dict(os.environ)
    env.update({
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "http.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
        "GIT_CONFIG_KEY_1": "user.name",
        "GIT_CONFIG_VALUE_1": "soccerdl-bot",
        "GIT_CONFIG_KEY_2": "user.email",
        "GIT_CONFIG_VALUE_2": "soccerdl-bot@users.noreply.github.com",
    })
    subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", clone_dir],
                   env=env, check=True, capture_output=True, text=True)
    shutil.copytree(data_dir, os.path.join(clone_dir, "data"), dirs_exist_ok=True)
    shutil.copytree(stage_dir, os.path.join(clone_dir, "stages", "05_app"), dirs_exist_ok=True)
    subprocess.run(["git", "add", "-A"], cwd=clone_dir, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "stage 05: GP lens grid (app artifacts)"],
                   cwd=clone_dir, env=env, check=True, capture_output=True, text=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, env=env, check=True)


def main():
    out_root = "/content/soccerdl_out" if os.path.exists("/tmp/github_token") else "output"
    data_dir = os.path.join(out_root, "data")
    stage_dir = os.path.join(out_root, "stages", "05_app")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    annual = pd.read_csv(io.BytesIO(fetch_bytes(f"{DATA_ROOT}/gp_annual_confederation_frame.csv")))
    rows = []

    for conf in CONFEDERATIONS:
        nc = os.path.join(tempfile.gettempdir(), f"gp_fetch_{conf}.nc")
        with open(nc, "wb") as f:
            f.write(fetch_bytes(f"{DATA_ROOT}/gp_posterior_{conf}.nc"))
        idata = az.from_netcdf(nc)
        post = idata.posterior
        amps = post[f"amplitude_{conf}"].values.reshape(-1)
        lss = post[f"length_scale_{conf}"].values.reshape(-1)
        noises = post[f"noise_{conf}"].values.reshape(-1)
        idx = np.random.default_rng(20260815).choice(len(amps), size=N_SAMPLES, replace=False)
        amps, lss, noises = amps[idx], lss[idx], noises[idx]

        group = annual[annual["confederation"] == conf].sort_values("year")
        years = group["year"].astype(float).to_numpy()
        y = group["mean_elo"].to_numpy()
        x_mean, x_std = years.mean(), years.std()
        y_mean, y_std = y.mean(), y.std()
        x = (years - x_mean) / x_std
        y_norm = (y - y_mean) / y_std
        x_pred = (PRED_YEARS - x_mean) / x_std

        base_ls_median = float(np.median(lss))
        for factor in SLIDER_FACTORS:
            ls_eff = lss * factor
            means = np.empty((N_SAMPLES, len(PRED_YEARS)))
            variances = np.empty((N_SAMPLES, len(PRED_YEARS)))
            for s in range(N_SAMPLES):
                m, v = gp_predictive(x, y_norm, x_pred, ls_eff[s], amps[s], noises[s])
                means[s] = m
                variances[s] = v
            mean_pred = means.mean(axis=0)
            total_var = variances.mean(axis=0) + means.var(axis=0)
            sd = np.sqrt(total_var)
            for i, year in enumerate(PRED_YEARS):
                rows.append({
                    "confederation": conf,
                    "slider_level": factor,
                    "length_scale_years": round(base_ls_median * factor * x_std, 2),
                    "year": int(year),
                    "mean": mean_pred[i] * y_std + y_mean,
                    "p5": (mean_pred[i] - 1.645 * sd[i]) * y_std + y_mean,
                    "p95": (mean_pred[i] + 1.645 * sd[i]) * y_std + y_mean,
                })
        print(f"{conf}: base median length-scale = {base_ls_median * x_std:.2f} years")

    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(data_dir, "gp_lens_grid.csv"), index=False)
    print("gp_lens_grid rows:", len(grid))
    print("artifacts:", ", ".join(sorted(os.listdir(data_dir))))

    src = "/content/build_gp_grid.py"
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(stage_dir, "build_gp_grid.py"))
    if os.path.exists("/tmp/github_token"):
        token = open("/tmp/github_token", encoding="utf-8").read().splitlines()[0].strip()
        push_to_github(GITHUB_REPO, token, data_dir, stage_dir)
        print("commit: True | pushed to", GITHUB_REPO)
    else:
        print("commit: skipped (no /tmp/github_token)")


if __name__ == "__main__":
    main()
