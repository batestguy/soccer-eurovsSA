# -*- coding: utf-8 -*-
"""Stage 02: prior specification and prior-predictive checks."""

import base64
import datetime as dt
import io
import json
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
MATCH_SOURCE = f"{DATA_ROOT}/source/results_65d212a.csv"
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]


def fetch_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "soccerdl-stage02"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_csv(name):
    return pd.read_csv(io.BytesIO(fetch_bytes(f"{DATA_ROOT}/{name}")))


def k_factor(tournament):
    tournament = str(tournament or "")
    if tournament == "FIFA World Cup":
        return 60.0
    if tournament in {
        "UEFA Euro",
        "Copa América",
        "African Cup of Nations",
        "Gold Cup",
        "Oceania Nations Cup",
        "AFC Asian Cup",
        "CONCACAF Championship",
    }:
        return 50.0
    if "qualification" in tournament or tournament in {
        "UEFA Nations League",
        "Confederations Cup",
        "FIFA Series",
        "CONMEBOL-UEFA Cup of Champions",
        "CONCACAF Nations League",
        "Olympic Games",
    }:
        return 40.0
    return 30.0


def goal_factor(goal_difference):
    if goal_difference <= 0:
        return 1.0
    return 1.5 + 0.125 * (min(goal_difference, 5) - 1)


def compute_elo_history(matches):
    ratings = {}
    rows = []

    for row in matches.sort_values("date").itertuples(index=False):
        home = ratings.setdefault(row.home_team, 1500.0)
        away = ratings.setdefault(row.away_team, 1500.0)
        home_advantage = 0.0 if row.neutral else 100.0
        expected_home = 1.0 / (1.0 + 10.0 ** ((away - (home + home_advantage)) / 400.0))
        score_home = (
            1.0
            if row.home_score > row.away_score
            else 0.5
            if row.home_score == row.away_score
            else 0.0
        )
        delta = (
            k_factor(row.tournament)
            * goal_factor(abs(row.home_score - row.away_score))
            * (score_home - expected_home)
        )
        ratings[row.home_team] += delta
        ratings[row.away_team] -= delta
        rows.extend(
            [
                (row.date, row.home_team, ratings[row.home_team]),
                (row.date, row.away_team, ratings[row.away_team]),
            ]
        )

    return pd.DataFrame(rows, columns=["date", "team", "elo"])


def build_model_frame(matches, canon, alignment, team_confederations):
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["neutral"] = matches["neutral"].astype(str).str.upper().eq("TRUE")
    elo_history = compute_elo_history(matches)
    elo_history["date"] = pd.to_datetime(elo_history["date"])

    conf_lookup = team_confederations.set_index("team")["confederation"].to_dict()
    rows = []
    missing_conf = set()

    for wc in canon[canon["winner"].notna()].itertuples(index=False):
        start = pd.Timestamp(wc.start)
        end = pd.Timestamp(wc.end)
        wc_matches = matches[
            (matches["tournament"] == "FIFA World Cup")
            & (matches["date"] >= start)
            & (matches["date"] <= end)
        ]
        teams = sorted(set(wc_matches["home_team"]) | set(wc_matches["away_team"]))
        if not teams:
            raise ValueError(f"No World Cup match rows found for {wc.wc_year}")
        if wc.winner not in teams:
            raise ValueError(f"Winner {wc.winner!r} missing from World Cup {wc.wc_year}")

        prior = elo_history[elo_history["date"] < start]
        prior = prior.sort_values("date").groupby("team").tail(1).set_index("team")["elo"]
        aligned = alignment[alignment["wc_year"] == wc.wc_year]
        aligned_champions = dict(
            zip(aligned["confederation"], aligned["champion"])
        )

        for position, team in enumerate(teams):
            confederation = conf_lookup.get(team, "Other")
            if confederation not in CONFEDERATIONS:
                missing_conf.add(team)
                continue
            champion = aligned_champions.get(confederation)
            rows.append(
                {
                    "wc_year": int(wc.wc_year),
                    "team": team,
                    "confederation": confederation,
                    "elo_pre_wc": float(prior.get(team, 1500.0)),
                    "continental_champion": int(
                        champion not in (None, "UNKNOWN") and team == champion
                    ),
                    "winner_position": position,
                    "is_winner": int(team == wc.winner),
                }
            )

    frame = pd.DataFrame(rows)
    if missing_conf:
        print("WARNING: excluded teams without a modeled confederation:", sorted(missing_conf))
    frame["elo_z"] = (frame["elo_pre_wc"] - frame["elo_pre_wc"].mean()) / frame[
        "elo_pre_wc"
    ].std()
    return frame


def build_prior_model(frame):
    groups = []
    for wc_year, group in frame.groupby("wc_year", sort=True):
        group = group.reset_index(drop=True)
        groups.append((int(wc_year), group))

    coords = {"confederation": CONFEDERATIONS}
    with pm.Model(coords=coords) as model:
        beta_elo = pm.Normal("beta_elo", mu=1.0, sigma=0.5)
        mu_cc = pm.Normal("mu_cc", mu=0.20, sigma=0.50)
        sigma_cc = pm.HalfNormal("sigma_cc", sigma=0.50)
        # non-centered hierarchical forms (reduce NUTS divergences on small data)
        cc_raw = pm.Normal("cc_raw", mu=0.0, sigma=1.0, dims="confederation")
        cc_effect = pm.Deterministic(
            "cc_effect", mu_cc + sigma_cc * cc_raw, dims="confederation"
        )
        sigma_conf = pm.HalfNormal("sigma_conf", sigma=0.50)
        conf_raw = pm.Normal("conf_raw", mu=0.0, sigma=1.0, dims="confederation")
        conf_offset = pm.Deterministic(
            "conf_offset", sigma_conf * conf_raw, dims="confederation"
        )

        for wc_year, group in groups:
            conf_idx = np.array([CONFEDERATIONS.index(x) for x in group["confederation"]])
            logits = (
                beta_elo * group["elo_z"].to_numpy()
                + conf_offset[conf_idx]
                + cc_effect[conf_idx] * group["continental_champion"].to_numpy()
            )
            probabilities = pm.math.softmax(logits)
            pm.Deterministic(f"p_{wc_year}", probabilities)
            pm.Categorical(
                f"winner_{wc_year}",
                p=probabilities,
                observed=int(group.index[group["is_winner"] == 1][0]),
            )
    return model


def build_gp_prior(years):
    x = ((years - years.mean()) / years.std()).to_numpy()[:, None]
    with pm.Model() as model:
        amplitude = pm.HalfNormal("amplitude", sigma=0.70)
        length_scale = pm.Gamma("length_scale", alpha=2.0, beta=1.0)
        covariance = amplitude**2 * pm.gp.cov.ExpQuad(1, ls=length_scale)
        gp = pm.gp.Latent(cov_func=covariance)
        gp.prior("f", X=x)
    return model, x


def save_prior_probability_plot(idata, frame, path):
    wc_year = int(frame["wc_year"].max())
    group = frame[frame["wc_year"] == wc_year].reset_index(drop=True)
    samples = idata.prior[f"p_{wc_year}"].stack(sample=("chain", "draw")).values
    means = samples.mean(axis=1)
    lows = np.quantile(samples, 0.05, axis=1)
    highs = np.quantile(samples, 0.95, axis=1)
    order = np.argsort(means)[::-1]
    order = order[: min(12, len(order))]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    y = np.arange(len(order))
    ax.errorbar(
        means[order],
        y,
        xerr=[means[order] - lows[order], highs[order] - means[order]],
        fmt="o",
        color="#D95F02",
        ecolor="#7F2704",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(group.loc[order, "team"])
    ax.invert_yaxis()
    ax.set_xlabel("Prior probability of winning")
    ax.set_title(f"Prior predictive probabilities — World Cup {wc_year}\n90% prior intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_cc_prior_plot(idata, path):
    samples = idata.prior["cc_effect"].stack(sample=("chain", "draw"))
    means = samples.mean("sample").values
    lows = samples.quantile(0.05, dim="sample").values
    highs = samples.quantile(0.95, dim="sample").values
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    y = np.arange(len(order))
    ax.errorbar(
        means[order],
        y,
        xerr=[means[order] - lows[order], highs[order] - means[order]],
        fmt="o",
        color="#1B7837",
        ecolor="#276419",
        capsize=3,
    )
    ax.axvline(0, color="#444444", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(np.array(CONFEDERATIONS)[order])
    ax.set_xlabel("Prior log-odds shift for continental champion")
    ax.set_title("Hierarchical prior — continental champion effect\n90% prior intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_gp_prior_plot(idata, years, path):
    samples = idata.prior["f"].stack(sample=("chain", "draw")).values
    mean = samples.mean(axis=1)
    low = np.quantile(samples, 0.05, axis=1)
    high = np.quantile(samples, 0.95, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    for draw in samples[:, : min(60, samples.shape[1])].T:
        ax.plot(years, draw, color="#2C7FB8", alpha=0.08, lw=0.8)
    ax.fill_between(years, low, high, color="#74A9CF", alpha=0.30, label="90% prior interval")
    ax.plot(years, mean, color="#045A8D", lw=2, label="prior mean")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("World Cup year")
    ax.set_ylabel("Latent GP effect")
    ax.set_title("Gaussian-process prior predictive draws")
    ax.legend(frameon=False)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def push_to_github(repo, token, files_root, stage_root):
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
    shutil.copytree(files_root, os.path.join(clone_dir, "data"), dirs_exist_ok=True)
    shutil.copytree(stage_root, os.path.join(clone_dir, "stages", "02_priors"), dirs_exist_ok=True)
    subprocess.run(["git", "add", "-A"], cwd=clone_dir, env=env, check=True)
    subprocess.run(
        ["git", "commit", "-m", "stage 02: priors and prior predictive checks"],
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
    stage_dir = os.path.join(out_root, "stages", "02_priors")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    print("Python:", sys.version.split()[0], "| PyMC:", pm.__version__)
    canon = fetch_csv("historical_canon.csv")
    alignment = fetch_csv("alignment_engine.csv")
    team_confederations = fetch_csv("team_confederations.csv")
    matches = pd.read_csv(io.BytesIO(fetch_bytes(MATCH_SOURCE)))
    frame = build_model_frame(matches, canon, alignment, team_confederations)
    frame.to_csv(os.path.join(data_dir, "prior_model_frame.csv"), index=False)
    print("model frame:", len(frame), "rows;", frame["wc_year"].nunique(), "World Cups")

    model = build_prior_model(frame)
    with model:
        prior = pm.sample_prior_predictive(draws=600, random_seed=20260815)
    az.to_netcdf(prior, os.path.join(data_dir, "prior.nc"))
    save_prior_probability_plot(
        prior,
        frame,
        os.path.join(data_dir, "prior_predictive_probabilities.png"),
    )
    save_cc_prior_plot(prior, os.path.join(data_dir, "prior_continental_effects.png"))

    years = canon.loc[canon["winner"].notna(), "wc_year"].astype(float)
    gp_model, _ = build_gp_prior(years)
    with gp_model:
        gp_prior = pm.sample_prior_predictive(draws=400, random_seed=20260815)
    az.to_netcdf(gp_prior, os.path.join(data_dir, "gp_prior.nc"))
    save_gp_prior_plot(gp_prior, years.to_numpy(), os.path.join(data_dir, "gp_prior.png"))

    beta = prior.prior["beta_elo"].values.ravel()
    cc = prior.prior["mu_cc"].values.ravel()
    report = [
        "# PRIOR PREDICTIVE REPORT — stage 02",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model frame: {len(frame)} team-World Cup rows across {frame['wc_year'].nunique()} editions",
        "- Outcome likelihood: one categorical World Cup winner per edition (softmax)",
        "- Prior draws: 600 winner-model draws; 400 GP prior draws",
        "",
        "## Priors",
        "- `beta_elo ~ Normal(1.0, 0.5)` on standardized pre-tournament Elo",
        "- `mu_cc ~ Normal(0.20, 0.50)`; `sigma_cc ~ HalfNormal(0.50)`",
        "- `cc_effect[confederation] ~ Normal(mu_cc, sigma_cc)`",
        "- `sigma_conf ~ HalfNormal(0.50)`; confederation offsets are hierarchical",
        "- GP amplitude `~ HalfNormal(0.70)`; length scale `~ Gamma(2, 1)`",
        "",
        "## Sanity summary",
        f"- beta_elo prior mean/90% interval: {beta.mean():.3f} "
        f"[{np.quantile(beta, 0.05):.3f}, {np.quantile(beta, 0.95):.3f}]",
        f"- mean continental effect prior mean/90% interval: {cc.mean():.3f} "
        f"[{np.quantile(cc, 0.05):.3f}, {np.quantile(cc, 0.95):.3f}]",
        "- Prior probabilities are shown as intervals, never point predictions.",
        "- The prior model is deliberately weak enough that Elo informs ranking without",
        "  making the champion feature deterministic.",
        "",
        "## Artifacts",
        "- `prior.nc`: complete prior/prior-predictive InferenceData",
        "- `gp_prior.nc`: GP prior InferenceData",
        "- `prior_predictive_probabilities.png`: 2022 prior winner probabilities",
        "- `prior_continental_effects.png`: hierarchical champion-effect priors",
        "- `gp_prior.png`: prior GP trajectories",
        "- `prior_model_frame.csv`: exact frame used by the prior model",
        "",
        "## Gate notes",
        "- AFCON 2025 resolved as Morocco and Gold Cup 2025 resolved as Mexico from the",
        "  pinned match source; both now feed the 2026 alignment table.",
    ]
    with open(os.path.join(data_dir, "prior_predictive_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    source_file = "/content/stage02_priors.py"
    if os.path.exists(source_file):
        shutil.copyfile(source_file, os.path.join(stage_dir, "stage02_priors.py"))
    else:
        try:
            shutil.copyfile(__file__, os.path.join(stage_dir, "stage02_priors.py"))
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
