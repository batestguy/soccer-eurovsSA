# -*- coding: utf-8 -*-
# =====================================================================
# STAGE 04 — MONTE CARLO ORACLE + do()-CONTRAST (retrospective replay)
# Bayesian World Cup Prediction
#
# Locked spec (SESSION_HANDOFF.md §4):
#   - 2026 field derived from World Cup match participation ONLY.
#   - Elo frozen strictly before 2026-06-11 (no post-tournament leakage).
#   - 50,000 simulated World Cups from the posterior softmax model.
#   - do(continental_champion=1) vs do(0): counterfactual simulation,
#     NEVER labeled an estimated causal effect (see dag_validation.md).
# =====================================================================

import base64
import datetime as dt
import io
import json
import os
import shutil
import subprocess
import urllib.request

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GITHUB_REPO = "batestguy/soccer-eurovsSA"
DATA_ROOT = "https://raw.githubusercontent.com/batestguy/soccer-eurovsSA/main/data"
MATCH_SOURCE = f"{DATA_ROOT}/source/results_65d212a.csv"
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]
CUTOFF = pd.Timestamp("2026-06-11")   # Elo frozen before this instant
N_SIMS = 50_000
SEED = 20260815


def fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "soccerdl-stage04"})
    return urllib.request.urlopen(req, timeout=180).read()


def fetch_csv(name):
    return pd.read_csv(io.BytesIO(fetch_bytes(f"{DATA_ROOT}/{name}")))


def load_stage02_module():
    source = fetch_bytes(
        "https://raw.githubusercontent.com/batestguy/soccer-eurovsSA/main/"
        "stages/02_priors/stage02_priors.py"
    )
    ns = {"__name__": "stage02_module"}
    exec(compile(source, "stage02_priors.py", "exec"), ns)
    return ns


def load_posterior():
    tmp = "/tmp/posterior_fetch04.nc"
    with open(tmp, "wb") as f:
        f.write(fetch_bytes(f"{DATA_ROOT}/posterior.nc"))
    return az.from_netcdf(tmp)


def derive_2026_field(matches, elo_history, conf_lookup, champion_by_conf):
    """Teams that played in the 2026 World Cup; Elo as of pre-tournament freeze."""
    start, end = CUTOFF, pd.Timestamp("2026-07-19")
    wc = matches[(matches["tournament"] == "FIFA World Cup") &
                 (matches["date"] >= start) & (matches["date"] <= end)]
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
    if not teams:
        raise ValueError("No 2026 World Cup matches found in source")

    prior = elo_history[elo_history["date"] < CUTOFF]
    prior = prior.sort_values("date").groupby("team").tail(1).set_index("team")["elo"]

    rows = []
    for team in teams:
        conf = conf_lookup.get(team, "Other")
        rows.append({
            "team": team,
            "confederation": conf,
            "elo_pre_wc": float(prior.get(team, 1500.0)),
            "continental_champion": int(champion_by_conf.get(conf) == team),
        })
    field = pd.DataFrame(rows)
    if (field["confederation"] == "Other").any():
        print("WARNING: 2026 teams without a modeled confederation:",
              field.loc[field["confederation"] == "Other", "team"].tolist())
    return field


def softmax_rows(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


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
    shutil.copytree(stage_dir, os.path.join(clone_dir, "stages", "04_simulator"), dirs_exist_ok=True)
    subprocess.run(["git", "add", "-A"], cwd=clone_dir, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "stage 04: Monte Carlo Oracle and do()-contrast"],
                   cwd=clone_dir, env=env, check=True, capture_output=True, text=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, env=env, check=True)


def save_barchart(field, res, path):
    order = res["p_win_mean"].sort_values(ascending=False).head(20).index[::-1]
    fig, ax = plt.subplots(figsize=(9, 10), dpi=170)
    y = np.arange(len(order))
    ax.errorbar(res.loc[order, "p_win_mean"].values * 100, y,
                xerr=[(res.loc[order, "p_win_mean"] - res.loc[order, "p_win_p5"]).values * 100,
                      (res.loc[order, "p_win_p95"] - res.loc[order, "p_win_mean"]).values * 100],
                fmt="o", color="#D95F02", ecolor="#7F2704", capsize=3, ms=6)
    ax.set_yticks(y)
    ax.set_yticklabels(field.loc[order, "team"])
    ax.invert_yaxis()
    ax.set_xlabel("Probability of winning the 2026 World Cup (%)")
    ax.set_title("Monte Carlo Oracle — 2026 retrospective replay\n"
                 "50,000 simulations, 90% credible intervals", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_do_contrast(field, contr, path):
    order = contr["diff_mean"].sort_values(ascending=False).head(20).index[::-1]
    fig, ax = plt.subplots(figsize=(9, 10), dpi=170)
    y = np.arange(len(order))
    ax.errorbar(contr.loc[order, "diff_mean"].values * 100, y,
                xerr=[(contr.loc[order, "diff_mean"] - contr.loc[order, "diff_p5"]).values * 100,
                      (contr.loc[order, "diff_p95"] - contr.loc[order, "diff_mean"]).values * 100],
                fmt="o", color="#1B7837", ecolor="#276419", capsize=3, ms=6)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(field.loc[order, "team"])
    ax.invert_yaxis()
    ax.set_xlabel("Δ P(win) = P(do(champion=1)) − P(do(champion=0))  (pp)")
    ax.set_title("do()-contrast — continental champion on/off (counterfactual simulation)\n"
                 "90% intervals; NOT an estimated causal effect", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    out_root = "/content/soccerdl_out" if os.path.exists("/tmp/github_token") else "output"
    data_dir = os.path.join(out_root, "data")
    stage_dir = os.path.join(out_root, "stages", "04_simulator")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    print("Stage 04 — Monte Carlo Oracle (retrospective replay)")
    stage02 = load_stage02_module()
    compute_elo_history = stage02["compute_elo_history"]

    matches = pd.read_csv(io.BytesIO(fetch_bytes(MATCH_SOURCE)))
    matches["date"] = pd.to_datetime(matches["date"])
    matches["neutral"] = matches["neutral"].astype(str).str.upper().eq("TRUE")
    matches["tournament"] = matches["tournament"].fillna("").astype(str)

    team_conf = fetch_csv("team_confederations.csv")
    conf_lookup = team_conf.set_index("team")["confederation"].to_dict()
    alignment = fetch_csv("alignment_engine.csv")
    champ_2026 = alignment[alignment["wc_year"] == 2026]
    champion_by_conf = dict(zip(champ_2026["confederation"], champ_2026["champion"]))
    print("2026 aligned champions:", champion_by_conf)

    # standardization from the training frame (same scale as prior_model_frame)
    frame = fetch_csv("prior_model_frame.csv")
    mu = float(frame["elo_pre_wc"].mean())
    sd = float(frame["elo_pre_wc"].std())

    print("computing full-history Elo (frozen before 2026-06-11) ...")
    elo_history = compute_elo_history(matches)
    field = derive_2026_field(matches, elo_history, conf_lookup, champion_by_conf)
    field["elo_z"] = (field["elo_pre_wc"] - mu) / sd
    print("2026 field teams:", len(field))
    field.to_csv(os.path.join(data_dir, "sim_field_2026.csv"), index=False)

    # posterior parameter draws
    print("loading posterior ...")
    idata = load_posterior()
    post = idata.posterior
    beta_all = post["beta_elo"].values.reshape(-1)
    conf_off_all = post["conf_offset"].values.reshape(-1, len(CONFEDERATIONS))
    cc_eff_all = post["cc_effect"].values.reshape(-1, len(CONFEDERATIONS))
    n_post = beta_all.shape[0]
    print(f"posterior draws available: {n_post}")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_post, size=N_SIMS)
    elo_z = field["elo_z"].to_numpy()
    conf_idx = np.array([CONFEDERATIONS.index(c) for c in field["confederation"]])
    cc_actual = field["continental_champion"].to_numpy(dtype=float)

    def p_matrix(cc_vals):
        logits = (beta_all[idx][:, None] * elo_z[None, :]
                  + conf_off_all[idx][:, conf_idx]
                  + cc_eff_all[idx][:, conf_idx] * cc_vals[None, :])
        return softmax_rows(logits)

    print(f"simulating {N_SIMS:,} World Cups x3 (natural, do1, do0) ...")
    p_nat = p_matrix(cc_actual)
    p_do1 = p_matrix(np.ones_like(cc_actual))
    p_do0 = p_matrix(np.zeros_like(cc_actual))

    u = rng.random((N_SIMS, 1))
    winners = np.sum(np.cumsum(p_nat, axis=1) < u, axis=1)
    win_freq = np.bincount(winners, minlength=len(field)) / N_SIMS

    res = pd.DataFrame({
        "team": field["team"],
        "confederation": field["confederation"],
        "elo_pre_wc": field["elo_pre_wc"].round(1),
        "continental_champion": field["continental_champion"].astype(int),
        "p_win_mean": p_nat.mean(axis=0),
        "p_win_p5": np.quantile(p_nat, 0.05, axis=0),
        "p_win_p95": np.quantile(p_nat, 0.95, axis=0),
        "win_freq": win_freq,
    })
    res = res.sort_values("p_win_mean", ascending=False).reset_index(drop=True)
    res.to_csv(os.path.join(data_dir, "monte_carlo_results.csv"), index=False)

    diff = p_do1 - p_do0
    contr = pd.DataFrame({
        "team": field["team"],
        "confederation": field["confederation"],
        "p_win_do1": p_do1.mean(axis=0),
        "p_win_do0": p_do0.mean(axis=0),
        "diff_mean": diff.mean(axis=0),
        "diff_p5": np.quantile(diff, 0.05, axis=0),
        "diff_p95": np.quantile(diff, 0.95, axis=0),
    })
    contr = contr.sort_values("diff_mean", ascending=False).reset_index(drop=True)
    contr.to_csv(os.path.join(data_dir, "do_contrast.csv"), index=False)

    save_barchart(field, res, os.path.join(data_dir, "monte_carlo_barchart.png"))
    save_do_contrast(field, contr, os.path.join(data_dir, "do_contrast.png"))

    config = {
        "label": "retrospective pre-tournament replay",
        "n_simulations": N_SIMS,
        "random_seed": SEED,
        "cutoff_elo": str(CUTOFF.date()),
        "field_source": "2026 World Cup match participation (pinned source)",
        "posterior_source": "data/posterior.nc",
        "n_posterior_draws": int(n_post),
        "model": "softmax winner model: logit = beta_elo*elo_z + conf_offset[conf] + cc_effect[conf]*champion",
        "standardization": {"mean": float(mu), "std": float(sd)},
        "do_contrast": "P(win|do(champion=1)) - P(win|do(champion=0)); counterfactual simulation, not causal",
    }
    with open(os.path.join(data_dir, "simulation_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ---- report ---------------------------------------------------------
    def pct(x):
        return f"{x * 100:.1f}%"

    top = res.head(12)
    conf_sum = res.groupby("confederation")["p_win_mean"].sum().sort_values(ascending=False)
    conf_freq = pd.Series(
        np.bincount([conf_idx[int(w)] for w in winners], minlength=len(CONFEDERATIONS)) / N_SIMS,
        index=CONFEDERATIONS).sort_values(ascending=False)
    do_top = contr.head(5)
    do_bot = contr.tail(5)[::-1]

    lines = [
        "# MONTE CARLO ORACLE — STAGE 04 REPORT",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Retrospective pre-tournament replay of the 2026 World Cup.",
        f"- Field: {len(field)} teams from match participation; Elo frozen before {CUTOFF.date()}.",
        f"- Simulations: {N_SIMS:,} per scenario (natural, do(champion=1), do(champion=0)).",
        f"- Posterior draws sampled with replacement from posterior.nc ({n_post} draws).",
        "",
        "## No-leakage statement",
        "- Team list uses participation only (no scores, no results).",
        "- Elo recomputed from full match history and frozen strictly before 2026-06-11.",
        "- 2026 continental champions (resolved): " +
          ", ".join(f"{c}={v}" for c, v in champion_by_conf.items()),
        "",
        "## Headline — P(win) with 90% interval and empirical winner frequency",
        "| # | Team | Conf | P(win) | 90% interval | Win freq (50k sims) |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in top.iterrows():
        lines.append(f"| {i + 1} | {r['team']} | {r['confederation']} | "
                     f"{pct(r['p_win_mean'])} | {pct(r['p_win_p5'])}–{pct(r['p_win_p95'])} | "
                     f"{pct(r['win_freq'])} |")
    lines += [
        "",
        "## Confederation win probability (sum of team probs) and winner frequency",
        "| Confederation | P(a team from conf wins) | Empirical winner conf freq |",
        "|---|---|---|",
    ]
    for conf in conf_sum.index:
        lines.append(f"| {conf} | {pct(conf_sum[conf])} | {pct(conf_freq[conf])} |")
    lines += [
        "",
        "## do()-contrast (counterfactual simulation, NOT causal)",
        "Δ = P(win | do(champion=1)) − P(win | do(champion=0)) for every team, 50k sims each.",
        "Largest positive:",
        "| Team | Conf | Δ mean | 90% interval |",
        "|---|---|---|---|",
    ]
    for r in do_top.iterrows():
        lines.append(f"| {r[1]['team']} | {r[1]['confederation']} | "
                     f"{(r[1]['diff_mean'] * 100):+.2f}pp | "
                     f"{(r[1]['diff_p5'] * 100):+.2f} to {(r[1]['diff_p95'] * 100):+.2f}pp |")
    lines += ["Most negative:"]
    for r in do_bot.iterrows():
        lines.append(f"| {r[1]['team']} | {r[1]['confederation']} | "
                     f"{(r[1]['diff_mean'] * 100):+.2f}pp | "
                     f"{(r[1]['diff_p5'] * 100):+.2f} to {(r[1]['diff_p95'] * 100):+.2f}pp |")
    lines += [
        "",
        "## Honesty statement",
        "These are posterior-predictive simulations from the Stage 03 softmax model. Because the",
        "Stage 01 DAG shows `continental_champion` is confounded by latent team strength and every",
        "back-door adjustment set requires the unmeasured variable, the do()-contrast is a",
        "**counterfactual simulation**, not an estimated causal effect. All quantities are",
        "distributions with credible intervals — never point predictions.",
        "",
        "## Artifacts",
        "- `monte_carlo_results.csv`, `do_contrast.csv`, `sim_field_2026.csv`",
        "- `monte_carlo_barchart.png`, `do_contrast.png`",
        "- `simulation_config.json`, `stage04_report.md`",
    ]
    with open(os.path.join(data_dir, "stage04_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    src = "/content/stage04_simulator.py"
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(stage_dir, "stage04_simulator.py"))

    print("artifacts:", ", ".join(sorted(os.listdir(data_dir))))
    print("top-5:", ", ".join(f"{r.team} {r.p_win_mean * 100:.1f}%" for r in top.head(5).itertuples()))
    if os.path.exists("/tmp/github_token"):
        token = open("/tmp/github_token", encoding="utf-8").read().splitlines()[0].strip()
        push_to_github(GITHUB_REPO, token, data_dir, stage_dir)
        print("commit: True | pushed to", GITHUB_REPO)
    else:
        print("commit: skipped (no /tmp/github_token)")


if __name__ == "__main__":
    main()
