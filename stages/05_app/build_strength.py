# -*- coding: utf-8 -*-
# =====================================================================
# STAGE 05 BUILD 1 — CONTINENTAL STRENGTH
# Hierarchical panel random-effects on log(monthly Elo):
#   log(elo_it) ~ N(alpha_0 + alpha_c[i,t] + u_i + S(t).theta_c[i,t], sigma)
#   alpha_c ~ N(0, sigma_continent)   continent strength -> ranking / pairwise
#   u_i    ~ N(0, sigma_team)         team effects (nested in continent)
#   S(t)   natural-cubic spline basis (df fixed) x continent (non-centered)
# Historical confederation membership handled via conf_membership.csv.
#
# Outputs (pushed to GitHub): conf_membership.csv, strength_posterior.nc,
# strength_pairwise.csv, strength_ranking.png, strength_report.md,
# strength_model_meta.json, strength_trends.csv, strength_diff_curves.csv.
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
import pymc as pm
import pytensor.tensor as pt

GITHUB_REPO = "batestguy/soccer-eurovsSA"
DATA_ROOT = "https://raw.githubusercontent.com/batestguy/soccer-eurovsSA/main/data"
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]
SPLINE_DF = 8
FAST = os.environ.get("SOCCERDL_FAST") == "1"


def _has_numpyro():
    try:
        import numpyro  # noqa: F401
        return True
    except Exception:
        return False

# Membership overrides: (team, first_month_inclusive, confederation_from_then_on)
# months before `start` use the "before" confederation (or Other to exclude).
MEMBERSHIP = {
    "Australia":  {"start": "2006-01", "after": "AFC", "before": "OFC"},
    "Israel":     {"start": "1994-07", "after": "UEFA", "before": "Other"},
    "Kazakhstan": {"start": "2002-01", "after": "UEFA", "before": "AFC"},
}


def fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "soccerdl-strength"})
    return urllib.request.urlopen(req, timeout=180).read()


def fetch_csv(name):
    return pd.read_csv(io.BytesIO(fetch_bytes(f"{DATA_ROOT}/{name}")))


# ----------------------------------------------------------------------
# B-spline basis (Cox-de Boor), degree 3, boundary knots repeated
# ----------------------------------------------------------------------
def _bspline_basis(x, i, k, t):
    x = np.asarray(x, dtype=float)
    if k == 0:
        return ((x >= t[i]) & (x < t[i + 1])).astype(float)
    denom_a = t[i + k] - t[i]
    denom_b = t[i + k + 1] - t[i + 1]
    a = (x - t[i]) / denom_a if denom_a != 0 else np.zeros_like(x)
    b = (t[i + k + 1] - x) / denom_b if denom_b != 0 else np.zeros_like(x)
    return a * _bspline_basis(x, i, k - 1, t) + b * _bspline_basis(x, i + 1, k - 1, t)


def bspline_basis(x, n_basis, degree=3):
    """Cubic regression-spline basis with boundary knots at xmin/xmax.
    Knots: n_basis - degree interior quantile knots, boundaries repeated (degree+1) times."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    n_interior = n_basis - degree - 1
    interior = np.quantile(x, np.linspace(0, 1, n_interior + 2)[1:-1]) if n_interior > 0 else np.array([])
    knots = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
    n = len(knots) - degree - 1
    cols = [_bspline_basis(x, i, degree, knots) for i in range(n)]
    basis = np.column_stack(cols)
    # fix the right boundary (x == hi should contribute to the last basis)
    basis[x >= hi] = 0.0
    basis[x >= hi, -1] = 1.0
    return basis, knots


# ----------------------------------------------------------------------
# panel assembly
# ----------------------------------------------------------------------
def build_panel(chronology, team_conf):
    rows = []
    conf_lookup = team_conf.set_index("team")["confederation"].to_dict()
    overrides = {t: (pd.Period(m["start"], "M"), m["after"], m["before"]) for t, m in MEMBERSHIP.items()}
    for row in chronology.itertuples(index=False):
        month = pd.Period(row.month, "M")
        base = conf_lookup.get(row.team, "Other")
        conf = base
        if row.team in overrides:
            start, after, before = overrides[row.team]
            conf = after if month >= start else before
        if conf not in CONFEDERATIONS:
            continue
        rows.append((month, row.team, conf, np.log(row.elo)))
    panel = pd.DataFrame(rows, columns=["month", "team", "confederation", "log_elo"])
    return panel


# ----------------------------------------------------------------------
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
    subprocess.run(["git", "commit", "-m", "stage 05 build: continental strength model"],
                   cwd=clone_dir, env=env, check=True, capture_output=True, text=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, env=env, check=True)


def main():
    out_root = "/content/soccerdl_out" if os.path.exists("/tmp/github_token") else "output"
    data_dir = os.path.join(out_root, "data")
    stage_dir = os.path.join(out_root, "stages", "05_app")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    print("Python:", sys.version.split()[0], "| PyMC:", pm.__version__, "| fast:", FAST)

    # ---- panel ---------------------------------------------------------
    chronology = fetch_csv("ranking_chronology.csv")
    team_conf = fetch_csv("team_confederations.csv")
    panel = build_panel(chronology, team_conf)
    membership_rows = [{"team": t, "change_month": m["start"], "conf_after": m["after"],
                        "conf_before": m["before"]} for t, m in MEMBERSHIP.items()]
    pd.DataFrame(membership_rows).to_csv(os.path.join(data_dir, "conf_membership.csv"), index=False)
    print("panel rows:", len(panel), "teams:", panel["team"].nunique(),
          "months:", panel["month"].nunique())

    # time + spline
    panel["t"] = panel["month"].map(lambda p: p.year + (p.month - 1) / 12.0)
    t_mean, t_std = panel["t"].mean(), panel["t"].std()
    t_std_v = (panel["t"] - t_mean) / t_std
    design, knots = bspline_basis(t_std_v.to_numpy(), SPLINE_DF)
    panel["team_id"] = pd.factorize(panel["team"])[0]
    panel["conf_idx"] = panel["confederation"].map(lambda c: CONFEDERATIONS.index(c)).to_numpy()
    teams = sorted(panel["team"].unique())
    team_map = {t: i for i, t in enumerate(teams)}

    if FAST:  # smoke: subsample rows for a tiny fit
        panel = panel.sample(n=min(4000, len(panel)), random_state=0).sort_values("month")
        design = bspline_basis(((panel["t"] - t_mean) / t_std).to_numpy(), SPLINE_DF)[0]
        panel = panel.reset_index(drop=True)

    n = len(panel)
    log_elo = panel["log_elo"].to_numpy()
    cont_idx = panel["conf_idx"].to_numpy()
    team_idx = panel["team_id"].to_numpy()
    print("design shape:", design.shape, "df:", SPLINE_DF)

    # ---- model ----------------------------------------------------------
    coords = {"continent": CONFEDERATIONS, "spline": range(SPLINE_DF),
              "team": teams}
    with pm.Model(coords=coords) as model:
        alpha0 = pm.Normal("alpha0", mu=float(np.log(1500.0)), sigma=0.5)
        z_alpha = pm.Normal("z_alpha", 0, 1, dims="continent")
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.20)
        alpha_c = pm.Deterministic("alpha_c", z_alpha * sigma_alpha, dims="continent")
        z_team = pm.Normal("z_team", 0, 1, dims="team")
        sigma_team = pm.HalfNormal("sigma_team", 0.30)
        u_i = pm.Deterministic("u_i", z_team * sigma_team, dims="team")
        z_theta = pm.Normal("z_theta", 0, 1, dims=("continent", "spline"))
        sigma_spline = pm.HalfNormal("sigma_spline", 0.05)
        theta_c = pm.Deterministic("theta_c", z_theta * sigma_spline, dims=("continent", "spline"))
        sigma_resid = pm.HalfNormal("sigma_resid", 0.15)
        spline_design = pt.as_tensor_variable(design)
        mu = (alpha0 + alpha_c[cont_idx] + u_i[team_idx]
              + pt.sum(spline_design * theta_c[cont_idx], axis=-1))
        pm.Normal("obs", mu=mu, sigma=sigma_resid, observed=log_elo)

    draws = 10 if FAST else 300
    tune = 10 if FAST else 300
    chains = 1 if FAST else 2
    sampler = "numpyro" if _has_numpyro() else None
    with model:
        kwargs = dict(draws=draws, tune=tune, chains=chains, cores=1,
                      target_accept=0.90, random_seed=20260816, progressbar=False)
        if sampler:
            kwargs["nuts_sampler"] = sampler
        posterior = pm.sample(**kwargs)
    az.to_netcdf(posterior, os.path.join(data_dir, "strength_posterior.nc"))
    print("saved strength_posterior.nc")

    # ---- derived quantities ----------------------------------------------
    months = pd.period_range("1992-01", "2026-08", freq="M")
    grid_t = np.array([p.year + (p.month - 1) / 12.0 for p in months])
    grid_std = (grid_t - t_mean) / t_std
    grid_design, _ = bspline_basis(grid_std, SPLINE_DF)

    post = posterior.posterior
    alpha_c_post = post["alpha_c"].values.reshape(-1, len(CONFEDERATIONS))      # (S,6)
    theta_post = post["theta_c"].values.reshape(-1, len(CONFEDERATIONS), SPLINE_DF)  # (S,6,df)
    alpha0_post = post["alpha0"].values.reshape(-1)
    S = alpha_c_post.shape[0]
    log_trend = (alpha0_post[:, None, None]
                 + alpha_c_post[:, :, None]
                 + np.einsum("tj,scj->sct", grid_design, theta_post))            # (S,6,T) log-elo
    elo_trend = np.exp(log_trend)

    # trends table
    trend_rows = []
    for c, conf in enumerate(CONFEDERATIONS):
        for j, month in enumerate(months):
            lo_l, hi_l = np.quantile(log_trend[:, c, j], [0.05, 0.95])
            lo_e, hi_e = np.quantile(elo_trend[:, c, j], [0.05, 0.95])
            trend_rows.append({"confederation": conf, "month": str(month),
                               "log_mean": log_trend[:, c, j].mean(),
                               "log_p5": lo_l, "log_p95": hi_l,
                               "elo_mean": elo_trend[:, c, j].mean(),
                               "elo_p5": lo_e, "elo_p95": hi_e})
    pd.DataFrame(trend_rows).to_csv(os.path.join(data_dir, "strength_trends.csv"), index=False)

    # pairwise: level (alpha) + overall-average (over grid) differences
    pairs = [(a, b) for a in range(6) for b in range(a + 1, 6)]
    pair_rows = []
    diff_rows = []
    for (a, b) in pairs:
        na, nb = CONFEDERATIONS[a], CONFEDERATIONS[b]
        level = alpha_c_post[:, a] - alpha_c_post[:, b]                          # (S,)
        avg_log = (log_trend[:, a, :] - log_trend[:, b, :]).mean(axis=1)         # (S,)
        avg_elo = (elo_trend[:, a, :] - elo_trend[:, b, :]).mean(axis=1)
        pair_rows.append({
            "pair": f"{na} vs {nb}", "continent_a": na, "continent_b": nb,
            "level_mean": level.mean(), "level_p5": np.quantile(level, 0.05),
            "level_p95": np.quantile(level, 0.95),
            "avg_log_diff_mean": avg_log.mean(), "avg_log_diff_p5": np.quantile(avg_log, 0.05),
            "avg_log_diff_p95": np.quantile(avg_log, 0.95),
            "avg_elo_diff_mean": avg_elo.mean(), "avg_elo_diff_p5": np.quantile(avg_elo, 0.05),
            "avg_elo_diff_p95": np.quantile(avg_elo, 0.95),
            "p_a_stronger": float((avg_log > 0).mean()),
        })
        for j, month in enumerate(months):
            d_log = log_trend[:, a, j] - log_trend[:, b, j]
            d_elo = elo_trend[:, a, j] - elo_trend[:, b, j]
            diff_rows.append({"pair": f"{na} vs {nb}", "continent_a": na, "continent_b": nb,
                              "month": str(month),
                              "log_mean": d_log.mean(), "log_p5": np.quantile(d_log, 0.05),
                              "log_p95": np.quantile(d_log, 0.95),
                              "elo_mean": d_elo.mean(), "elo_p5": np.quantile(d_elo, 0.05),
                              "elo_p95": np.quantile(d_elo, 0.95)})
    pairwise = pd.DataFrame(pair_rows).sort_values("avg_log_diff_mean", ascending=False)
    pairwise.to_csv(os.path.join(data_dir, "strength_pairwise.csv"), index=False)
    pd.DataFrame(diff_rows).to_csv(os.path.join(data_dir, "strength_diff_curves.csv"), index=False)

    # ranking plot
    order = np.argsort(alpha_c_post.mean(axis=0))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    y = np.arange(6)
    lo = np.quantile(alpha_c_post, 0.05, axis=0)[order]
    hi = np.quantile(alpha_c_post, 0.95, axis=0)[order]
    mean = alpha_c_post.mean(axis=0)[order]
    ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o", color="#1B7837",
                ecolor="#276419", capsize=4, ms=7)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([CONFEDERATIONS[i] for i in order])
    ax.set_xlabel("continent effect on log(monthly Elo)")
    ax.set_title("Continental strength (log-Elo random intercepts), 90% HDI")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(data_dir, "strength_ranking.png"), bbox_inches="tight")
    plt.close(fig)

    # meta
    meta = {"model": "hierarchical panel random effects on log(monthly Elo)",
            "spline_df": SPLINE_DF, "knots_std": knots.tolist(),
            "time_mean": float(t_mean), "time_std": float(t_std),
            "continents": CONFEDERATIONS, "months": [str(m) for m in months],
            "n_rows": int(n), "n_teams": int(len(teams)),
            "membership": MEMBERSHIP}
    with open(os.path.join(data_dir, "strength_model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # diagnostics + report
    summary = az.summary(posterior, var_names=["alpha_c", "sigma_alpha", "sigma_team",
                                               "sigma_spline", "sigma_resid", "alpha0"], round_to=3)
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    diverging = int(np.asarray(posterior.sample_stats["diverging"]).sum())
    head = pairwise.head(6)
    lines = [
        "# CONTINENTAL STRENGTH — STAGE 05 BUILD REPORT",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Panel: {n:,} team-month rows, {len(teams)} teams, {months.nunique()} months, log(monthly Elo).",
        f"- Model: log_elo ~ N(alpha_0 + alpha_c + u_i + S(t).theta_c, sigma); spline df={SPLINE_DF}.",
        f"- Sampling: {chains} chains x {draws} draws; divergences={diverging}; max R-hat={max_rhat:.3f}; min ESS={min_ess:.1f}.",
        "- Historical membership overrides applied (Australia, Israel, Kazakhstan).",
        "",
        "## Continental strength ranking (log-Elo random intercepts)",
        "| Continent | effect (mean) | 90% HDI |",
        "|---|---|---|",
    ]
    rk = pd.DataFrame({"c": CONFEDERATIONS, "m": alpha_c_post.mean(0),
                       "lo": np.quantile(alpha_c_post, 0.05, 0), "hi": np.quantile(alpha_c_post, 0.95, 0)})
    rk = rk.sort_values("m", ascending=False)
    for r in rk.itertuples():
        lines.append(f"| {r.c} | {r.m:.3f} | [{r.lo:.3f}, {r.hi:.3f}] |")
    lines += ["", "## Pairwise: which continent is stronger (overall average difference)",
              "A ahead of B if avg_log_diff > 0; P(A>B) = P(avg difference > 0).",
              "| Pair | avg log-diff | 90% HDI | avg Elo-pts diff | P(A>B) |",
              "|---|---|---|---|---|"]
    for r in head.itertuples():
        lines.append(f"| {r.pair} | {r.avg_log_diff_mean:+.3f} | "
                     f"[{r.avg_log_diff_p5:+.3f}, {r.avg_log_diff_p95:+.3f}] | "
                     f"{r.avg_elo_diff_mean:+.1f} | {r.p_a_stronger:.2f} |")
    lines += [
        "",
        "## Caveats",
        "- 'Effect' is structural/associational (continent is a fixed attribute); NOT a causal claim.",
        "- Differences are per the whole 1992-2026 window; the dynamic gap is in strength_diff_curves.csv.",
        "- Elo is the documented FIFA-rankings proxy; log-Elo difference translates to %-higher, and",
        "  Elo-points translation is exp() of the fitted log-trends (reference-level dependent).",
    ]
    with open(os.path.join(data_dir, "strength_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # provenance
    src = "/content/build_strength.py"
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(stage_dir, "build_strength.py"))

    print("artifacts:", ", ".join(sorted(os.listdir(data_dir))))
    print("top pair:", pairwise.iloc[0]["pair"], pairwise.iloc[0]["avg_log_diff_mean"])
    if os.path.exists("/tmp/github_token"):
        token = open("/tmp/github_token", encoding="utf-8").read().splitlines()[0].strip()
        push_to_github(GITHUB_REPO, token, data_dir, stage_dir)
        print("commit: True | pushed to", GITHUB_REPO)
    else:
        print("commit: skipped (no /tmp/github_token)")


if __name__ == "__main__":
    import sys
    main()
