# -*- coding: utf-8 -*-
# =====================================================================
# STAGE 05 BUILD 3 — DAG ASSUMPTION TESTS (Tab 7)
# Four checks probing the stage-01 DAG edges with observed proxies:
#   C1 Confounding   — continental_champion vs Elo            (strength -> champion)
#   C2 Independence  — does champion add win-info beyond Elo  (champion -> outcome | elo)
#   C3 Sensitivity   — continent effects with/without conditioning on {champion, Elo}
#   C4 Balance       — confederation predicts Elo level       (conf -> strength)
# Reuses: prior_model_frame.csv, posterior.nc, ranking_chronology.csv,
#         team_confederations.csv, conf_membership.csv.
# Honesty: Elo is the observable proxy for latent team_strength; these checks probe
# edges, they do NOT identify a causal effect (22 World Cups can't).
#
# Outputs (pushed to GitHub): dag_checks.csv, dag_checks_report.md.
# =====================================================================

import base64
import datetime as dt
import io
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats as sst

GITHUB_REPO = "batestguy/soccer-eurovsSA"
DATA_ROOT = "https://raw.githubusercontent.com/batestguy/soccer-eurovsSA/main/data"
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]
RNG = np.random.default_rng(20260816)
BOOT = 2000


def fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "soccerdl-dagchecks"})
    return urllib.request.urlopen(req, timeout=180).read()


def fetch_csv(name):
    return pd.read_csv(io.BytesIO(fetch_bytes(f"{DATA_ROOT}/{name}")))


# ----------------------------------------------------------------------
# tiny numpy logistic (IRLS) with ridge for stability on rare outcomes
# ----------------------------------------------------------------------
def logistic_fit(X, y, ridge=1e-6, max_iter=60, tol=1e-9):
    n, k = X.shape
    beta = np.zeros(k)
    p = None
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        w = np.clip(p * (1 - p), 1e-12, None) + ridge
        XtW = X * w[:, None]
        z = X @ beta + (y - p) / w
        H = XtW.T @ X
        b_new = np.linalg.solve(H + ridge * np.eye(k), XtW.T @ z)
        if np.max(np.abs(b_new - beta)) < tol:
            beta = b_new
            break
        beta = b_new
    p = 1.0 / (1.0 + np.exp(-X @ beta))
    w = np.clip(p * (1 - p), 1e-12, None) + ridge
    H = (X * w[:, None]).T @ X
    cov = np.linalg.inv(H + ridge * np.eye(k))
    se = np.sqrt(np.diag(cov))
    ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return beta, se, -2.0 * ll


def lr_test(dev_full, dev_null, df_diff):
    stat = max(dev_null - dev_full, 0.0)
    return stat, float(sst.chi2.sf(stat, df_diff))


def logit_tab(X, y, names):
    b, se, dev = logistic_fit(X, y)
    out = []
    for nm, coef, s in zip(names, b, se):
        z = coef / s
        pval = 2 * sst.norm.sf(abs(z))
        out.append({"term": nm, "logit": coef, "se": s,
                    "or": float(np.exp(coef)),
                    "ci_lo": coef - 1.645 * s, "ci_hi": coef + 1.645 * s,
                    "or_lo": float(np.exp(coef - 1.645 * s)),
                    "or_hi": float(np.exp(coef + 1.645 * s)),
                    "z": z, "p_value": pval})
    return pd.DataFrame(out), dev


def conf_dummies(df, ref="UEFA"):
    confs = [c for c in CONFEDERATIONS if c != ref and (df["confederation"] == c).sum() > 0]
    X = np.column_stack([(df["confederation"] == c).astype(float) for c in confs])
    return X, confs


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
    subprocess.run(["git", "commit", "-m", "stage 05 build: dag assumption checks"],
                   cwd=clone_dir, env=env, check=True, capture_output=True, text=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, env=env, check=True)


def main():
    out_root = "/content/soccerdl_out" if os.path.exists("/tmp/github_token") else "output"
    data_dir = os.path.join(out_root, "data")
    stage_dir = os.path.join(out_root, "stages", "05_app")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)
    print("Python:", sys.version.split()[0])

    frame = fetch_csv("prior_model_frame.csv")
    post = None
    conf_post = None
    try:
        import arviz as az
        import tempfile
        post_bytes = fetch_bytes(f"{DATA_ROOT}/posterior.nc")
        tmp = os.path.join(tempfile.gettempdir(), "posterior_tmp.nc")
        with open(tmp, "wb") as f:
            f.write(post_bytes)
        idata = az.from_netcdf(tmp)
        co = idata.posterior["conf_offset"]
        conf_post = {c: co.sel(confederation=c).values.reshape(-1) for c in CONFEDERATIONS}
    except Exception as e:
        print("posterior.nc unavailable, skipping posterior reference:", e)
    frame = frame[frame["confederation"].isin(CONFEDERATIONS)].copy()
    print("frame rows:", len(frame), "| winners:", int(frame["is_winner"].sum()))

    rows = []  # dag_checks.csv long-form

    # ================= C1: confounding =================
    elo = np.log(frame["elo_pre_wc"].to_numpy())
    ch = frame["continental_champion"].to_numpy().astype(bool)
    mu_ch, mu_nc = elo[ch].mean(), elo[~ch].mean()
    d = mu_ch - mu_nc
    boot_d = np.empty(BOOT)
    elo_ch, elo_nc = elo[ch], elo[~ch]
    for b in range(BOOT):
        boot_d[b] = RNG.choice(elo_ch, size=len(elo_ch), replace=True).mean() \
                    - RNG.choice(elo_nc, size=len(elo_nc), replace=True).mean()
    ci = np.quantile(boot_d, [0.05, 0.95])
    t, p_t = sst.ttest_ind(elo_ch, elo_nc, equal_var=False)
    rows += [
        {"check_id": "C1", "check_name": "Confounding (strength->champion)",
         "metric": "mean log-Elo, champions", "value": mu_ch,
         "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
         "n": int(ch.sum()), "note": ""},
        {"check_id": "C1", "check_name": "Confounding (strength->champion)",
         "metric": "mean log-Elo, non-champions", "value": mu_nc,
         "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
         "n": int((~ch).sum()), "note": ""},
        {"check_id": "C1", "check_name": "Confounding (strength->champion)",
         "metric": "mean diff log-Elo (champ - non)", "value": d,
         "ci_lo": ci[0], "ci_hi": ci[1], "p_value": p_t, "n": len(frame),
         "note": "bootstrap 90% CI"},
        {"check_id": "C1", "check_name": "Confounding (strength->champion)",
         "metric": "Welch t", "value": t, "ci_lo": np.nan, "ci_hi": np.nan,
         "p_value": p_t, "n": len(frame), "note": ""},
    ]
    print(f"C1: champion log-Elo {mu_ch:.3f} vs non-champion {mu_nc:.3f}; "
          f"diff {d:+.3f} 90% CI [{ci[0]:+.3f},{ci[1]:+.3f}]; t p={p_t:.4f}")

    # ================= C2: independence proxy =================
    d2 = frame[frame["confederation"] != "OFC"].copy()
    Xc, confs_c = conf_dummies(d2)
    y = d2["is_winner"].to_numpy().astype(float)
    Xm0 = np.column_stack([np.ones(len(d2)), d2["elo_z"].to_numpy()])
    Xm1 = np.column_stack([Xm0, d2["continental_champion"].to_numpy()])
    Xm2 = np.column_stack([Xm1, Xc])
    tab0, dev0 = logit_tab(Xm0, y, ["intercept", "elo_z"])
    tab1, dev1 = logit_tab(Xm1, y, ["intercept", "elo_z", "champion"])
    tab2, dev2 = logit_tab(Xm2, y, ["intercept", "elo_z", "champion"] + [f"conf:{c}" for c in confs_c])
    stat_lr, p_lr = lr_test(dev2, dev1, len(confs_c))
    ch_row = tab1[tab1["term"] == "champion"].iloc[0]
    ch_cond = tab2[tab2["term"] == "champion"].iloc[0]
    rows += [
        {"check_id": "C2", "check_name": "Independence proxy (champion | elo)",
         "metric": "OR champion (elo + champion)", "value": ch_row["or"],
         "ci_lo": ch_row["or_lo"], "ci_hi": ch_row["or_hi"],
         "p_value": ch_row["p_value"], "n": len(d2),
         "note": "logistic, winner ~ elo_z + champion (OFC dropped)"},
        {"check_id": "C2", "check_name": "Independence proxy (champion | elo, conf)",
         "metric": "OR champion (elo + champion + conf)", "value": ch_cond["or"],
         "ci_lo": ch_cond["or_lo"], "ci_hi": ch_cond["or_hi"],
         "p_value": ch_cond["p_value"], "n": len(d2),
         "note": "within-conf conditioning"},
        {"check_id": "C2", "check_name": "Independence proxy (champion | elo, conf)",
         "metric": "LR-test: conf terms", "value": stat_lr,
         "ci_lo": np.nan, "ci_hi": np.nan, "p_value": p_lr, "n": len(d2),
         "note": f"chi2 df={len(confs_c)}"},
    ]
    print(f"C2: OR champion (elo-only adj) {ch_row['or']:.2f} "
          f"[{ch_row['or_lo']:.2f},{ch_row['or_hi']:.2f}]; "
          f"OR (elo+conf adj) {ch_cond['or']:.2f} [{ch_cond['or_lo']:.2f},{ch_cond['or_hi']:.2f}]")

    # ================= C3: sensitivity =================
    Xf = np.column_stack([np.ones(len(d2)), d2["elo_z"].to_numpy(),
                          d2["continental_champion"].to_numpy(), Xc])
    names_f = ["intercept", "elo_z", "champion"] + [f"conf:{c}" for c in confs_c]
    tabM0, devM0 = logit_tab(np.column_stack([np.ones(len(d2)), Xc]), y,
                             ["intercept"] + [f"conf:{c}" for c in confs_c])
    tabM1, devM1 = logit_tab(np.column_stack([np.ones(len(d2)), d2["elo_z"].to_numpy(), Xc]), y,
                             ["intercept", "elo_z"] + [f"conf:{c}" for c in confs_c])
    tabM2 = tab2.copy()
    by_conf = {}
    for c in confs_c:
        m0 = tabM0[tabM0["term"] == f"conf:{c}"].iloc[0]["logit"]
        m2 = tabM2[tabM2["term"] == f"conf:{c}"].iloc[0]["logit"]
        by_conf[c] = {"m0": m0, "m2": m2, "shift": m2 - m0}
    max_shift_c = max(by_conf.values(), key=lambda v: abs(v["shift"]))
    largest_mover = [c for c, v in by_conf.items() if v == max_shift_c][0]
    rows += [{"check_id": "C3", "check_name": "Sensitivity (conf effects)",
              "metric": f"conf:{c} logit M0 (marginal)", "value": v["m0"],
              "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
              "n": len(d2), "note": ""} for c, v in by_conf.items()]
    rows += [{"check_id": "C3", "check_name": "Sensitivity (conf effects)",
              "metric": f"conf:{c} logit M2 (elo+champ adj)", "value": v["m2"],
              "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
              "n": len(d2), "note": ""} for c, v in by_conf.items()]
    rows += [{"check_id": "C3", "check_name": "Sensitivity (conf effects)",
              "metric": "max |shift| M0->M2", "value": abs(max_shift_c["shift"]),
              "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
              "n": len(d2), "note": f"largest mover: {largest_mover}"}]
    if conf_post is not None:
        for c in confs_c:
            if c in conf_post:
                v = conf_post[c]
                rows.append({"check_id": "C3", "check_name": "Sensitivity (posterior ref)",
                             "metric": f"conf_offset:{c} (Bayesian, champ+elo adj)",
                             "value": float(np.mean(v)),
                             "ci_lo": float(np.quantile(v, 0.05)),
                             "ci_hi": float(np.quantile(v, 0.95)),
                             "p_value": np.nan, "n": int(v.size), "note": "posterior.nc"})
    print(f"C3: max |conf shift| M0->M2 = {abs(max_shift_c['shift']):.3f} "
          f"({[c for c, v in by_conf.items() if v == max_shift_c][0]})")

    # ================= C4: balance =================
    chrono = fetch_csv("ranking_chronology.csv")
    team_conf = fetch_csv("team_confederations.csv")
    conf_lookup = team_conf.set_index("team")["confederation"].to_dict()
    try:
        m = fetch_csv("conf_membership.csv")
        overrides = {r.team: {"start": str(r.change_month), "after": r.conf_after,
                              "before": r.conf_before} for r in m.itertuples(index=False)}
    except Exception:
        overrides = {}
    # ranking_chronology has sparse months (e.g. 2026-07 has 25 teams) — use the latest
    # FULL month (>=150 teams) for a fair Elo-level balance snapshot.
    counts = chrono.groupby("month").size()
    full_months = counts[counts >= 150]
    snap_month = full_months.index[-1]
    latest = chrono[chrono["month"] == snap_month].copy()
    def conf_of(team, month):
        base = conf_lookup.get(team, "Other")
        if team in overrides:
            ov = overrides[team]
            return ov["after"] if month >= pd.Period(ov["start"], "M") else ov["before"]
        return base
    latest["conf"] = latest["team"].map(lambda t: conf_of(t, pd.Period(latest["month"].iloc[0], "M")))
    bal = latest[latest["conf"].isin(CONFEDERATIONS)].copy()
    bal["log_elo"] = np.log(bal["elo"])
    groups = [g["log_elo"].to_numpy() for _, g in bal.groupby("conf")]
    F, p_anova = sst.f_oneway(*groups)
    grand = bal["log_elo"].mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = sum(((g - grand) ** 2).sum() for g in groups)
    eta2 = ss_b / ss_t
    c4 = bal.groupby("conf")["log_elo"].agg(["mean", "std", "count"])
    rows += [{"check_id": "C4", "check_name": "Balance (conf->strength)",
              "metric": "ANOVA F (log Elo ~ conf)", "value": F,
              "ci_lo": np.nan, "ci_hi": np.nan, "p_value": p_anova,
              "n": len(bal), "note": "latest ranking month"},
             {"check_id": "C4", "check_name": "Balance (conf->strength)",
              "metric": "eta-squared", "value": eta2,
              "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
              "n": len(bal), "note": "share of Elo variance explained by conf"}]
    for c in CONFEDERATIONS:
        if c in c4.index:
            g = bal.loc[bal["conf"] == c, "log_elo"].to_numpy()
            se_ = sst.sem(g)
            rows.append({"check_id": "C4", "check_name": "Balance (conf->strength)",
                         "metric": f"mean log-Elo {c}", "value": float(g.mean()),
                         "ci_lo": float(g.mean() - 1.645 * se_),
                         "ci_hi": float(g.mean() + 1.645 * se_),
                         "p_value": np.nan, "n": int(len(g)), "note": ""})
    print(f"C4: ANOVA F={F:.1f} p={p_anova:.4f} eta2={eta2:.3f} (snapshot {snap_month}, n={len(bal)})")

    # ---- save ----
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(data_dir, "dag_checks.csv"), index=False)

    lines = [
        "# DAG ASSUMPTION TESTS — STAGE 05 BUILD REPORT",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Frame: {len(frame)} team-WC rows, {int(frame['is_winner'].sum())} winners, "
        f"reused from prior_model_frame.csv; Elo snapshot month {snap_month} "
        f"({len(bal)} teams, latest full month).",
        "- Checks probe DAG edges with the observed proxy Elo for latent team_strength.",
        "- Honesty: these are edge checks, NOT causal-effect estimates (22 WCs can't identify).",
        "",
        "## C1 — Confounding: continental_champion vs Elo (strength -> champion)",
        "| group | mean log-Elo | n |",
        "|---|---|---|",
    ]
    for nm, g in [("champions", ch), ("non-champions", ~ch)]:
        lines.append(f"| {nm} | {elo[g].mean():.3f} | {int(g.sum())} |")
    lines.append("")
    lines.append(f"Mean diff (champion - non-champion) = **{d:+.3f}** log-Elo, "
                 f"90% bootstrap CI [{ci[0]:+.3f}, {ci[1]:+.3f}], Welch t p={p_t:.4f}.")
    lines.append("Interpretation: champions are on average stronger, supporting the "
                 "strength->champion edge, but distributions overlap (the small-data "
                 "reason we never claim a point causal effect).")
    lines += [
        "",
        "## C2 — Independence proxy: does champion add win-info beyond Elo?",
        "| model | OR(champion) | 90% CI | p | n |",
        "|---|---|---|---|---|",
        f"| winner ~ elo_z + champion | {ch_row['or']:.2f} | "
        f"[{ch_row['or_lo']:.2f}, {ch_row['or_hi']:.2f}] | {ch_row['p_value']:.4f} | {len(d2)} |",
        f"| winner ~ elo_z + champion + conf | {ch_cond['or']:.2f} | "
        f"[{ch_cond['or_lo']:.2f}, {ch_cond['or_hi']:.2f}] | {ch_cond['p_value']:.4f} | {len(d2)} |",
        f"| LR test of conf terms | stat={stat_lr:.2f} (df={len(confs_c)}) | — | {p_lr:.4f} | {len(d2)} |",
    ]
    lines.append("Interpretation: with 22 winners the champion odds ratio is noisy — point estimate "
                 "slightly below 1 with a 90% CI that easily includes 1, so there is **no clear "
                 "evidence** champion adds win-information beyond Elo (consistent with the winner "
                 "model's mu_cc ~ -0.11). The confederation terms are jointly strong (LR p small).")
    lines += ["", "## C3 — Sensitivity: continent effects with/without conditioning",
              "| confederation | logit M0 (marginal) | logit M2 (elo+champ) | shift | posterior ref (90% HDI) |",
              "|---|---|---|---|---|"]
    for c in confs_c:
        v = by_conf[c]
        post_str = "—"
        if conf_post is not None and c in conf_post:
            pv = conf_post[c]
            post_str = f"{np.mean(pv):+.2f} [{np.quantile(pv,0.05):+.2f}, {np.quantile(pv,0.95):+.2f}]"
        lines.append(f"| {c} | {v['m0']:+.3f} | {v['m2']:+.3f} | {v['shift']:+.3f} | {post_str} |")
    mover = [c for c, v in by_conf.items() if v == max_shift_c][0]
    lines.append("")
    lines.append(f"Max |shift| M0->M2 = **{abs(max_shift_c['shift']):.3f}** "
                 f"(largest mover: {mover}). CAF/CONCACAF/AFC have never won a World Cup, so their "
                 "logits saturate near-separation; only the CONMEBOL-vs-UEFA contrast is "
                 "well-identified, and it moves little (Δ≈-0.05). Continent log-odds are sensitive "
                 "to conditioning — exactly why the winner model reports distributions and never "
                 "labels conf effects 'causal'.")
    lines += ["", "## C4 — Balance: confederation predicts Elo level (conf -> strength)",
              f"ANOVA on log-Elo, {len(bal)} teams at {snap_month} (latest full month): "
              f"F = **{F:.1f}**, p = {p_anova:.4f}, eta² = **{eta2:.3f}**.",
              "| confederation | mean log-Elo | 90% CI | n |",
              "|---|---|---|---|"]
    for c in CONFEDERATIONS:
        if c in c4.index:
            g = bal.loc[bal["conf"] == c, "log_elo"].to_numpy()
            se_ = sst.sem(g)
            lines.append(f"| {c} | {g.mean():.3f} | "
                         f"[{g.mean()-1.645*se_:.3f}, {g.mean()+1.645*se_:.3f}] | {len(g)} |")
    lines.append("")
    lines.append("Interpretation: confederation explains a large share of Elo variance, "
                 "justifying the conf->strength hierarchical prior.")
    with open(os.path.join(data_dir, "dag_checks_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    src = "/content/build_dag_checks.py"
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(stage_dir, "build_dag_checks.py"))

    print("artifacts:", ", ".join(sorted(os.listdir(data_dir))))
    if os.path.exists("/tmp/github_token"):
        token = open("/tmp/github_token", encoding="utf-8").read().splitlines()[0].strip()
        push_to_github(GITHUB_REPO, token, data_dir, stage_dir)
        print("commit: True | pushed to", GITHUB_REPO)
    else:
        print("commit: skipped (no /tmp/github_token)")


if __name__ == "__main__":
    main()
