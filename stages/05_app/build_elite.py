# -*- coding: utf-8 -*-
# =====================================================================
# STAGE 05 BUILD 2 — ELITE COMPOSITION (Ranking Dynamics tab)
# Metric A: share of the top-N slice per confederation, monthly 1992-01..2026-08
#   for thresholds N in {5,10,20}, from the monthly Elo ranking.
# Model: simple exponential smoothing (SES) per continent x threshold, alpha fitted
#   by grid search (min SSE) on the observed proportions; +12-month forecast anchored
#   at the last observed level; "which confederation line is ahead over time" summary
#   (running leader, % window led, final ranking).
#
# Outputs (pushed to GitHub): elite_composition.csv, elite_fit.csv, elite_summary.csv.
# =====================================================================

import base64
import datetime as dt
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd

GITHUB_REPO = "batestguy/soccer-eurovsSA"
DATA_ROOT = "https://raw.githubusercontent.com/batestguy/soccer-eurovsSA/main/data"
CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]
THRESHOLDS = [5, 10, 20]
WINDOW = ("1992-01", "2026-07")  # 2026-07 = latest observed ranking month in ranking_chronology.csv
FORECAST_MONTHS = 12

# Membership overrides (same source of truth as build_strength.py / conf_membership.csv).
MEMBERSHIP = {
    "Australia":  {"start": "2006-01", "after": "AFC", "before": "OFC"},
    "Israel":     {"start": "1994-07", "after": "UEFA", "before": "Other"},
    "Kazakhstan": {"start": "2002-01", "after": "UEFA", "before": "AFC"},
}


def fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "soccerdl-elite"})
    return urllib.request.urlopen(req, timeout=180).read()


def fetch_csv(name):
    return pd.read_csv(io.BytesIO(fetch_bytes(f"{DATA_ROOT}/{name}")))


def membership_lookup():
    """Try the pushed conf_membership.csv (source of truth); fall back to embedded dict."""
    try:
        m = fetch_csv("conf_membership.csv")
        d = {}
        for r in m.itertuples(index=False):
            d[r.team] = {"start": str(r.change_month), "after": r.conf_after, "before": r.conf_before}
        return d
    except Exception as e:  # pragma: no cover - network fallback
        print("conf_membership.csv fetch failed, using embedded:", e)
        return dict(MEMBERSHIP)


def assign_confederation(team, month, base_lookup, overrides):
    base = base_lookup.get(team, "Other")
    if team in overrides:
        ov = overrides[team]
        return ov["after"] if month >= pd.Period(ov["start"], "M") else ov["before"]
    return base


def build_monthly_share(chronology, team_conf, overrides):
    """Monthly Metric-A share (0..1) per confederation per threshold."""
    conf_lookup = team_conf.set_index("team")["confederation"].to_dict()
    months = pd.period_range(WINDOW[0], WINDOW[1], freq="M")

    comp = []  # rows for elite_composition.csv
    prev_rank = None
    for month in months:
        sub = chronology[chronology["month"] == str(month)]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["conf"] = sub["team"].map(
            lambda t: assign_confederation(t, month, conf_lookup, overrides))
        sub = sub[sub["conf"].isin(CONFEDERATIONS)].sort_values("elo", ascending=False)
        for th in THRESHOLDS:
            top = sub.head(th)
            for c in CONFEDERATIONS:
                share = float((top["conf"] == c).mean())
                comp.append({"month": str(month), "threshold": th,
                             "confederation": c, "share": share,
                             "share_pct": share * 100.0})
    return pd.DataFrame(comp)


def ses_forecast(y, alpha):
    """One-step-ahead SES recursion; returns fitted levels (last = level forecast)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    level = np.empty(n)
    level[0] = y[0]
    for t in range(1, n):
        level[t] = alpha * y[t] + (1.0 - alpha) * level[t - 1]
    return level


def fit_alpha(y, horizon=FORECAST_MONTHS, warmup=36):
    """Fit alpha by rolling-origin SSE at the FORECAST horizon (flat level forecast).
    1-step SSE collapses to alpha~1 for persistent series; the app forecasts h months
    ahead, so we optimise the same horizon. numpy only, no scipy dependency."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    best_a, best_sse = 0.5, np.inf
    for a in np.linspace(0.01, 0.99, 99):
        lvl = ses_forecast(y, a)
        errs = []
        for t0 in range(warmup, n - horizon):
            errs.append((lvl[t0] - y[t0 + horizon]) ** 2)
        sse = float(np.sum(errs)) if errs else np.inf
        if sse < best_sse:
            best_a, best_sse = float(a), sse
    return best_a, best_sse


def running_leader_stats(comp, fit_by):
    """Per threshold: months-led (observed months only), % window led, final share/rank,
    forecast share."""
    rows = []
    for th in THRESHOLDS:
        d = comp[(comp["threshold"] == th)].pivot(index="month", columns="confederation",
                                                  values="share").sort_index()
        n_months = len(d.index)
        led = {}
        for m in d.index:
            mx = d.loc[m].max()
            for c in d.columns:
                if d.loc[m, c] >= mx - 1e-9:
                    led[c] = led.get(c, 0) + 1
        final = d.iloc[-1]
        fcast = fit_by[(th, "fcast_share")]
        for c in CONFEDERATIONS:
            rows.append({
                "threshold": th, "confederation": c,
                "alpha": fit_by[(th, c, "alpha")],
                "months_led": led.get(c, 0),
                "pct_window_led": 100.0 * led.get(c, 0) / n_months,
                "final_share": float(final.get(c, np.nan)),
                "final_rank": int(np.searchsorted(-final.sort_values(ascending=False).to_numpy(),
                                                  -float(final.get(c, np.nan))) + 1),
                "forecast_share": float(fcast[c].iloc[0]),
            })
    return pd.DataFrame(rows)


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
    subprocess.run(["git", "commit", "-m", "stage 05 build: elite composition"],
                   cwd=clone_dir, env=env, check=True, capture_output=True, text=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, env=env, check=True)


def main():
    out_root = "/content/soccerdl_out" if os.path.exists("/tmp/github_token") else "output"
    data_dir = os.path.join(out_root, "data")
    stage_dir = os.path.join(out_root, "stages", "05_app")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    print("Python:", sys.version.split()[0])

    chronology = fetch_csv("ranking_chronology.csv")
    team_conf = fetch_csv("team_confederations.csv")
    overrides = membership_lookup()
    print("chronology rows:", len(chronology), "teams:", chronology["team"].nunique())

    comp = build_monthly_share(chronology, team_conf, overrides)
    comp = comp.sort_values(["threshold", "month", "confederation"]).reset_index(drop=True)
    comp.to_csv(os.path.join(data_dir, "elite_composition.csv"), index=False)
    print("composition rows:", len(comp),
          "| months:", comp["month"].nunique(),
          "| per-threshold monthly-sum checks:",
          round(comp.groupby(["threshold", "month"])["share"].sum().mean(), 4))

    # ---- SES fit + 12-month forecast per (threshold, confederation) -------
    months = pd.period_range(WINDOW[0], WINDOW[1], freq="M")
    fit_rows = []
    fit_by = {}
    for th in THRESHOLDS:
        d = comp[comp["threshold"] == th].pivot(index="month", columns="confederation",
                                                values="share")
        d = d.reindex([str(m) for m in months]).ffill()
        fcast_months = pd.period_range(pd.Period(WINDOW[1], "M") + 1,
                                       periods=FORECAST_MONTHS, freq="M")
        fcast_index = [str(m) for m in fcast_months]
        fcast = pd.DataFrame(0.0, index=fcast_index, columns=CONFEDERATIONS)
        for c in CONFEDERATIONS:
            y = d[c].fillna(0.0).to_numpy()
            alpha, sse = fit_alpha(y)
            lvl = ses_forecast(y, alpha)
            fc = float(lvl[-1])  # anchored flat forecast
            fcast[c] = fc
            fit_by[(th, c, "alpha")] = alpha
            for j, m in enumerate(months):
                fit_rows.append({"month": str(m), "threshold": th, "confederation": c,
                                 "fitted": float(lvl[j]), "fitted_pct": float(lvl[j]) * 100.0,
                                 "is_forecast": False})
            for m in fcast_index:
                fit_rows.append({"month": m, "threshold": th, "confederation": c,
                                 "fitted": fc, "fitted_pct": fc * 100.0,
                                 "is_forecast": True})
        fit_by[(th, "fcast_share")] = fcast
        print(f"top {th}: alpha range "
              f"[{min(fit_by[(th, c, 'alpha')] for c in CONFEDERATIONS):.3f}, "
              f"{max(fit_by[(th, c, 'alpha')] for c in CONFEDERATIONS):.3f}]")

    fit = pd.DataFrame(fit_rows)
    fit.to_csv(os.path.join(data_dir, "elite_fit.csv"), index=False)
    print("fit rows:", len(fit), "| forecast months:", fit[fit["is_forecast"]]["month"].nunique())

    # ---- summary: which line is ahead over time ---------------------------
    summary = running_leader_stats(comp, fit_by)
    summary.to_csv(os.path.join(data_dir, "elite_summary.csv"), index=False)

    # ---- report ------------------------------------------------------------
    lines = [
        "# ELITE COMPOSITION — STAGE 05 BUILD REPORT",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Metric A: share of the top-N monthly Elo ranking per confederation "
        f"(thresholds {THRESHOLDS}), window {WINDOW[0]}..{WINDOW[1]}.",
        f"- Model: SES per confederation x threshold (alpha fitted by rolling-origin "
        f"+{FORECAST_MONTHS}-month forecast SSE); forecast anchored at the fitted SES level.",
        "- Fit uses observed months only; gaps (Covid 2020 etc.) carried forward for continuity.",
        "- 2026-07 is the latest observed ranking month in ranking_chronology.csv.",
        "- Membership overrides applied (Australia, Israel, Kazakhstan).",
        "",
        "## Running leader (% of window led, final share/rank)",
        "| Threshold | Confederation | alpha | % window led | final share | final rank | forecast (+12m) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary.sort_values(["threshold", "final_rank"]).itertuples(index=False):
        lines.append(f"| {r.threshold} | {r.confederation} | {r.alpha:.3f} | "
                     f"{r.pct_window_led:.1f} | {r.final_share:.2f} | {r.final_rank} | "
                     f"{r.forecast_share:.2f} |")
    with open(os.path.join(data_dir, "elite_summary_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # provenance
    src = "/content/build_elite.py"
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(stage_dir, "build_elite.py"))

    print("artifacts:", ", ".join(sorted(os.listdir(data_dir))))
    top = summary[summary["threshold"] == 5].sort_values("final_rank")
    print("top-5 leader(s):", ", ".join(top[top["final_rank"] == 1]["confederation"]),
          "final share", round(top[top["final_rank"] == 1]["final_share"].iloc[0], 2))
    if os.path.exists("/tmp/github_token"):
        token = open("/tmp/github_token", encoding="utf-8").read().splitlines()[0].strip()
        push_to_github(GITHUB_REPO, token, data_dir, stage_dir)
        print("commit: True | pushed to", GITHUB_REPO)
    else:
        print("commit: skipped (no /tmp/github_token)")


if __name__ == "__main__":
    main()
