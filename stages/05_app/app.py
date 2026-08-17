# -*- coding: utf-8 -*-
# =====================================================================
# STAGE 05 — GRADIO APP  (Bayesian World Cup Prediction)  — 8 tabs
#
#  1 Continental Strength   hierarchical panel on log(monthly Elo):
#                           A/B spline trends + bands + difference panel
#  2 Forest Plot            posterior micro-evidence (winner model)
#  3 Monte Carlo            2026 future cast (50k sims)
#  4 do()-What-If           counterfactual champion on/off
#  5 Prior Predictive       what the model believed before data
#  6 Causal: Continent -> Winner  DAG merged + posterior continent effects
#  7 DAG Assumption Tests   4 edge checks (selector)
#  8 Ranking Dynamics       elite composition (top 5/10/20, SES + forecast)
#
# Inference is DECOUPLED from serving: the app only loads pre-trained
# artifacts (.nc / CSV) — it never runs MCMC at runtime.
#
# Run:  python app.py            -> launches the Gradio UI on :7860
#       python app.py --render   -> writes tab figures to out/ (gate check)
# Artifacts are read from <script_dir>/data or $SOCCERDL_DATA.
# =====================================================================

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az

CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "OFC", "AFC"]
PERIODS = ["Monthly", "Quarterly", "Annual"]
THRESHOLDS = [5, 10, 20]
FONT = 10
C_PALETTE = {"UEFA": "#1f77b4", "CONMEBOL": "#2ca02c", "CAF": "#ff7f0e",
             "CONCACAF": "#d62728", "OFC": "#9467bd", "AFC": "#17becf"}

APP_CSS = """
:root {
  --pitch-ink: #153b2e;
  --programme-paper: #f5f0e6;
  --programme-gold: #c79a3b;
}
.gradio-container {
  background:
    linear-gradient(rgba(21, 59, 46, .035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(21, 59, 46, .035) 1px, transparent 1px),
    var(--programme-paper) !important;
  background-size: 28px 28px;
}
.hero-card {
  border-top: 5px solid var(--pitch-ink);
  border-bottom: 1px solid rgba(21, 59, 46, .25);
  padding: 1.1rem 1.25rem .9rem;
  margin-bottom: .8rem;
  background: rgba(255,255,255,.72);
  box-shadow: 0 10px 28px rgba(21,59,46,.08);
}
.hero-kicker {
  color: #8a6725;
  font-size: .76rem;
  font-weight: 750;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.hero-card h1 {
  color: var(--pitch-ink);
  font-family: Georgia, 'Times New Roman', serif;
  font-size: clamp(1.8rem, 4vw, 3.2rem);
  line-height: 1;
  margin: .35rem 0 .65rem;
}
.method-note {
  border-left: 4px solid var(--programme-gold);
  background: rgba(255,255,255,.62);
  padding: .7rem .9rem;
  color: #38443f;
}
.quiet-signature { text-align: right; opacity: .58; font-size: .72rem; margin-top: 1rem; }
.quiet-signature a { color: var(--pitch-ink) !important; text-decoration: none; }
.tab-nav button { font-weight: 650; }
"""

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_CANDIDATES = [
    os.path.join(_APP_DIR, "data"),
    os.path.abspath(os.path.join(_APP_DIR, "..", "..", "data")),
    os.environ.get("SOCCERDL_DATA", "data"),
]
DATA_DIR = next((p for p in _DATA_CANDIDATES if os.path.isdir(p)), "data")


# ----------------------------------------------------------------------
# loaders (lazy)
# ----------------------------------------------------------------------
_cache = {}


def load_csv(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(os.path.join(DATA_DIR, name))
    return _cache[name]


def load_nc(name):
    if name not in _cache:
        _cache[name] = az.from_netcdf(os.path.join(DATA_DIR, name))
    return _cache[name]


def load_text(name):
    if name not in _cache:
        with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
            _cache[name] = f.read()
    return _cache[name]


def _month_num(months):
    m = pd.PeriodIndex(months, freq="M")
    return m.year + (m.month - 1) / 12.0


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _hdi(array, prob=0.90):
    lo = (1 - prob) / 2
    return np.quantile(array, lo), np.quantile(array, 1 - lo)


def _resample(df, period, value_cols):
    """Average monthly series into quarterly/annual (or keep monthly)."""
    if period not in PERIODS:
        raise ValueError(f"Unknown period aggregation: {period}")
    if period == "Monthly":
        return df.copy()
    t = df.copy()
    m = pd.PeriodIndex(t["month"], freq="M")
    t["t"] = m.year + (m.month - 1) / 12.0
    t["_k"] = m.asfreq("Q").astype(str) if period == "Quarterly" else m.year.astype(str)
    # Keeping 'month' here silently prevents quarterly/annual aggregation.
    # Retain only true series identifiers (for example confederation or pair).
    groups = [c for c in t.columns
              if c not in value_cols and c not in ("month", "_k", "t")]
    agg = {c: "mean" for c in value_cols}
    agg["t"] = "mean"
    return t.groupby(groups + ["_k"], as_index=False, sort=True).agg(agg)


def _signed_pair(a, b):
    pairs = set(load_csv("strength_pairwise.csv")["pair"])
    if f"{a} vs {b}" in pairs:
        return f"{a} vs {b}", 1
    return f"{b} vs {a}", -1


def _team_conf_series():
    """Cached team-month log-Elo with historical confederation membership."""
    key = "_team_conf_series"
    if key not in _cache:
        chrono = load_csv("ranking_chronology.csv")
        team_conf = load_csv("team_confederations.csv")
        conf_lookup = team_conf.set_index("team")["confederation"].to_dict()
        try:
            mem = load_csv("conf_membership.csv")
            overrides = {r.team: {"start": str(r.change_month), "after": r.conf_after,
                                  "before": r.conf_before} for r in mem.itertuples(index=False)}
        except Exception:
            overrides = {}

        def conf_of(team, month):
            base = conf_lookup.get(team, "Other")
            if team in overrides:
                ov = overrides[team]
                return ov["after"] if month >= pd.Period(ov["start"], "M") else ov["before"]
            return base

        df = chrono.copy()
        df["log_elo"] = np.log(df["elo"])
        df["t"] = _month_num(df["month"])
        df["confederation"] = [conf_of(t, pd.Period(m, "M"))
                               for t, m in zip(df["team"], df["month"])]
        _cache[key] = df[df["confederation"].isin(CONFEDERATIONS)]
    return _cache[key]


# ----------------------------------------------------------------------
# TAB 1 — Continental Strength
# ----------------------------------------------------------------------
def tab_strength(conf_a, conf_b, period):
    if conf_a == conf_b:
        conf_b = next(c for c in CONFEDERATIONS if c != conf_a)

    trends = load_csv("strength_trends.csv")
    diff = load_csv("strength_diff_curves.csv")
    pair_rows = load_csv("strength_pairwise.csv")

    vcols = ["log_mean", "log_p5", "log_p95", "elo_mean", "elo_p5", "elo_p95"]
    ta = _resample(trends[trends["confederation"] == conf_a], period, vcols)
    tb = _resample(trends[trends["confederation"] == conf_b], period, vcols)
    if period == "Monthly":
        ta["t"], tb["t"] = _month_num(ta["month"]), _month_num(tb["month"])

    base, sign = _signed_pair(conf_a, conf_b)
    d = diff[diff["pair"] == base].copy()
    if sign < 0:
        d["log_mean"] = -d["log_mean"]; d["log_p5"] = -d["log_p95"]; d["log_p95"] = -d["log_p5"]
        d["elo_mean"] = -d["elo_mean"]; d["elo_p5"] = -d["elo_p95"]; d["elo_p95"] = -d["elo_p5"]
    d = _resample(d, period, vcols)
    if period == "Monthly":
        d["t"] = _month_num(d["month"])

    pr = pair_rows[pair_rows["pair"] == base].iloc[0]
    p_ab = float(pr["p_a_stronger"]) if sign > 0 else 1 - float(pr["p_a_stronger"])
    dlog = float(pr["avg_log_diff_mean"]) if sign > 0 else -float(pr["avg_log_diff_mean"])
    lo = float(pr["avg_log_diff_p5"]) if sign > 0 else -float(pr["avg_log_diff_p95"])
    hi = float(pr["avg_log_diff_p95"]) if sign > 0 else -float(pr["avg_log_diff_p5"])
    delo = float(pr["avg_elo_diff_mean"]) if sign > 0 else -float(pr["avg_elo_diff_mean"])
    elo_lo = float(pr["avg_elo_diff_p5"]) if sign > 0 else -float(pr["avg_elo_diff_p95"])
    elo_hi = float(pr["avg_elo_diff_p95"]) if sign > 0 else -float(pr["avg_elo_diff_p5"])
    n_zero = int(((d["log_p5"] > 0) | (d["log_p95"] < 0)).sum())
    n_zero_elo = int(((d["elo_p5"] > 0) | (d["elo_p95"] < 0)).sum())
    n_tot = len(d)
    latest = d.sort_values("t").iloc[-1]

    team_ser = _team_conf_series()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), dpi=140,
                                   gridspec_kw={"height_ratios": [1.6, 1.0]})

    for _, g in team_ser.groupby("team"):
        ax1.plot(g["t"], g["log_elo"], lw=0.35, color="#d9d9d9", alpha=0.55, zorder=1)
    for conf, color in ((conf_a, C_PALETTE[conf_a]), (conf_b, C_PALETTE[conf_b])):
        sub = team_ser[team_ser["confederation"] == conf]
        for _, g in sub.groupby("team"):
            ax1.plot(g["t"], g["log_elo"], lw=0.6, color=color, alpha=0.75, zorder=2)

    ax1.fill_between(ta["t"], ta["log_p5"], ta["log_p95"], color="#74A9CF", alpha=0.30, zorder=3)
    ax1.fill_between(tb["t"], tb["log_p5"], tb["log_p95"], color="#FDBE85", alpha=0.30, zorder=3)
    ax1.plot(ta["t"], ta["log_mean"], color="#045A8D", lw=2.2, label=conf_a, zorder=4)
    ax1.plot(tb["t"], tb["log_mean"], color="#E6550D", lw=2.2, label=conf_b, zorder=4)
    ax1.set_title("Continental strength — spline trend on log(monthly Elo), 90% bands\n"
                  "faint = every team's series; solid = fitted continental trend", fontsize=11)
    ax1.set_ylabel("log(Elo)")
    ax1.legend(loc="upper left", frameon=False, fontsize=FONT)
    ax1.grid(alpha=0.2)

    ax2.plot(d["t"], d["log_mean"], color="#1B7837", lw=2.0, zorder=4)
    ax2.fill_between(d["t"], d["log_p5"], d["log_p95"], color="#A6DBA0", alpha=0.45, zorder=3)
    ax2.axhline(0, color="#555555", lw=1)
    if n_zero:
        ax2.fill_between(d["t"], 0, np.where((d["log_p5"] > 0) | (d["log_p95"] < 0),
                                             d["log_mean"], np.nan),
                         color="#1B7837", alpha=0.18)
    ax2.set_title(f"Dynamic difference: {conf_a} - {conf_b}  ({period.lower()}, 90% band)",
                  fontsize=11)
    ax2.set_xlabel("time")
    ax2.set_ylabel("delta log(Elo)")
    ax2.grid(alpha=0.2)
    ax2.text(
        0.012, 0.97,
        f"DYNAMIC GAP\n"
        f"band excludes zero: {n_zero}/{n_tot} periods (log) · "
        f"{n_zero_elo}/{n_tot} periods (Elo pts)\n"
        f"latest: {latest['log_mean']:+.3f} log · {latest['elo_mean']:+.0f} Elo pts",
        transform=ax2.transAxes, va="top", ha="left", fontsize=8.6, color="#153b2e",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#a6b9ae",
              "alpha": 0.92},
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.text(
        0.012, 0.012,
        f"OVERALL 1992–2026 AVERAGE  |  {conf_a}-{conf_b}: "
        f"{dlog:+.3f} log [{lo:+.3f}, {hi:+.3f}]  ·  "
        f"{delo:+.0f} Elo pts [{elo_lo:+.0f}, {elo_hi:+.0f}]  ·  "
        f"P({conf_a}>{conf_b})={p_ab:.2f}\n"
        "90% HDI · structural/associational comparison, not a causal effect",
        fontsize=8.6, color="#333333",
    )

    pair_table = pair_rows[["pair", "avg_log_diff_mean", "avg_log_diff_p5",
                            "avg_log_diff_p95", "avg_elo_diff_mean", "p_a_stronger"]].copy()
    pair_table.columns = ["pair", "avg log-diff", "90% lo", "90% hi",
                          "avg Elo-pts diff", "P(A>B)"]
    return fig, pair_table.sort_values("avg log-diff", ascending=False)


# ----------------------------------------------------------------------
# TAB 2 — Forest Plot (shared with Tab 6)
# ----------------------------------------------------------------------
def _forest_fig(metric):
    idata = load_nc("posterior.nc")
    post = idata.posterior
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    if metric == "beta_elo":
        values = post["beta_elo"].values.reshape(-1)
        ax.hist(values, bins=40, color="#2C7FB8", alpha=0.8)
        lo, hi = _hdi(values)
        ax.axvline(lo, color="#D95F02", ls="--", lw=1.2)
        ax.axvline(hi, color="#D95F02", ls="--", lw=1.2)
        ax.axvline(0, color="#555555", lw=1)
        ax.set_title(f"Posterior - {metric} (90% HDI [{lo:.2f}, {hi:.2f}])", fontsize=12)
        ax.set_xlabel("value"); ax.set_ylabel("density")
        ax.grid(alpha=0.2)
    else:
        var = "cc_effect" if metric.startswith("continental") else "conf_offset"
        values = post[var].values
        stacked = values.reshape(-1, len(CONFEDERATIONS))
        means = stacked.mean(axis=0)
        order = np.argsort(means)
        y = np.arange(len(order))
        lo = np.quantile(stacked, 0.05, axis=0)[order]
        hi = np.quantile(stacked, 0.95, axis=0)[order]
        crosses = (lo <= 0) & (hi >= 0)
        ax.axvline(0, color="#555555", lw=1)
        for i in range(len(order)):
            color = "#8C6D1F" if crosses[i] else "#1B7837"
            ax.errorbar([means[order[i]]], [y[i]],
                        xerr=[[means[order[i]] - lo[i]], [hi[i] - means[order[i]]]],
                        fmt="o", color=color, ecolor=color, capsize=3, ms=6)
        ax.set_yticks(y)
        ax.set_yticklabels([CONFEDERATIONS[i] for i in order])
        ax.set_xlabel("posterior value (log-odds scale)")
        ax.set_title(f"Posterior - {metric} per confederation (90% HDI)", fontsize=12)
        ax.text(0.99, 0.02,
                "orange = 90% HDI crosses zero (no reliable effect)",
                transform=ax.transAxes, ha="right", fontsize=8.5, color="#555555")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def tab_forest(metric):
    return _forest_fig(metric)


# ----------------------------------------------------------------------
# TAB 3 — Monte Carlo
# ----------------------------------------------------------------------
def tab_monte_carlo(region, top_n):
    res = load_csv("monte_carlo_results.csv")
    if region != "All":
        res = res[res["confederation"] == region]
    res = res.sort_values("p_win_mean", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(res))), dpi=140)
    y = np.arange(len(res))
    ax.errorbar(res["p_win_mean"] * 100, y,
                xerr=[(res["p_win_mean"] - res["p_win_p5"]) * 100,
                      (res["p_win_p95"] - res["p_win_mean"]) * 100],
                fmt="o", color="#D95F02", ecolor="#7F2704", capsize=3, ms=7)
    ax.scatter(res["win_freq"] * 100, y, marker="x", s=40, color="#045A8D",
               label="empirical winner freq (50k sims)")
    ax.axvline(0, color="#cccccc", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(res["team"])
    ax.set_xlabel("Probability of winning the 2026 World Cup (%)")
    ax.set_title(f"Monte Carlo Oracle - 2026 replay ({region}), 90% intervals", fontsize=12)
    ax.legend(frameon=False, fontsize=FONT)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# TAB 4 — do()-What-If
# ----------------------------------------------------------------------
def tab_do(region):
    contr = load_csv("do_contrast.csv")
    if region != "All":
        contr = contr[contr["confederation"] == region]
    contr = contr.sort_values("diff_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(contr))), dpi=140)
    y = np.arange(len(contr))
    ax.errorbar(contr["diff_mean"] * 100, y,
                xerr=[(contr["diff_mean"] - contr["diff_p5"]) * 100,
                      (contr["diff_p95"] - contr["diff_mean"]) * 100],
                fmt="o", color="#1B7837", ecolor="#276419", capsize=3, ms=6)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(contr["team"])
    ax.set_xlabel("delta P(win) = P(do(champion=1)) - P(do(champion=0))  (pp)")
    ax.set_title("do()-What-If - continental champion on vs off (counterfactual simulation)",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# TAB 5 — Prior Predictive
# ----------------------------------------------------------------------
def tab_prior_predictive(mode):
    frame = load_csv("prior_model_frame.csv")
    f2022 = frame[frame["wc_year"] == 2022].sort_values("winner_position").reset_index(drop=True)
    teams = f2022["team"].tolist()
    if mode == "Prior (before data)":
        p = load_nc("prior.nc").prior["p_2022"].stack(sample=("chain", "draw")).values
        title = "Prior predictive - what the model believed before data"
    else:
        p = load_nc("posterior.nc").posterior["p_2022"].stack(sample=("chain", "draw")).values
        title = "Posterior predictive - after 22 World Cups"
    means = p.mean(axis=1)
    order = np.argsort(means)[::-1][:12]
    order = order[::-1]
    lo = np.quantile(p, 0.05, axis=1)[order]
    hi = np.quantile(p, 0.95, axis=1)[order]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    y = np.arange(len(order))
    ax.errorbar(means[order] * 100, y, xerr=[(means[order] - lo) * 100, (hi - means[order]) * 100],
                fmt="o", color="#2C7FB8", ecolor="#084594", capsize=3, ms=7)
    ax.set_yticks(y)
    ax.set_yticklabels([teams[i] for i in order])
    ax.set_xlabel("Probability of winning 2022 (model output, %)")
    ax.set_title(f"{title}\n90% intervals - distributions, never points", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# TAB 6 — Causal: Continent -> Winner
# ----------------------------------------------------------------------
def tab_causal(metric):
    return _forest_fig(metric)


# ----------------------------------------------------------------------
# TAB 7 — DAG Assumption Tests
# ----------------------------------------------------------------------
def _extract_check_section(check_id):
    report = load_text("dag_checks_report.md")
    m = re.search(rf"^## {check_id}[^\n]*\n(.*?)(?=^## |\Z)", report, re.S | re.M)
    return m.group(1).strip() if m else "(section not found)"


def tab_dag_check(check_id):
    df = load_csv("dag_checks.csv")
    sub = df[df["check_id"] == check_id]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    if check_id == "C1":
        m = sub[sub["metric"].str.startswith("mean log-Elo,")]
        labels = [("champions" if "champions" in x else "non-champions") for x in m["metric"]]
        vals = m["value"].to_numpy()
        ax.bar(labels, vals, color=["#1B7837", "#B3B3B3"], alpha=0.85)
        ax.set_ylabel("mean log-Elo")
        ax.set_title("C1 - confounding: champions vs non-champions Elo", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
    elif check_id == "C2":
        m = sub[sub["metric"] == "OR champion (elo + champion)"]
        m2 = sub[sub["metric"] == "OR champion (elo + champion + conf)"]
        vals = [m["value"].iloc[0], m2["value"].iloc[0]]
        err = [[vals[0] - m["ci_lo"].iloc[0], vals[1] - m2["ci_lo"].iloc[0]],
               [m["ci_hi"].iloc[0] - vals[0], m2["ci_hi"].iloc[0] - vals[1]]]
        ax.errorbar([0, 1], vals, yerr=err, fmt="o", color="#1B7837",
                    ecolor="#276419", capsize=4, ms=7)
        ax.axhline(1, color="#555555", ls="--", lw=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["+ Elo", "+ Elo + conf"])
        ax.set_ylabel("odds ratio (champion)")
        ax.set_title("C2 - independence proxy: OR of champion on P(win)", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
    elif check_id == "C3":
        m0 = sub[sub["metric"].str.endswith("M0 (marginal)")]
        m2 = sub[sub["metric"].str.endswith("M2 (elo+champ adj)")]
        confs = [x.replace("conf:", "").replace(" logit M0 (marginal)", "") for x in m0["metric"]]
        x = np.arange(len(confs)); w = 0.35
        ax.bar(x - w / 2, m0["value"].to_numpy(), w, label="M0 marginal", color="#B3B3B3")
        ax.bar(x + w / 2, m2["value"].to_numpy(), w, label="M2 elo+champ", color="#1B7837")
        ax.set_xticks(x); ax.set_xticklabels(confs)
        ax.set_ylabel("confederation logit")
        ax.set_title("C3 - sensitivity: continent effects with/without conditioning", fontsize=12)
        ax.legend(frameon=False, fontsize=FONT)
        ax.grid(axis="y", alpha=0.25)
    elif check_id == "C4":
        m = sub[sub["metric"].str.startswith("mean log-Elo ")]
        confs = [x.replace("mean log-Elo ", "") for x in m["metric"]]
        vals = m["value"].to_numpy()
        y = np.arange(len(confs))
        lo = m["ci_lo"].to_numpy(); hi = m["ci_hi"].to_numpy()
        ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="o", color="#1B7837",
                    ecolor="#276419", capsize=3, ms=6)
        ax.set_yticks(y); ax.set_yticklabels(confs)
        ax.set_xlabel("mean log-Elo (90% CI)")
        ax.set_title("C4 - balance: confederation predicts Elo level", fontsize=12)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    table = sub[["check_id", "metric", "value", "ci_lo", "ci_hi", "p_value", "n"]]
    table = table.rename(columns={"ci_lo": "CI lo", "ci_hi": "CI hi", "p_value": "p",
                                  "value": "value"})
    return fig, _extract_check_section(check_id), table


# ----------------------------------------------------------------------
# TAB 8 — Ranking Dynamics (elite composition)
# ----------------------------------------------------------------------
def tab_ranking(threshold):
    comp = load_csv("elite_composition.csv")
    fit = load_csv("elite_fit.csv")
    summ = load_csv("elite_summary.csv")
    c = comp[comp["threshold"] == threshold].reset_index(drop=True)
    f = fit[fit["threshold"] == threshold].reset_index(drop=True)
    s = summ[summ["threshold"] == threshold]

    cm = _month_num(c["month"])
    fm = _month_num(f["month"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=140, sharex=True,
                                   gridspec_kw={"height_ratios": [2.0, 1.0]})
    for conf in CONFEDERATIONS:
        color = C_PALETTE[conf]
        cc = c[c["confederation"] == conf]
        ff = f[f["confederation"] == conf]
        fc = f[(f["confederation"] == conf) & (f["is_forecast"])]
        ax1.plot(cm[cc.index], cc["share_pct"], "o", ms=2.5, color=color, alpha=0.35, zorder=2)
        ax1.plot(fm[ff.index], ff["fitted_pct"], color=color, lw=1.8, label=conf, zorder=3)
        if len(fc):
            ax1.plot(fm[fc.index], fc["fitted_pct"], color=color, lw=1.6, ls="--", zorder=3)
    ax1.set_ylabel("share of top-%d (%%)" % threshold)
    ax1.set_title(f"Elite composition - descriptive SES fitted share of the top {threshold} "
                  "(dashed = deterministic +12m forecast)", fontsize=11)
    ax1.legend(ncol=3, frameon=False, fontsize=FONT, loc="upper left")
    ax1.grid(alpha=0.2)
    last_obs = f[f["is_forecast"] == False]["month"].max()
    if pd.notna(last_obs):
        ax1.axvline(_month_num([last_obs])[0], color="#888888", ls=":", lw=1)

    piv = c.pivot(index="month", columns="confederation", values="share_pct").reindex(
        sorted(c["month"].unique()))
    rank = piv.rank(axis=1, method="min", ascending=False)
    for conf in CONFEDERATIONS:
        ax2.plot(_month_num(rank.index), rank[conf], color=C_PALETTE[conf], lw=1.4, label=conf)
    ax2.set_ylabel("rank (1 = leader)")
    ax2.invert_yaxis()
    ax2.set_yticks([1, 2, 3, 4, 5, 6])
    ax2.set_title("Which confederation leads the slice, by month", fontsize=10)
    ax2.grid(alpha=0.2)
    fig.tight_layout()

    s_out = s.drop(columns=["threshold"]).sort_values("final_rank")
    s_out.columns = ["confederation", "SES alpha", "months led", "% window led",
                     "final share", "final rank", "forecast share (+12m)"]
    return fig, s_out


# ----------------------------------------------------------------------
# app
# ----------------------------------------------------------------------
def build_app():
    import gradio as gr

    with gr.Blocks(title="Bayesian World Cup Prediction", css=APP_CSS) as demo:
        gr.Markdown(
            "<div class='hero-card'>"
            "<div class='hero-kicker'>Bayesian decision support · 2026 replay</div>"
            "<h1>World Cup Prediction</h1>"
            "<div>Probabilistic lenses report <strong>distributions with intervals</strong>, "
            "never point-only certainty. Ranking Dynamics is a separate deterministic, "
            "descriptive SES view. The <code>do()</code> lens is a counterfactual simulation, "
            "not an estimated causal effect. All inference is pre-computed.</div>"
            "</div>"
        )
        with gr.Tab("1 - Continental Strength"):
            gr.Markdown(
                "<div class='method-note'><strong>Model-quality disclosure.</strong> "
                "The retained four-chain strength fit has <strong>5 divergences</strong>, "
                "maximum <strong>R-hat 1.081</strong>, and minimum <strong>ESS 39.7</strong>. "
                "Treat it as an exploratory structural comparison. UEFA's large team pool can "
                "lower its typical-team mean relative to CONMEBOL's smaller, stronger pool.</div>"
            )
            with gr.Row():
                c_a = gr.Dropdown(CONFEDERATIONS, value="UEFA", label="Continent A")
                c_b = gr.Dropdown(CONFEDERATIONS, value="CONMEBOL", label="Continent B")
                per = gr.Radio(PERIODS, value="Monthly", label="Period aggregation")
            st_out = gr.Plot(label="Trends + difference")
            st_table = gr.Dataframe(headers=["pair", "avg log-diff", "90% lo", "90% hi",
                                             "avg Elo-pts diff", "P(A>B)"],
                                    label="Pairwise overall-average differences (15 pairs)")
            c_a.change(tab_strength, [c_a, c_b, per], [st_out, st_table])
            c_b.change(tab_strength, [c_a, c_b, per], [st_out, st_table])
            per.change(tab_strength, [c_a, c_b, per], [st_out, st_table])
            demo.load(tab_strength, [c_a, c_b, per], [st_out, st_table])
        with gr.Tab("2 - Forest Plot"):
            metric = gr.Radio(["continental champion effect", "confederation offset", "beta_elo"],
                              value="continental champion effect", label="Parameter")
            forest_out = gr.Plot()
            metric.change(tab_forest, metric, forest_out)
            demo.load(tab_forest, metric, forest_out)
        with gr.Tab("3 - Monte Carlo"):
            with gr.Row():
                region = gr.Dropdown(["All"] + CONFEDERATIONS, value="All", label="Regional filter")
                top_n = gr.Slider(5, 48, value=15, step=1, label="Top-N teams")
            mc_out = gr.Plot()
            region.change(tab_monte_carlo, [region, top_n], mc_out)
            top_n.change(tab_monte_carlo, [region, top_n], mc_out)
            demo.load(tab_monte_carlo, [region, top_n], mc_out)
        with gr.Tab("4 - do()-What-If"):
            region2 = gr.Dropdown(["All"] + CONFEDERATIONS, value="All", label="Regional filter")
            do_out = gr.Plot()
            region2.change(tab_do, region2, do_out)
            demo.load(tab_do, region2, do_out)
        with gr.Tab("5 - Prior Predictive"):
            mode = gr.Radio(["Prior (before data)", "Posterior (after data)"],
                            value="Prior (before data)", label="What-If: prior strength")
            pp_out = gr.Plot()
            mode.change(tab_prior_predictive, mode, pp_out)
            demo.load(tab_prior_predictive, mode, pp_out)
        with gr.Tab("6 - Causal: Continent to Winner"):
            gr.Markdown(
                "**Causal structure - honest framing.** The DAG below says `team_strength` "
                "(latent) sits on every back-door path from `continental_champion` to "
                "`wc_outcome`, so **no observable-only adjustment set exists** and a causal "
                "effect of champion status is **not identifiable** from 22 World Cups. "
                "The posterior plots are *associational* (structural), and the do()-What-If "
                "tab is a *counterfactual simulation*."
            )
            with gr.Row():
                gr.Image(os.path.join(DATA_DIR, "dag.png"), label="Assumed causal structure")
            with gr.Row():
                c_metric = gr.Radio(["confederation offset", "continental champion effect",
                                     "beta_elo"],
                                    value="confederation offset",
                                    label="Posterior effect (associational)")
                c_out = gr.Plot()
                c_metric.change(tab_causal, c_metric, c_out)
                demo.load(tab_causal, c_metric, c_out)
            gr.Markdown(load_text("dag_validation.md"))
        with gr.Tab("7 - DAG Assumption Tests"):
            check = gr.Radio(["C1", "C2", "C3", "C4"], value="C1",
                             label="Check: C1 confounding | C2 independence proxy | "
                                   "C3 sensitivity | C4 balance")
            check_fig = gr.Plot()
            check_text = gr.Markdown()
            check_table = gr.Dataframe(label="Check results")
            check.change(tab_dag_check, check, [check_fig, check_text, check_table])
            demo.load(tab_dag_check, check, [check_fig, check_text, check_table])
        with gr.Tab("8 - Ranking Dynamics"):
            gr.Markdown(
                "<div class='method-note'><strong>Descriptive only.</strong> Curves are "
                "deterministic simple-exponential-smoothing fits to observed top-N shares. "
                "The dashed 12-month extension is a scenario forecast with no probabilistic "
                "uncertainty interval and is not a Bayesian posterior forecast.</div>"
            )
            thr = gr.Radio(THRESHOLDS, value=20,
                           label="Top-N slice (share of the top-N Elo ranking per confederation)")
            rank_fig = gr.Plot()
            rank_table = gr.Dataframe(label="Running leader summary (which line is ahead)")
            thr.change(tab_ranking, thr, [rank_fig, rank_table])
            demo.load(tab_ranking, thr, [rank_fig, rank_table])
        gr.Markdown(
            "<div class='quiet-signature'><a href='https://deerflow.tech' target='_blank' "
            "rel='noopener'>Created by Deerflow</a></div>"
        )
    return demo


# ----------------------------------------------------------------------
# gate helper: render all tabs to PNGs without launching
# ----------------------------------------------------------------------
def render_all(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.abspath(out_dir)
    figs = {}
    fig, _ = tab_strength("UEFA", "CONMEBOL", "Monthly")
    figs["tab_1_strength.png"] = fig
    figs["tab_2_forest.png"] = tab_forest("continental champion effect")
    figs["tab_3_monte_carlo.png"] = tab_monte_carlo("All", 15)
    figs["tab_4_do.png"] = tab_do("All")
    figs["tab_5_prior_predictive.png"] = tab_prior_predictive("Prior (before data)")
    figs["tab_6_causal.png"] = tab_causal("confederation offset")
    for check in ["C1", "C2", "C3", "C4"]:
        fig, _, _ = tab_dag_check(check)
        figs[f"tab_7_dag_check_{check}.png"] = fig
    fig, _ = tab_ranking(20)
    figs["tab_8_ranking_dynamics.png"] = fig
    for name, fig in figs.items():
        fig.savefig(os.path.join(out_dir, name), bbox_inches="tight")
        plt.close(fig)
    print("rendered:", ", ".join(sorted(figs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", nargs="?", const="rendered", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.render is not None:
        render_all(args.out or "rendered")
    else:
        import gradio as gr  # imported here so --render works without gradio
        build_app().launch(server_name="0.0.0.0", server_port=7860)
