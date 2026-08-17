"""Stage 05 release gate.

Validates the static artifact bundle, exercises every selector option, renders
all 11 canonical figures, and confirms that the staged and Space app files are
byte-identical.  This script never imports PyMC and never fits a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "stages" / "05_app"
SPACE_DIR = ROOT / "spaces"
DATA_DIR = SPACE_DIR / "data"

SCHEMAS = {
    "strength_trends.csv": [
        "confederation", "month", "log_mean", "log_p5", "log_p95",
        "elo_mean", "elo_p5", "elo_p95",
    ],
    "strength_diff_curves.csv": [
        "pair", "continent_a", "continent_b", "month", "log_mean",
        "log_p5", "log_p95", "elo_mean", "elo_p5", "elo_p95",
    ],
    "strength_pairwise.csv": [
        "pair", "continent_a", "continent_b", "level_mean", "level_p5",
        "level_p95", "avg_log_diff_mean", "avg_log_diff_p5",
        "avg_log_diff_p95", "avg_elo_diff_mean", "avg_elo_diff_p5",
        "avg_elo_diff_p95", "p_a_stronger",
    ],
    "monte_carlo_results.csv": [
        "team", "confederation", "elo_pre_wc", "continental_champion",
        "p_win_mean", "p_win_p5", "p_win_p95", "win_freq",
    ],
    "do_contrast.csv": [
        "team", "confederation", "p_win_do1", "p_win_do0",
        "diff_mean", "diff_p5", "diff_p95",
    ],
    "elite_composition.csv": [
        "month", "threshold", "confederation", "share", "share_pct",
    ],
    "elite_fit.csv": [
        "month", "threshold", "confederation", "fitted", "fitted_pct",
        "is_forecast",
    ],
    "elite_summary.csv": [
        "threshold", "confederation", "alpha", "months_led",
        "pct_window_led", "final_share", "final_rank", "forecast_share",
    ],
    "dag_checks.csv": [
        "check_id", "check_name", "metric", "value", "ci_lo", "ci_hi",
        "p_value", "n", "note",
    ],
    "prior_model_frame.csv": [
        "wc_year", "team", "confederation", "elo_pre_wc",
        "continental_champion", "winner_position", "is_winner", "elo_z",
    ],
    "ranking_chronology.csv": ["month", "team", "elo"],
    "team_confederations.csv": ["team", "confederation"],
    "conf_membership.csv": ["team", "change_month", "conf_after", "conf_before"],
}

STATIC_FILES = [
    "posterior.nc", "prior.nc", "dag.png", "dag_validation.md",
    "dag_checks_report.md", "strength_model_meta.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_app():
    os.environ["SOCCERDL_DATA"] = str(DATA_DIR)
    spec = importlib.util.spec_from_file_location("stage05_release_app", STAGE_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_figure(fig, label: str) -> None:
    _require(fig is not None and len(fig.axes) > 0, f"{label}: no axes")
    artists = sum(
        len(ax.lines) + len(ax.patches) + len(ax.collections) + len(ax.images)
        for ax in fig.axes
    )
    _require(artists > 0, f"{label}: no plotted artists")
    fig.canvas.draw()
    plt.close(fig)


def validate_files_and_schemas() -> dict:
    for name, expected in SCHEMAS.items():
        path = DATA_DIR / name
        _require(path.is_file() and path.stat().st_size > 0, f"missing/empty: {name}")
        actual = pd.read_csv(path, nrows=0).columns.tolist()
        _require(actual == expected, f"{name}: schema mismatch {actual}")
    for name in STATIC_FILES:
        path = DATA_DIR / name
        _require(path.is_file() and path.stat().st_size > 0, f"missing/empty: {name}")

    mc = pd.read_csv(DATA_DIR / "monte_carlo_results.csv")
    _require(len(mc) == 48 and mc["team"].nunique() == 48, "Monte Carlo field is not 48 teams")
    _require(np.isclose(mc["p_win_mean"].sum(), 1.0, atol=1e-10), "P(win) does not sum to 1")
    _require(np.isclose(mc["win_freq"].sum(), 1.0, atol=1e-10), "winner frequency does not sum to 1")
    _require(((mc["p_win_p5"] <= mc["p_win_mean"]) &
              (mc["p_win_mean"] <= mc["p_win_p95"])).all(), "Monte Carlo intervals invalid")

    contrast = pd.read_csv(DATA_DIR / "do_contrast.csv")
    _require(len(contrast) == 48 and contrast["team"].nunique() == 48,
             "do()-contrast field is not 48 teams")
    _require(((contrast["diff_p5"] <= contrast["diff_mean"]) &
              (contrast["diff_mean"] <= contrast["diff_p95"])).all(),
             "do()-contrast intervals invalid")

    trends = pd.read_csv(DATA_DIR / "strength_trends.csv")
    _require(len(trends) == 2496 and trends["month"].nunique() == 416,
             "strength trend coverage is not 6 x 416")
    _require(((trends["log_p5"] <= trends["log_mean"]) &
              (trends["log_mean"] <= trends["log_p95"])).all(),
             "strength log intervals invalid")
    _require(pd.read_csv(DATA_DIR / "strength_pairwise.csv")["pair"].nunique() == 15,
             "strength pairwise table is not 15 pairs")

    elite = pd.read_csv(DATA_DIR / "elite_composition.csv")
    share_sums = elite.groupby(["month", "threshold"])["share_pct"].sum()
    _require(np.allclose(share_sums.to_numpy(), 100.0, atol=1e-8),
             "Metric-A shares do not sum to 100%")
    _require(sorted(elite["threshold"].unique().tolist()) == [5, 10, 20],
             "elite thresholds are incomplete")
    fit = pd.read_csv(DATA_DIR / "elite_fit.csv")
    forecast_counts = fit[fit["is_forecast"].astype(str).str.lower() == "true"].groupby(
        ["threshold", "confederation"]
    ).size()
    _require(len(forecast_counts) == 18 and (forecast_counts == 12).all(),
             "SES forecast is not 12 months for every series")

    checks = pd.read_csv(DATA_DIR / "dag_checks.csv")
    _require(set(checks["check_id"]) == {"C1", "C2", "C3", "C4"},
             "DAG checks C1-C4 are incomplete")

    posterior = az.from_netcdf(DATA_DIR / "posterior.nc")
    try:
        _require({"beta_elo", "cc_effect", "conf_offset", "p_2022"}.issubset(
            set(posterior.posterior.data_vars)), "posterior variables missing")
    finally:
        posterior.close()
    prior = az.from_netcdf(DATA_DIR / "prior.nc")
    try:
        _require("p_2022" in prior.prior.data_vars, "prior p_2022 missing")
    finally:
        prior.close()

    stage_app = STAGE_DIR / "app.py"
    space_app = SPACE_DIR / "app.py"
    stage_readme = STAGE_DIR / "README_SPACES.md"
    space_readme = SPACE_DIR / "README.md"
    _require(stage_app.read_bytes() == space_app.read_bytes(), "app.py byte parity failed")
    _require(stage_readme.read_bytes() == space_readme.read_bytes(), "README byte parity failed")
    source = space_app.read_text(encoding="utf-8").lower()
    _require("import pymc" not in source and "pm.sample(" not in source,
             "runtime app contains a model-fitting call")
    _require("gp_lens" not in source, "obsolete GP-lens packaging remains in app")

    return {
        "p_win_sum": float(mc["p_win_mean"].sum()),
        "win_freq_sum": float(mc["win_freq"].sum()),
        "strength_months": int(trends["month"].nunique()),
        "pairwise_rows": 15,
        "app_sha256": _sha256(stage_app),
    }


def validate_aggregations(app) -> dict:
    trends = app.load_csv("strength_trends.csv")
    values = ["log_mean", "log_p5", "log_p95", "elo_mean", "elo_p5", "elo_p95"]
    uefa = trends[trends["confederation"] == "UEFA"]
    expected = {"Monthly": 416, "Quarterly": 139, "Annual": 35}
    observed = {period: len(app._resample(uefa, period, values)) for period in expected}
    _require(observed == expected, f"strength aggregation counts wrong: {observed}")
    return observed


def validate_selectors(app) -> dict:
    calls = 0

    # Every confederation appears in both selector positions; every period is exercised.
    for idx, conf_a in enumerate(app.CONFEDERATIONS):
        conf_b = app.CONFEDERATIONS[(idx + 1) % len(app.CONFEDERATIONS)]
        period = app.PERIODS[idx % len(app.PERIODS)]
        fig, table = app.tab_strength(conf_a, conf_b, period)
        _require(len(table) == 15, "strength table lost a pair")
        _assert_figure(fig, f"strength {conf_a}/{conf_b}/{period}")
        calls += 1

    metrics = ["continental champion effect", "confederation offset", "beta_elo"]
    for metric in metrics:
        _assert_figure(app.tab_forest(metric), f"forest {metric}")
        _assert_figure(app.tab_causal(metric), f"causal {metric}")
        calls += 2

    for idx, region in enumerate(["All"] + app.CONFEDERATIONS):
        top_n = 5 if idx % 2 == 0 else 48
        _assert_figure(app.tab_monte_carlo(region, top_n), f"Monte Carlo {region}/{top_n}")
        _assert_figure(app.tab_do(region), f"do {region}")
        calls += 2

    for mode in ["Prior (before data)", "Posterior (after data)"]:
        _assert_figure(app.tab_prior_predictive(mode), f"prior toggle {mode}")
        calls += 1

    for check_id in ["C1", "C2", "C3", "C4"]:
        fig, text, table = app.tab_dag_check(check_id)
        _require(bool(text.strip()) and not table.empty, f"{check_id}: empty report/table")
        _assert_figure(fig, f"DAG {check_id}")
        calls += 1

    for threshold in app.THRESHOLDS:
        fig, table = app.tab_ranking(threshold)
        _require(len(table) == 6, f"top-{threshold}: incomplete summary")
        _assert_figure(fig, f"ranking top-{threshold}")
        calls += 1

    demo = app.build_app()
    config = demo.get_config_file()
    tabs = [item["props"]["label"] for item in config["components"]
            if item.get("type") == "tabitem"]
    _require(len(tabs) == 8, f"expected 8 tabs, found {len(tabs)}: {tabs}")
    return {"function_calls": calls, "tabs": tabs}


def validate_canonical_render(app, out_dir: Path) -> dict:
    app.render_all(out_dir)
    pngs = sorted(out_dir.glob("*.png"))
    _require(len(pngs) == 11, f"expected 11 canonical figures, found {len(pngs)}")
    sizes = {path.name: path.stat().st_size for path in pngs}
    _require(all(size > 5_000 for size in sizes.values()), f"empty/small figure: {sizes}")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None,
                        help="Directory for the 11 canonical PNGs (temporary by default).")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON report path.")
    args = parser.parse_args()

    summary = {"artifacts": validate_files_and_schemas()}
    app = _load_app()
    summary["aggregations"] = validate_aggregations(app)
    summary["selectors"] = validate_selectors(app)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        summary["figures"] = validate_canonical_render(app, args.out)
    else:
        with tempfile.TemporaryDirectory(prefix="soccerdl-stage05-") as temp:
            summary["figures"] = validate_canonical_render(app, Path(temp))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("STAGE 05 RELEASE GATE: PASS")


if __name__ == "__main__":
    main()
