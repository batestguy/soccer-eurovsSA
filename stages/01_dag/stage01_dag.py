# -*- coding: utf-8 -*-
# =====================================================================
# STAGE 01 — CAUSAL DAG
# Bayesian World Cup Prediction
#
# Produces: dag.dot, dag.png, dag_validation.md (implied independencies
# + identification statement) and pushes them to the GitHub repo.
#
# Pure networkx (no dagitty dependency). The graph is small, so the
# d-separation / back-door analysis is computed directly:
#   - implied CIs via the global Markov property, verified with nx.d_separated
#   - back-door adjustment sets by enumerating blocking sets
#
# DAG = modeling discipline + user-facing explainability + do()-simulation
# generator. NEVER labeled "estimated causal effect": continental_champion
# is confounded by latent team_strength and we only have 22 World Cups.
# =====================================================================

import os
import sys
import base64
import shutil
import subprocess
import datetime as dt
from itertools import combinations

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GITHUB_REPO = "batestguy/soccer-eurovsSA"

NODES = ["team_strength", "confederation", "fifa_ranking",
         "continental_champion", "wc_outcome"]
LATENT = {"team_strength"}
EDGES = [("confederation", "team_strength"),
         ("confederation", "continental_champion"),
         ("team_strength", "fifa_ranking"),
         ("team_strength", "continental_champion"),
         ("team_strength", "wc_outcome"),
         ("fifa_ranking", "wc_outcome"),
         ("continental_champion", "wc_outcome")]

NODE_LABELS = {
    "team_strength": "team strength\n(latent)",
    "confederation": "confederation",
    "fifa_ranking": "FIFA points\n(elo proxy)",
    "continental_champion": "continental\nchampion",
    "wc_outcome": "WC outcome",
}


def build_graph():
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)
    return g


def d_separated(g, x, y, z):
    """d-separation via the moral-criterion (works on any networkx DiGraph)."""
    anc = set(x) | set(y) | set(z)
    for n in list(anc):
        anc |= nx.ancestors(g, n)
    sub = g.subgraph(anc)
    moral = nx.Graph()
    moral.add_nodes_from(sub.nodes)
    moral.add_edges_from(sub.edges())
    for n in sub.nodes:
        parents = list(sub.predecessors(n))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral.add_edge(parents[i], parents[j])
    moral.remove_nodes_from(z)
    return not nx.has_path(moral, next(iter(x)), next(iter(y)))


def implied_independencies(g):
    """Global Markov independencies: node _||_ (non-descendants \ parents) | parents."""
    out = []
    for v in g.nodes:
        parents = set(g.predecessors(v))
        non_desc = set(g.nodes) - nx.descendants(g, v) - {v}
        for u in sorted(non_desc - parents):
            sep = sorted(parents)
            check = d_separated(g, {u}, {v}, set(sep))
            out.append((u, v, sep, check))
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def blocks(g, Z, path):
    """Return True if conditioning set Z BLOCKS the (undirected) path."""
    for i in range(1, len(path) - 1):
        v = path[i]
        prev, nxt = path[i - 1], path[i + 1]
        collider = g.has_edge(prev, v) and g.has_edge(nxt, v)
        if collider:
            if v not in Z and not (set(nx.descendants(g, v)) & set(Z)):
                return True
        else:
            if v in Z:
                return True
    return False


def adjustment_sets(g, exposure, outcome):
    g_u = g.to_undirected()
    backdoors = [p for p in nx.all_simple_paths(g_u, exposure, outcome)
                 if g.has_edge(p[1], p[0])]
    candidates = [set(c) for r in range(len(NODES) + 1)
                  for c in combinations(set(NODES) - {exposure, outcome}, r)]
    valid = []
    for Z in candidates:
        if all(blocks(g, Z, p) for p in backdoors):
            valid.append(Z)
    valid.sort(key=lambda z: (len(z), sorted(z)))
    return backdoors, valid


def arrow_path(g, path):
    """Render a path with each directed edge drawn as `a -> b` or `b <- a`."""
    parts = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if g.has_edge(a, b):
            parts.append("-> " + b)
        else:
            parts.append("<- " + b)
    return " ".join(parts)


# ---------------------------------------------------------------------
def render_dag(path):
    """Clean, graphviz-style rendering with matplotlib (no external dot binary)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    pos = {"confederation": (0.06, 0.22),
           "team_strength": (0.06, 0.80),
           "fifa_ranking": (0.45, 0.80),
           "continental_champion": (0.45, 0.20),
           "wc_outcome": (0.90, 0.50)}

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=200)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.axis("off")

    # edges
    for (u, v) in EDGES:
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>",
                                     mutation_scale=24, lw=1.7,
                                     color="#455A64", shrinkA=12, shrinkB=14))

    info = {
        "team_strength": ("team strength\n(latent)", True),
        "confederation": ("confederation", False),
        "fifa_ranking": ("FIFA points\n(elo proxy)", False),
        "continental_champion": ("continental\nchampion", False),
        "wc_outcome": ("WC outcome", False),
    }
    for n in NODES:
        label, latent = info[n]
        x, y = pos[n]
        w, h = (0.20, 0.14) if not latent else (0.20, 0.14)
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.008", linewidth=2.0,
                             edgecolor="#78909C" if latent else "#E65100",
                             linestyle=(0, (5, 3)) if latent else "solid",
                             facecolor="#ECEFF1" if latent else "#FFF3E0",
                             zorder=3)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10, fontweight="bold" if not latent else "normal",
                color="#263238", zorder=4)

    # legend
    ax.plot([-0.045, 0.045], [1.00, 1.00], transform=ax.transAxes,
            linestyle=(0, (5, 3)), color="#78909C", lw=2)
    ax.text(0.055, 1.00, "latent (unmeasured)", transform=ax.transAxes,
            fontsize=8.5, va="center", color="#37474F")
    ax.plot([0.30, 0.39], [1.00, 1.00], transform=ax.transAxes,
            color="#E65100", lw=2)
    ax.text(0.40, 1.00, "observed", transform=ax.transAxes,
            fontsize=8.5, va="center", color="#37474F")
    ax.plot([0.58, 0.67], [1.00, 1.00], transform=ax.transAxes,
            color="#455A64", lw=2, marker=">", ms=6)
    ax.text(0.68, 1.00, "causal arrow", transform=ax.transAxes,
            fontsize=8.5, va="center", color="#37474F")

    ax.set_title("World Cup outcome — assumed causal structure\n"
                 "This DAG is modeling discipline + explainability + the Monte Carlo do()-simulation generator",
                 fontsize=11.5, fontweight="bold", pad=26)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
def push_to_github(repo, token, files_root, commit_msg):
    if not token:
        return False, "no token"
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
    try:
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/{repo}.git", clone_dir],
                       env=env, check=True, capture_output=True, text=True)
        shutil.copytree(files_root, clone_dir, dirs_exist_ok=True)
        subprocess.run(["git", "add", "-A"], cwd=clone_dir, env=env,
                       check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=clone_dir, env=env,
                       check=True, capture_output=True, text=True)
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, env=env,
                       check=True, capture_output=True, text=True)
        return True, "pushed to " + repo
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or e.stdout or str(e))[:400]


def main():
    out_root = "/content/soccerdl_out" if os.path.exists("/tmp/github_token") else "output"
    data_dir = os.path.join(out_root, "data")
    stage_dir = os.path.join(out_root, "stages", "01_dag")
    for d in (data_dir, stage_dir):
        os.makedirs(d, exist_ok=True)

    print("Python:", sys.version.split()[0], "| networkx:", nx.__version__)
    g = build_graph()

    # DOT
    dot = "digraph dag {\n  rankdir=LR;\n" + \
          "\n".join(f'  "{u}" -> "{v}";' for u, v in EDGES) + "\n}"
    with open(os.path.join(data_dir, "dag.dot"), "w", encoding="utf-8") as f:
        f.write(dot)

    # PNG
    render_dag(os.path.join(data_dir, "dag.png"))

    # Analysis
    indep = implied_independencies(g)
    backdoors, valid_adj = adjustment_sets(g, "continental_champion", "wc_outcome")
    min_adj = [z for z in valid_adj if len(z) == min(len(x) for x in valid_adj)] if valid_adj else []
    any_without_latent = any(not (z & LATENT) for z in valid_adj)

    print("implied independencies:")
    for u, v, sep, check in indep:
        status = "verified (d-separated)" if check else "FAILED d-separation!"
        print(f"  {u} _||_ {v} | {{{', '.join(sep)}}}  [{status}]")
    print("backdoor paths from continental_champion to wc_outcome:")
    for p in backdoors:
        print("  " + arrow_path(g, p))
    print("valid adjustment sets:", [sorted(z) for z in valid_adj])
    print("minimal:", [sorted(z) for z in min_adj])

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    indep_lines = [f"- `{u} _||_ {v} | {{{', '.join(sep)}}}` "
                   + ("" if check else "   [FAILED d-separation!]")
                   for u, v, sep, check in indep]
    md = [
        "# DAG VALIDATION — stage 01",
        f"Generated: {now}",
        "",
        "## Model",
        "Nodes: `" + ", ".join(NODES) + "`",
        "",
        "## Edge rationale",
        "- confederation -> team_strength: hierarchical pooling — region sets the prior base.",
        "- team_strength -> fifa_ranking: ratings measure strength (measurement model).",
        "- team_strength -> continental_champion: strong teams win continental trophies.",
        "- team_strength -> wc_outcome: strength is the dominant direct driver.",
        "- continental_champion -> wc_outcome: the feature of interest.",
        "- fifa_ranking -> wc_outcome: current form carries predictive information",
        "  (the GP 'macro-trend' lens).",
        "- confederation -> continental_champion: the aligned trophy's identity and",
        "  competitiveness depend on the region (OFC vs UEFA).",
        "",
        "## Implied conditional independencies (global Markov property, verified)",
        "\n".join(indep_lines),
        "",
        "Every statement conditions on `team_strength` (latent), so none is directly",
        "testable with observed data. The observable proxy check that remains: within a",
        "confederation, champion and non-champion Elo distributions overlap — the reason",
        "a point estimate would be dishonest.",
        "",
        "## Back-door paths: continental_champion -> wc_outcome",
        "\n".join(" - " + arrow_path(g, p) for p in backdoors),
        "",
        "## Causal identification — the honest statement",
        "Valid back-door adjustment sets (enumerated):",
        "\n".join(" - `{" + ", ".join(sorted(z)) + "}`" for z in valid_adj),
        "",
        f"Every valid set includes the **latent** `team_strength`"
        + (" — confirmed: no observable-only set exists." if any_without_latent is False
           else " — but an observable-only set exists? review."),
        "With 22 World Cups we therefore CANNOT identify a causal effect of",
        "continental_champion from observational data. Claims about it are (1) a",
        "*descriptive association* from the hierarchical regression, and (2)",
        "*counterfactual simulations* from the Monte Carlo Oracle via `do()` on this DAG",
        "(run the generative model with continental_champion forced to 0 vs 1).",
        "Nothing is ever labeled an 'estimated causal effect'.",
        "",
        "## Data gaps feeding this stage",
        "- AFCON 2025 and Gold Cup 2025 champions are UNKNOWN (post-curation cutoff);",
        "  they sit inside the 18-month window for 2026 and must be filled before",
        "  building the 2026 `continental_champion` feature.",
    ]
    with open(os.path.join(data_dir, "dag_validation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    source_file = "/content/stage01_dag.py"
    if os.path.exists(source_file):
        shutil.copyfile(source_file, os.path.join(stage_dir, "stage01_dag.py"))
    else:
        try:
            shutil.copyfile(__file__, os.path.join(stage_dir, "stage01_dag.py"))
        except Exception:
            pass

    print("artifacts:", ", ".join(sorted(os.listdir(data_dir))))

    # push
    if os.path.exists("/tmp/github_token"):
        token = open("/tmp/github_token").read().splitlines()[0].strip()
        assets = "/content/soccerdl_assets"
        if os.path.exists(assets):
            shutil.rmtree(assets)
        shutil.copytree(os.path.join(out_root, "data"), os.path.join(assets, "data"))
        shutil.copytree(os.path.join(out_root, "stages"), os.path.join(assets, "stages"))
        with open(os.path.join(assets, "README.md"), "w", encoding="utf-8") as f:
            f.write("""# Soccer Deep Learning — Bayesian World Cup Prediction

Stage 01 = causal DAG (modeling discipline + explainability + do()-simulation).
Artifacts:
- `data/dag.dot`              graphviz source
- `data/dag.png`              rendered DAG (latent strength shown as a square)
- `data/dag_validation.md`    implied independencies + identification statement
""")
        ok, msg = push_to_github(GITHUB_REPO, token, assets,
                                 "stage 01: causal DAG (structure + validation)")
        print("commit:", ok, "|", msg)
    else:
        print("commit: skipped (no /tmp/github_token)")


if __name__ == "__main__":
    main()
