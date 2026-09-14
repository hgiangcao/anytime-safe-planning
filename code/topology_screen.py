"""Map-only pre-screen for the validity boundary.

Reviewers asked whether topology features predict the boundary *before* the
planner is run.  This script computes three candidate statistics from the map
file alone -- no simulation, no flow solve -- and scores each against the
model-level term already measured by the frozen native check
(results/checks/tie_resolution.json, d_TU at eps_tie = 0.01: what the fluid
field is worth once a planner follows it faithfully):

  rho1  dead-end share       fraction of free cells with <= 1 free 4-neighbour
  rho2  corridor share       fraction of free cells with <= 2 free 4-neighbours
  art   articulation share   fraction of free cells that are cut vertices

A statistic "separates" if some threshold puts every map whose model-level term
is positive on one side and every negative map on the other.  With one open map
and four obstacle maps that is a single threshold fit to five points, so the
result is registered as a prediction for new topologies, not a validated
classifier -- the script reports the separating margin and the rank correlation
with the magnitude so both are on the record.

Writes results/topology_screen.json and paper/topology_screen_numbers.tex.
Run:  ../.venv/bin/python code/topology_screen.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from mapf import maps as mapmod                      # noqa: E402
from common import BENCH_MAPS                        # noqa: E402

TIE = os.path.join(PROJ, "results", "checks", "tie_resolution.json")
OUT = os.path.join(PROJ, "results", "topology_screen.json")
TEX = os.path.join(PROJ, "paper", "topology_screen_numbers.tex")


# ---- map-only statistics --------------------------------------------------
def free_adjacency(name):
    """Adjacency list over free cells, indexed 0..n_free-1."""
    m = mapmod.MAPS[name]()
    ci = m["cell_index"]
    return [[int(ci[u]) for u in nb] for nb in mapmod.neighbors_lists(m)]


def articulation_flags(adj):
    """Iterative Hopcroft-Tarjan cut vertices (recursion would overflow on the
    5,699-cell warehouse)."""
    n = len(adj)
    disc = [-1] * n
    low = [0] * n
    art = [False] * n
    t = 0
    for s in range(n):
        if disc[s] != -1:
            continue
        disc[s] = low[s] = t
        t += 1
        stack = [(s, -1, iter(adj[s]))]
        root_children = 0
        while stack:
            u, p, it = stack[-1]
            advanced = False
            for v in it:
                if disc[v] == -1:
                    disc[v] = low[v] = t
                    t += 1
                    if u == s:
                        root_children += 1
                    stack.append((v, u, iter(adj[v])))
                    advanced = True
                    break
                elif v != p:
                    low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                if stack:
                    pu = stack[-1][0]
                    low[pu] = min(low[pu], low[u])
                    if pu != s and low[u] >= disc[pu]:
                        art[pu] = True
        if root_children > 1:
            art[s] = True
    return art


def map_stats(name):
    adj = free_adjacency(name)
    deg = np.array([len(a) for a in adj])
    return dict(
        cells=int(deg.size),
        rho1=float((deg <= 1).mean()),
        rho2=float((deg <= 2).mean()),
        art=float(np.mean(articulation_flags(adj))),
    )


# ---- measured target ------------------------------------------------------
def model_term_by_map():
    """Mean model-level term (d_TU at eps_tie = 0.01) per map, from the frozen
    native check.  No new runs."""
    cells = json.load(open(TIE))["cells"]
    by = {}
    for c in cells:
        by.setdefault(c["map"], []).append(c["d_TU_fine"]["mean"])
    return {k: float(np.mean(v)) for k, v in by.items()}, {
        k: len(v) for k, v in by.items()}


# ---- scoring --------------------------------------------------------------
def separates(values, signs):
    """Return (ok, lo, hi): does a threshold exist with every positive-sign map
    below every negative-sign map?  lo/hi bracket the admissible thresholds."""
    pos = [values[k] for k in values if signs[k] > 0]
    neg = [values[k] for k in values if signs[k] < 0]
    if not pos or not neg:
        return False, None, None
    lo, hi = max(pos), min(neg)
    return lo < hi, lo, hi


def spearman(a, b):
    ra = np.argsort(np.argsort(a)) + 1.0
    rb = np.argsort(np.argsort(b)) + 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    model, n_cells = model_term_by_map()
    names = list(BENCH_MAPS)
    assert set(names) == set(model), (names, sorted(model))

    stats = {n: map_stats(n) for n in names}
    signs = {n: (1 if model[n] > 0 else -1) for n in names}

    scored = {}
    for feat in ("rho1", "rho2", "art"):
        vals = {n: stats[n][feat] for n in names}
        ok, lo, hi = separates(vals, signs)
        scored[feat] = dict(
            values={n: round(vals[n], 4) for n in names},
            separates=bool(ok),
            threshold_lo=None if lo is None else round(lo, 4),
            threshold_hi=None if hi is None else round(hi, 4),
            spearman_with_model=round(
                spearman([vals[n] for n in names], [model[n] for n in names]), 3),
        )

    rec = dict(
        question="can a map-only statistic predict the sign of the model-level "
                 "term before the planner is run?",
        target="mean d_TU at eps_tie=0.01 per map (model-level term), from "
               "results/checks/tie_resolution.json",
        source="frozen native check; no new simulation",
        n_maps=len(names),
        n_open=sum(1 for n in names if signs[n] > 0),
        cells_per_map=n_cells,
        model_term={n: round(model[n], 3) for n in names},
        stats={n: {k: round(v, 4) for k, v in stats[n].items()} for n in names},
        features=scored,
        caveat="one open map and four obstacle maps: a separating threshold is "
               "one free parameter fit to five points, so this is a registered "
               "prediction, not a validated classifier",
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)

    open_map = [n for n in names if signs[n] > 0]
    assert len(open_map) == 1, open_map
    om = open_map[0]
    obs = [n for n in names if signs[n] < 0]
    r2 = scored["rho2"]["values"]
    wh = "warehouse-10-20-10-2-1"

    lines = [
        "% AUTOGENERATED by code/topology_screen.py -- do not edit.",
        f"\\newcommand{{\\ScrMaps}}{{{len(names)}}}",
        f"\\newcommand{{\\ScrObs}}{{{len(obs)}}}",
        f"\\newcommand{{\\ScrOpenRho}}{{{r2[om]:.3f}}}",
        f"\\newcommand{{\\ScrObsRhoMin}}{{{min(r2[n] for n in obs):.3f}}}",
        f"\\newcommand{{\\ScrObsRhoMax}}{{{max(r2[n] for n in obs):.3f}}}",
        f"\\newcommand{{\\ScrWhRhoOne}}{{{stats[wh]['rho1']:.3f}}}",
        f"\\newcommand{{\\ScrWhArt}}{{{stats[wh]['art']:.3f}}}",
        f"\\newcommand{{\\ScrWhModel}}{{{model[wh]:+.2f}}}",
        f"\\newcommand{{\\ScrRhoRank}}{{{scored['rho2']['spearman_with_model']:.2f}}}",
    ]
    with open(TEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"maps={len(names)} open={len(open_map)}")
    for n in names:
        s = stats[n]
        print(f"  {n:<24} rho1={s['rho1']:.4f} rho2={s['rho2']:.4f} "
              f"art={s['art']:.4f}  model={model[n]:+.3f}")
    for feat, sc in scored.items():
        print(f"  {feat:<5} separates={sc['separates']!s:<5} "
              f"window=({sc['threshold_lo']}, {sc['threshold_hi']}) "
              f"spearman={sc['spearman_with_model']:+.2f}")
    print(f"wrote {os.path.relpath(OUT, PROJ)} and {os.path.relpath(TEX, PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
