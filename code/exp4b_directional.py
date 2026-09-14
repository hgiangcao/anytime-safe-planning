"""Exp 4b - directional evidence for the emergent one-way-lane claim.

The previous version supported "guidance induces one-way lanes" only with
heatmaps of UNDIRECTED cell-traversal counts, which sum direction out and in
which no lane structure is visible even in principle. This experiment measures
direction explicitly:

  * net directed flow  n_e = f_e - f_rev(e)  as a vector field per cell,
  * the directionality index
        Dir = sum_e |f_e - f_rev(e)| / sum_e (f_e + f_rev(e))  in [0, 1],
    reported for the stationary-flow candidate and for realised simulation traffic under
    every method (0 = perfectly bidirectional, 1 = every corridor one-way).

Outputs results/directionality.csv and the quiver figure.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import CACHE, MU, GAMMA, RESULTS, STEPS
from mapf import maps as mapmod
from mapf import guidance as guid

EXP = "exp4b"
CELLS = [("empty-32-32", 400), ("random-32-32-20", 200), ("room-32-32-4", 170)]
METHODS = ["unweighted", "lanes", "tolls", "tolls-online"]


def directionality(f, rev):
    f = np.asarray(f, float)
    tot = float((f + f[rev]).sum())
    return float(np.abs(f - f[rev]).sum() / max(tot, 1e-12))


def conditions():
    out = []
    for mp_, N in CELLS:
        for meth in METHODS:
            out.append(dict(map=mp_, N=N, method=meth, seed=0, steps=STEPS,
                            mu=MU, gamma=GAMMA, record_edges=True))
        # symmetric-init tolls: the ablation that should show NO lane structure
        out.append(dict(map=mp_, N=N, method="tolls", seed=0, steps=STEPS,
                        mu=MU, gamma=GAMMA, seed_lanes=False,
                        record_edges=True, label="tolls-sym"))
    return out


def collect():
    rows = []
    recs = {}
    for r in common.load_results(EXP):
        c = r["cond"]
        recs[(c["map"], c["N"], c.get("label", c["method"]))] = r
    for mp_, N in CELLS:
        m = mapmod.get_map(mp_)
        _, rev = mapmod.directed_edges(m)
        # model flow (zero simulation)
        f_model, info = guid.load_toll_flow(m, N, gamma=GAMMA, mu=MU,
                                            cache_dir=CACHE)
        rows.append(dict(map=mp_, N=N, source="model stationary candidate",
                         directionality=round(directionality(f_model, rev), 4),
                         throughput=""))
        try:
            f_sym, _ = guid.load_toll_flow(m, N, gamma=GAMMA, mu=MU,
                                           seed_lanes=False, cache_dir=CACHE)
            rows.append(dict(map=mp_, N=N, source="model flow (sym init)",
                             directionality=round(directionality(f_sym, rev), 4),
                             throughput=""))
        except Exception:
            pass
        for lab in METHODS + ["tolls-sym"]:
            r = recs.get((mp_, N, lab))
            if r is None or "edge_counts_file" not in r:
                continue
            z = np.load(os.path.join(common.RESULTS, r["edge_counts_file"]))
            ec = z["edge_counts"].astype(float)
            rows.append(dict(map=mp_, N=N, source=f"realised: {lab}",
                             directionality=round(directionality(ec, rev), 4),
                             throughput=round(r["throughput"], 3)))
    path = os.path.join(RESULTS, "directionality.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["map", "N", "source",
                                           "directionality", "throughput"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['map']:18s} N={r['N']:4d} {r['source']:26s} "
              f"Dir={r['directionality']:.3f}  tput={r['throughput']}")
    return rows


if __name__ == "__main__":
    common.precompute_tolls([(mp_, N) for mp_, N in CELLS], mu=MU, gamma=GAMMA)
    for mp_, N in CELLS:
        common.precompute_tolls([(mp_, N)], mu=MU, gamma=GAMMA, seed_lanes=False)
    common.run_pending(EXP, conditions(), max_minutes=600.0)
    collect()
