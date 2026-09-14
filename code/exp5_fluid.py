"""Exp 5 — fluid-model predicted vs realized throughput.

The Little's-law-calibrated flow solve predicts a throughput lam (tasks/step)
for each (map, N). Compare against realized simulator throughput of the
toll-guided runs (exp1 grid + exp3 empty curve). Aggregation only — no new
simulation; writes results/fluid_model.csv.
"""
import csv
import os

import numpy as np

import common
from mapf import maps as mapmod
from mapf import guidance as guid

GRID = ([(mn, N) for mn in ["empty-32x32", "random-32x32", "room-32x32",
                            "warehouse-33x36"] for N in [100, 250, 400]] +
        [("empty-32x32", N) for N in [200, 300, 500, 600, 700, 800]])


def realized(rows, mname, N):
    # dedupe by seed: exp1 "tolls" and exp3 "tolls-adapt" are the identical
    # deterministic condition, so pooling both double-counted seeds
    # (verification fix; means/stds unchanged, n_seeds now honest).
    by_seed = {}
    for r in rows:
        c = r["cond"]
        if (c["map"] == mname and c["N"] == N and c["method"] == "tolls"
                and "N_solve" not in c
                and not any(k in c for k in ("alpha", "beta", "gamma"))):
            by_seed[c["seed"]] = r["throughput"]
    return list(by_seed.values())


if __name__ == "__main__":
    common.precompute_tolls(GRID)
    rows = common.load_results("exp1") + common.load_results("exp3")
    out = []
    for mname, N in GRID:
        m = mapmod.get_map(mname)
        _, info = guid.wardrop_tolls(m, N, cache_dir=common.CACHE)
        tp = realized(rows, mname, N)
        out.append(dict(map=mname, N=N, predicted_lam=info["lam"],
                        mean_latency_model=info["mean_latency"],
                        realized_mean=float(np.mean(tp)) if tp else None,
                        realized_std=float(np.std(tp)) if tp else None,
                        n_seeds=len(tp)))
        print(out[-1])
    path = os.path.join(common.RESULTS, "fluid_model.csv")
    with open(path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(out)
    print("wrote", path)
