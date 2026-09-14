"""Exp 4 — congestion heatmap data: per-edge traversal counts (1000 steps,
seed 0) for unweighted / lanes / tolls / es on empty-32x32 at N=400, and
unweighted / tolls on warehouse-33x36 at N=400. Edge counts saved to
results/cache/edges_*.npz; figures rendered by make_figures.py.
"""
import os

import common


def conditions(include_es):
    conds = []
    for method in ["unweighted", "lanes", "tolls"] + (["es"] if include_es else []):
        conds.append(dict(map="empty-32x32", N=400, method=method, seed=0,
                          record_edges=True))
    for method in ["unweighted", "tolls"]:
        conds.append(dict(map="warehouse-33x36", N=400, method=method, seed=0,
                          record_edges=True))
    return conds


if __name__ == "__main__":
    include_es = os.path.exists(common.es_best_path())
    common.precompute_tolls([("empty-32x32", 400), ("warehouse-33x36", 400)])
    common.run_pending("exp4", conditions(include_es))
