"""Exp 3 — transfer across agent count on empty-32x32.

Weights derived/optimized at N=400, evaluated at N in {100..800}:
  - tolls-adapt    : flow model re-solved at each N (ours, ~1 s/solve)
  - tolls-fixed400 : toll weights frozen from the N=400 solve
  - es-fixed400    : CMA-ES weights frozen (from exp2; skipped until exp2 done)
  - unweighted, lanes : references (N-independent by construction)
10 seeds x 1000 steps each.
"""
import os

import common

MAP = "empty-32x32"
NS = [100, 200, 300, 400, 500, 600, 700, 800]


def conditions(include_es):
    out = []
    for N in NS:
        for seed in common.SEEDS:
            out.append(dict(map=MAP, N=N, method="unweighted", seed=seed,
                            label="unweighted"))
            out.append(dict(map=MAP, N=N, method="lanes", seed=seed,
                            label="lanes"))
            out.append(dict(map=MAP, N=N, method="tolls", seed=seed,
                            label="tolls-adapt"))
            out.append(dict(map=MAP, N=N, method="tolls", seed=seed,
                            N_solve=400, label="tolls-fixed400"))
            if include_es:
                out.append(dict(map=MAP, N=N, method="es", seed=seed,
                                label="es-fixed400"))
    return out


if __name__ == "__main__":
    include_es = os.path.exists(common.es_best_path())
    if not include_es:
        print("NOTE: es_best.npz missing (run exp2 first); skipping ES rows")
    common.precompute_tolls([(MAP, N) for N in NS])
    common.run_pending("exp3", conditions(include_es))
