"""Headline experiment on the standard MAPF benchmark maps.

Replaces exp1 (procedural maps + livelocking PIBT). Uses:
  - movingai.com benchmark maps (Stern et al. 2019),
  - PIBT with the swap operation (Okumura IJCAI-23) + stall escape,
  - density-matched fleet sizes (~10% / 25% / 40% of free cells),
  - the congestion-flow model with head-on coupling AND vertex capacity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BENCH_GRID, SEEDS, STEPS, MU, GAMMA, run_pending,
                    precompute_tolls)

EXP = "exp1b"
METHODS = ["unweighted", "lanes", "tolls", "tolls-online"]


def conditions():
    out = []
    for mp_, (_, Ns) in BENCH_GRID.items():
        for N in Ns:
            for meth in METHODS:
                for s in SEEDS:
                    out.append(dict(map=mp_, N=N, method=meth, seed=s,
                                    steps=STEPS, mu=MU, gamma=GAMMA))
    return out


if __name__ == "__main__":
    print("precomputing toll solves (serial, warms the cache)...")
    precompute_tolls([(mp_, N) for mp_, (_, Ns) in BENCH_GRID.items() for N in Ns],
                     mu=MU, gamma=GAMMA)
    run_pending(EXP, conditions(), max_minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 600.0)
