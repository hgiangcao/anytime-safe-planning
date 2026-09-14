"""Exp 7 - do derived guidance weights transfer to a different planner?

The paper's premise is that lifelong planners behave like selfish
shortest-path routers, which is a claim about a class. This experiment takes
the SAME toll weights (derived once, no re-tuning) and feeds them to windowed
prioritized planning (RHCR's low level: space-time A* + reservation table over
a bounded horizon) instead of PIBT.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SEEDS, STEPS, MU, GAMMA, run_pending, precompute_tolls)

EXP = "exp7"
# Prioritized planning is incomplete: measured against PIBT on the same cells,
# it deadlocks above roughly 25% density (empty-32-32@400: 0.2 vs 13.2
# tasks/step; room-32-32-4@170: 0.18 vs 3.04). The planner-transfer test is
# therefore run only where the second planner is itself in steady state --
# which is a limitation of the baseline planner, not of the guidance.
GRID = [("empty-32-32", [100, 250]),
        ("random-32-32-20", [100]),
        ("room-32-32-4", [70])]
PLANNER_KW = dict(window=8, period=4)
METHODS = ["unweighted", "lanes", "tolls", "tolls-online"]


def conditions():
    out = []
    for mp_, Ns in GRID:
        for N in Ns:
            for meth in METHODS:
                for s in SEEDS[:5]:
                    out.append(dict(map=mp_, N=N, method=meth, seed=s,
                                    steps=STEPS, mu=MU, gamma=GAMMA,
                                    planner="wpp", planner_kw=PLANNER_KW))
    return out


if __name__ == "__main__":
    precompute_tolls([(mp_, N) for mp_, Ns in GRID for N in Ns], mu=MU, gamma=GAMMA)
    run_pending(EXP, conditions(), max_minutes=600.0)
