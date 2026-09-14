"""Exp 3b - transfer of derived weights.

(a) Agent-count transfer: weights built once at a reference fleet size and
    evaluated across the whole range, vs re-solving per N and vs frozen search
    weights.
(b) Task-distribution shift: weights built under uniform demand, evaluated
    under a shifted (hot-spot) task distribution. This is the regime the
    online variant exists for, and it was listed as future work previously.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEEDS, STEPS, MU, GAMMA, run_pending, precompute_tolls

EXP_N = "exp3b_agentcount"
EXP_D = "exp3b_demandshift"
MAP = "empty-32-32"
N_REF = 400
N_GRID = [100, 200, 400, 600, 800]
HOT = [0.0, 0.4, 0.7, 0.9]
N_HOT = 250


def cond_agentcount():
    out = []
    for N in N_GRID:
        for s in SEEDS:
            out.append(dict(map=MAP, N=N, method="unweighted", seed=s, steps=STEPS))
            out.append(dict(map=MAP, N=N, method="lanes", seed=s, steps=STEPS))
            # frozen at the reference fleet size
            out.append(dict(map=MAP, N=N, method="tolls", seed=s, steps=STEPS,
                            N_solve=N_REF, mu=MU, gamma=GAMMA, label="tolls-frozen"))
            # re-solved per N
            out.append(dict(map=MAP, N=N, method="tolls", seed=s, steps=STEPS,
                            mu=MU, gamma=GAMMA, label="tolls-resolved"))
            out.append(dict(map=MAP, N=N, method="tolls-online", seed=s,
                            steps=STEPS, mu=MU, gamma=GAMMA))
    return out


def cond_demandshift():
    out = []
    for h in HOT:
        for s in SEEDS:
            for meth in ["unweighted", "lanes", "tolls", "tolls-online"]:
                out.append(dict(map=MAP, N=N_HOT, method=meth, seed=s,
                                steps=STEPS, mu=MU, gamma=GAMMA, hotspot=h))
    return out


if __name__ == "__main__":
    precompute_tolls([(MAP, N) for N in set(N_GRID + [N_REF, N_HOT])],
                     mu=MU, gamma=GAMMA)
    run_pending(EXP_N, cond_agentcount(), max_minutes=600.0)
    run_pending(EXP_D, cond_demandshift(), max_minutes=600.0)
