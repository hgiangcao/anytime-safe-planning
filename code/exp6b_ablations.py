"""Exp 6b - ablations of the model and the planner.

Model: head-on coupling gamma, vertex capacity mu, BPR exponent beta,
symmetry-breaking Frank-Wolfe initialisation, online re-solve period.
Planner: the swap operation on/off, which is what makes the obstacle maps
measurable at all.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEEDS, STEPS, MU, GAMMA, run_pending, precompute_tolls

EXP = "exp6b"
SDS = SEEDS[:5]
CELLS = [("empty-32-32", 400), ("random-32-32-20", 200)]
GAMMAS = [0.0, 1.0, 2.0, 4.0]
MUS = [0.0, 0.5, 1.0, 2.0]
BETAS = [1, 2, 4]
PERIODS = [100, 200, 400]
EPS = [0.0, 0.25, 0.5, 1.0, 2.0]


def conditions():
    out = []
    for mp_, N in CELLS:
        base = dict(map=mp_, N=N, steps=STEPS)
        for g in GAMMAS:
            for s in SDS:
                out.append(dict(base, method="tolls", seed=s, gamma=g, mu=MU,
                                label=f"gamma={g}"))
        for mu in MUS:
            for s in SDS:
                out.append(dict(base, method="tolls", seed=s, gamma=GAMMA, mu=mu,
                                label=f"mu={mu}"))
        for b in BETAS:
            for s in SDS:
                out.append(dict(base, method="tolls", seed=s, gamma=GAMMA, mu=MU,
                                beta=b, label=f"beta={b}"))
        for s in SDS:
            out.append(dict(base, method="tolls", seed=s, gamma=GAMMA, mu=MU,
                            seed_lanes=False, label="sym-init"))
        for p in PERIODS:
            for s in SDS:
                out.append(dict(base, method="tolls-online", seed=s,
                                gamma=GAMMA, mu=MU, period=p, label=f"period={p}"))
        for e in EPS:
            for s in SDS:
                out.append(dict(base, method="tolls", seed=s, gamma=GAMMA,
                                mu=MU, eps_tie=e, label=f"eps={e}"))
        for form in ("bpr", "queue"):
            for s in SDS:
                out.append(dict(base, method="tolls", seed=s, gamma=GAMMA,
                                mu=MU, form=form, label=f"form={form}"))
        # planner ablation: swap on/off, unweighted and tolled
        for sw in (True, False):
            for meth in ("unweighted", "tolls"):
                for s in SDS:
                    out.append(dict(base, method=meth, seed=s, gamma=GAMMA,
                                    mu=MU, swap=sw, tau=20 if sw else 0,
                                    label=f"{meth}-swap={sw}"))
    return out


if __name__ == "__main__":
    pre = []
    for mp_, N in CELLS:
        for g in GAMMAS:
            pre.append((mp_, N, dict(gamma=g, mu=MU)))
        for mu in MUS:
            pre.append((mp_, N, dict(gamma=GAMMA, mu=mu)))
        for b in BETAS:
            pre.append((mp_, N, dict(gamma=GAMMA, mu=MU, beta=b)))
        pre.append((mp_, N, dict(gamma=GAMMA, mu=MU, seed_lanes=False)))
        pre.append((mp_, N, dict(gamma=GAMMA, mu=MU, form="queue")))
    for mp_, N, kw in pre:
        precompute_tolls([(mp_, N)], **kw)
    run_pending(EXP, conditions(), max_minutes=600.0)
