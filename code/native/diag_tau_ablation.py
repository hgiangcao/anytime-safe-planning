"""POST-HOC DIAGNOSTIC (not preregistered): does the stall escape explain the native gap?

The native planner check finds that on `room-32-32-4` the toll field is worse than no guidance
under upstream LaCAM3's PIBT, while it is better under this repository's Python PIBT.  The most
obvious structural difference between the two planners is this repository's `tau` stall escape:
an agent that has been blocked for more than `tau` steps demotes "wait" to its last resort.
Upstream PIBT has no equivalent.

This script switches that escape off in the PYTHON arm (`tau=0`) and re-measures the same cells,
so the explanation can be tested rather than asserted.  It is explicitly post-hoc: no criterion
was registered for it and it is reported as a mechanism probe, not as evidence for the method.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PROJ = os.path.dirname(CODE)
sys.path.insert(0, CODE)
import common                                                          # noqa: E402

OUT = os.path.join(PROJ, "results", "native_check", "diag_tau_ablation.json")
MAP = "room-32-32-4"
NS = [170, 270]
METHODS = ["unweighted", "tolls"]
TAUS = [common.TAU, 0]
SEEDS = list(range(5))


def one(arg):
    N, method, tau, seed = arg
    cond = dict(map=MAP, N=N, method=method, seed=seed, steps=common.STEPS,
                mu=common.MU, gamma=common.GAMMA, tau=tau)
    r = common.run_condition(cond)
    return dict(N=N, method=method, tau=tau, seed=seed, throughput=r["throughput"])


def main():
    jobs = [(N, m, t, s) for N in NS for m in METHODS for t in TAUS for s in SEEDS]
    with mp.Pool(int(os.environ.get("NC_PROCS", "4"))) as pool:
        rows = pool.map(one, jobs)
    summary = []
    for N in NS:
        for tau in TAUS:
            g = {m: np.array([r["throughput"] for r in rows
                              if r["N"] == N and r["method"] == m and r["tau"] == tau])
                 for m in METHODS}
            summary.append(dict(map=MAP, N=N, tau=tau,
                                unweighted=float(g["unweighted"].mean()),
                                tolls=float(g["tolls"].mean()),
                                d_TU=float(g["tolls"].mean() - g["unweighted"].mean()),
                                seeds=len(SEEDS)))
    d = dict(note=("post-hoc mechanism probe, not preregistered; python arm only, "
                   "tau=0 removes the stall escape that upstream PIBT does not have"),
             seeds=len(SEEDS), steps=common.STEPS, summary=summary, rows=rows)
    with open(OUT, "w") as fh:
        json.dump(d, fh, indent=1)
    for s in summary:
        print(f"  N={s['N']:<4d} tau={s['tau']:<3d} unweighted {s['unweighted']:6.3f}  "
              f"tolls {s['tolls']:6.3f}  d_TU {s['d_TU']:+6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
