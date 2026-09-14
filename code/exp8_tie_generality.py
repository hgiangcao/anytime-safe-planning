"""Exp 8 - is the tie-destruction failure specific to derived tolls?

Sec. IV-E argues that the failure is a property of *any* real-valued guidance
fed to a rule-based planner, not of our construction. The test: sweep the tie
tolerance for the handcrafted crisscross-lane baseline as well, and for a
random-but-fixed weight field. If the effect is general, all real-valued
guidance should collapse at eps=0 on obstacle maps and recover at eps=1.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import GAMMA, MU, RESULTS, SEEDS, STEPS

EXP = "exp8"
CELLS = [("random-32-32-20", 200), ("room-32-32-4", 170),
         ("warehouse-10-20-10-2-1", 800), ("empty-32-32", 400)]
EPS = [0.0, 1.0]
METHODS = ["lanes", "tolls", "randw"]


def conditions():
    out = []
    for mp_, N in CELLS:
        for meth in METHODS:
            for e in EPS:
                for s in SEEDS[:5]:
                    out.append(dict(map=mp_, N=N, method=meth, seed=s,
                                    steps=STEPS, mu=MU, gamma=GAMMA,
                                    eps_tie=e, label=f"{meth}-eps{e}"))
    return out


def collect():
    rows = {}
    for r in common.load_results(EXP):
        c = r["cond"]
        rows.setdefault((c["map"], c["N"], c["label"]), []).append(r["throughput"])
    base = {}
    for r in common.load_results("exp1b"):
        c = r["cond"]
        if c["method"] == "unweighted":
            base.setdefault((c["map"], c["N"]), []).append(r["throughput"])
    out = []
    print(f'{"map":24s} {"N":>5s} {"guidance":9s} {"unw":>6s} {"eps=0":>7s} '
          f'{"eps=1":>7s} {"r(0)":>6s} {"r(1)":>6s}')
    for mp_, N in CELLS:
        u = float(np.mean(base.get((mp_, N), [np.nan])))
        for meth in METHODS:
            a = rows.get((mp_, N, f"{meth}-eps0.0"))
            b = rows.get((mp_, N, f"{meth}-eps1.0"))
            if not a or not b:
                continue
            ma, mb = float(np.mean(a)), float(np.mean(b))
            print(f"{mp_:24s} {N:5d} {meth:9s} {u:6.2f} {ma:7.2f} {mb:7.2f} "
                  f"{ma / u:6.2f} {mb / u:6.2f}")
            out.append(dict(map=mp_, N=N, guidance=meth, unweighted=round(u, 3),
                            eps0=round(ma, 3), eps1=round(mb, 3),
                            ratio_eps0=round(ma / u, 3), ratio_eps1=round(mb / u, 3)))
    with open(os.path.join(RESULTS, "tie_generality.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/tie_generality.json")
    return out


if __name__ == "__main__":
    common.precompute_tolls(CELLS, mu=MU, gamma=GAMMA)
    common.run_pending(EXP, conditions(), max_minutes=600.0)
    collect()
