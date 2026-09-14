#!/usr/bin/env python
"""Why does the native planner reverse the obstacle-map ordering? (round-4 item 17.P1.1)

Guidance reaches a native PIBT only through the integer distance table, as
``round(dist_w(u->goal)/eps_tie)``.  eps_tie is therefore the RESOLUTION at which the planner can
see the guidance: eps_tie = 1 keeps only coarse differences, eps_tie = 0.01 follows the optimized
field almost exactly.  The locked native-check protocol already ran both for the tolls arm on all
fifteen benchmark cells (unweighted needs no ablation -- it carries no guidance to quantize).

That sweep turns an unexplained disagreement into a mechanism.  If the toll advantage were real and
the native reversal an artifact of coarse binning, then following the guidance MORE precisely would
recover it.

Reads only frozen run files; writes results/checks/tie_resolution.json.  No new runs.

    python code/analyze_tie_resolution.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "native_check", "runs_bench")
OUT = os.path.join(ROOT, "results", "checks", "tie_resolution.json")
OPEN_MAP = "empty-32-32"


def load():
    acc = defaultdict(dict)
    for f in glob.glob(os.path.join(RUNS, "*.json")):
        b = os.path.basename(f)[:-5]
        m = re.match(r"(.+?)__(\w+)__N(\d+)__s(\d+)__eps([\d.]+)$", b)
        if not m:
            continue
        mp, meth, N, s, eps = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)
        d = json.load(open(f))
        t = d.get("throughput", d.get("tasks_per_step", d.get("T")))
        acc[(mp, N, meth, eps)][s] = float(t)
    return acc


def paired(a: dict, b: dict):
    """Paired-by-seed difference a - b over the seeds both arms share."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {"n_seeds": n, "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)) if n > 1 else None, "se": float(se),
            "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "n_negative": int((d < 0).sum())}


def main() -> None:
    acc = load()
    rows = []
    for (mp, N, meth, eps) in sorted(acc):
        if meth != "tolls" or eps != "1":
            continue
        u = acc.get((mp, N, "unweighted", "1"))
        t1, t2 = acc.get((mp, N, "tolls", "1")), acc.get((mp, N, "tolls", "0.01"))
        if not (u and t1 and t2):
            continue
        rows.append({"map": mp, "N": N, "open_map": mp == OPEN_MAP,
                     "d_TU_coarse": paired(t1, u), "d_TU_fine": paired(t2, u),
                     "d_fine_minus_coarse": paired(t2, t1)})

    def tally(sel):
        rs = [r for r in rows if sel(r)]
        eff = [r["d_fine_minus_coarse"]["mean"] for r in rs]
        # Exact additive split of the advantage the manuscript reports at the deployed bin width:
        #     d_TU(coarse)  =  d_TU(fine)               +  (coarse - fine)
        #                      model-level term            implementation-local tie-bin term
        # The first is what the fluid field is worth when the planner follows it; the second is what
        # quantizing a noninteger distance table is worth, and a random noninteger field earns it too.
        mdl = [r["d_TU_fine"]["mean"] for r in rs]
        bins = [-e for e in eff]
        return {"n_cells": len(rs), "n_fine_worse": int(sum(1 for e in eff if e < 0)),
                "mean_effect": float(np.mean(eff)) if eff else None,
                "min_effect": float(min(eff)) if eff else None,
                "max_effect": float(max(eff)) if eff else None,
                "n_ci_excludes_zero": int(sum(
                    1 for r in rs if r["d_fine_minus_coarse"]["ci95"][1] < 0
                    or r["d_fine_minus_coarse"]["ci95"][0] > 0)),
                "n_model_term_negative": int(sum(1 for m in mdl if m < 0)),
                "n_model_term_positive": int(sum(1 for m in mdl if m > 0)),
                "mean_model_term": float(np.mean(mdl)) if mdl else None,
                "min_model_term": float(min(mdl)) if mdl else None,
                "max_model_term": float(max(mdl)) if mdl else None,
                "mean_tiebin_term": float(np.mean(bins)) if bins else None,
                "min_tiebin_term": float(min(bins)) if bins else None,
                "max_tiebin_term": float(max(bins)) if bins else None}

    out = {"question": "why does the native arm reverse the obstacle-map toll ordering?",
           "decomposition": ("d_TU(coarse) = d_TU(fine) + (coarse - fine); the first term is the "
                             "model-level fluid/discrete mismatch, the second the implementation-"
                             "local tie-binning effect any noninteger field incurs"),
           "mechanism_tested": ("eps_tie is the resolution at which a native PIBT sees the guidance "
                                "(round(dist_w/eps_tie)); if coarse binning caused the reversal, "
                                "following the guidance more precisely would recover the advantage"),
           "source": "results/native_check/runs_bench (frozen, locked protocol); no new runs",
           "eps_coarse": 1.0, "eps_fine": 0.01,
           "all_cells": tally(lambda r: True),
           "open_map_only": tally(lambda r: r["open_map"]),
           "obstacle_maps_only": tally(lambda r: not r["open_map"]),
           "cells": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{'map':24s} {'N':>5s} {'coarse':>8s} {'fine':>8s} {'fine-coarse':>12s}  95% CI")
    for r in rows:
        e = r["d_fine_minus_coarse"]
        print(f"{r['map']:24s} {r['N']:5d} {r['d_TU_coarse']['mean']:+8.3f} "
              f"{r['d_TU_fine']['mean']:+8.3f} {e['mean']:+12.3f}  "
              f"[{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]")
    for k in ("all_cells", "open_map_only", "obstacle_maps_only"):
        t = out[k]
        print(f"\n{k}: {t['n_fine_worse']}/{t['n_cells']} worse under finer guidance; "
              f"mean {t['mean_effect']:+.3f} (range {t['min_effect']:+.3f}..{t['max_effect']:+.3f}); "
              f"{t['n_ci_excludes_zero']}/{t['n_cells']} CIs exclude zero")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
