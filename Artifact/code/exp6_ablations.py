"""exp6: sensitivity of the calibration conclusions to the predictor and to the
planner -- the two "is this an artefact of your stack?" questions.

(a) Predictor: constant-velocity only vs. constant-velocity + GRU residual.  The
    predictor changes the score distribution (and hence every radius), so if the
    calibration story were predictor-specific it would show up here.
(b) Planner: the sampling grid is widened from 50 to 162 candidate accelerations
    (32 directions x 5 magnitudes + coast + brake).  A denser optimiser reduces
    spurious infeasibility, so if the ranking were an artefact of a weak planner
    it would move.
Both are run on the closed-loop social-force regime, delta = 0.05, 120 episodes,
with the same paired seeds as exp3.

Run: .venv/bin/python code/exp6_ablations.py   (~4 min)
"""
from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa: F401  (numpy>=2 pickles under numpy 1.x)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
MODELS = os.path.join(ROOT, "results", "models")
CACHE = os.path.join(ROOT, "results", "cache")
OUT = os.path.join(ROOT, "results", "exp6_ablations.json")

N_EP = 120
DELTA = 0.05
METHODS = ["cs", "auc", "aucb", "union", "aci", "gauss"]


def run(tag, predictor, n_dir=16, mags=(0.7, 1.4, 2.0)):
    import torch
    torch.set_num_threads(2)
    import planner
    from envloop import make_calibrator, run_episode
    from exp3_planning import summarize
    n_cand = planner.configure(n_dir, mags)
    warm = np.load(os.path.join(DATA, "cal_sf.npy"))
    out = {"n_candidates": n_cand, "n_episodes": N_EP}
    for m in METHODS:
        cal = make_calibrator(m, DELTA, warm, cache_dir=CACHE)
        recs = []
        for e in range(N_EP):
            rng = np.random.default_rng(zlib.crc32(f"sf|{e}".encode()))
            recs.append(run_episode("sf", cal, m, predictor, rng))
        s = summarize(recs)
        out[m] = {k: s[k] for k in ("cert_window_frac", "window_viol_rate",
                                    "episode_viol_rate", "success_rate",
                                    "collision_rate", "fallback_step_frac",
                                    "mean_radius")}
        print(f"  {tag}/{m}: Wviol {s['window_viol_rate']:.3f} succ "
              f"{s['success_rate']:.2f} r {s['mean_radius']:.2f} "
              f"coll {s['collision_rate']:.3f}")
    planner.configure()  # restore
    return out


if __name__ == "__main__":
    from predictor import Predictor
    gru = Predictor(os.path.join(MODELS, "gru_sf.pt"))
    cv = Predictor(cv_only=True)
    res = {"config": {"regime": "sf", "delta": DELTA, "n_episodes": N_EP,
                      "methods": METHODS}}
    print("[a] predictor ablation")
    res["pred_gru_default"] = run("gru", gru)
    res["pred_cv_only"] = run("cv", cv)
    print("[b] planner ablation (denser sampling grid)")
    res["planner_dense"] = run("dense", gru, n_dir=32,
                               mags=(0.4, 0.8, 1.2, 1.6, 2.0))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("exp6 complete ->", OUT)
