"""exp10: legacy harder-vehicle and shift-robust stress-test generator.

The checked-in ``results/vehicle`` and ``exp10_vehicle.json`` artifacts were
generated before ``window_anchor_v1``.  They recentered prediction tubes within
a score window, used unbounded Gaussian actuation noise, and therefore provide
neither a containment-to-collision certificate nor evidence for the revised
controller.  They remain only as provenance-tagged empirical stress results.
Any new invocation uses zero latency because the current controller rejects a
stale prediction unless a latency-aware score has first been calibrated.

Two objections that a simulation-only study must answer directly.

(a) VEHICLE.  Secs. V-B..V-E use a 2-D double integrator, which can stop and
    reverse instantly and therefore makes a conservative inflation radius
    unusually cheap.  Here the identical calibration problem is run on a
    differential-drive unicycle with non-negative speed, bounded yaw rate
    (2.5 rad/s), and 5% Gaussian actuation noise.  Because that noise has no
    deterministic tracking bound, outcomes are empirical and not certified.

(b) SHIFT.  `cs_r` uses an illustrative total budget rho = 12.  The theorem
    requires an externally justified conditional pathwise budget; the held-out
    marginal diagnostic in exp12 is too uncertain to supply one.

Run: .venv/bin/python code/exp10_vehicle.py --minutes 20 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa: F401  (numpy>=2 pickles under numpy 1.x)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
MODELS = os.path.join(ROOT, "results", "models")
CACHE = os.path.join(ROOT, "results", "cache")
VEH = os.path.join(ROOT, "results", "vehicle")
OUT = os.path.join(ROOT, "results", "exp10_vehicle.json")

REAL_SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
# (tag, regime, vehicle, methods, n_episodes)
# (tag, regime, vehicle, methods, n_episodes, deltas)
JOBS = [("uni_sf", "sf", "uni", ["cs", "aucb", "union", "aci", "gauss", "uncal"], 150, [0.05]),
        ("uni_replay", "replay", "uni", ["cs", "aucb", "union", "aci", "gauss", "uncal"], 150, [0.05]),
        ("rob_orca", "orca", "di", ["cs", "cs_r"], 200, [0.05, 0.10]),
        ("rob_real", "real", "di", ["cs", "cs_r"], 100, [0.05, 0.10])]
DELTA = 0.05
LATENCY, ACT_NOISE = 0, 0.05


def _load(regime):
    import np2compat  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from predictor import Predictor
    if regime in ("sf", "orca"):
        return np.load(os.path.join(DATA, f"cal_{regime}.npy")), None, \
            Predictor(os.path.join(MODELS, f"gru_{regime}.pt"))
    if regime == "replay":
        warm = np.concatenate([np.load(os.path.join(DATA, "cal_sf.npy")),
                               np.load(os.path.join(DATA, "cal_orca.npy"))])
        pool = []
        for k in ("sf", "orca"):
            pr = Predictor(os.path.join(MODELS, f"gru_{k}.pt"))
            with open(os.path.join(DATA, f"replay_{k}.pkl"), "rb") as f:
                pool.extend((P, ev, pr) for P, ev in pickle.load(f))
        rng = np.random.default_rng(555)
        return warm, [pool[i] for i in rng.permutation(len(pool))], None
    if regime == "real":                      # all five LOSO folds pooled
        pool = []
        for sc in REAL_SCENES:
            pr = Predictor(os.path.join(MODELS, f"gru_real_{sc}.pt"))
            wm = np.load(os.path.join(DATA, f"cal_real_{sc}.npy"))
            with open(os.path.join(DATA, f"real_{sc}.pkl"), "rb") as f:
                for P, ev in pickle.load(f)["short"]:
                    pool.append((P, ev, pr, wm))
        return None, pool, None
    raise ValueError(regime)


def run_condition(args):
    import np2compat  # noqa: F401
    tag, regime, vehicle, method, n_target, delta = args
    import torch
    torch.set_num_threads(1)
    from envloop import make_calibrator, run_episode
    os.makedirs(VEH, exist_ok=True)
    path = os.path.join(VEH, f"{tag}_{method}_{delta}.json")
    ck = json.load(open(path)) if os.path.exists(path) else {"records": []}
    records = ck["records"]
    if len(records) >= n_target:
        return f"{tag}/{method}: complete ({len(records)})"
    warm, pool, pred = _load(regime)
    lat = LATENCY if vehicle == "uni" else 0
    noi = ACT_NOISE if vehicle == "uni" else 0.0
    t0 = time.time()
    while len(records) < n_target and time.time() - t0 < 19 * 60:
        e = len(records)
        rng = np.random.default_rng(zlib.crc32(f"{tag}|{e}".encode()))
        if regime == "real":
            P, ev, pr, wm = pool[e % len(pool)]
            cal = make_calibrator(method, delta, wm, cache_dir=CACHE)
            item, prd = (P, ev, pr), pr
            rg = "replay"
        elif regime == "replay":
            P, ev, pr = pool[e % len(pool)]
            cal = make_calibrator(method, delta, warm, cache_dir=CACHE)
            item, prd, rg = (P, ev, pr), pr, "replay"
        else:
            cal = make_calibrator(method, delta, warm, cache_dir=CACHE)
            item, prd, rg = None, pred, regime
        out = run_episode(rg, cal, method, prd, rng, replay_item=item,
                          vehicle=vehicle, latency=lat, act_noise=noi)
        out.pop("traj", None)
        records.append(out)
        if len(records) % 25 == 0:
            json.dump({"records": records}, open(path, "w"))
    json.dump({"records": records}, open(path, "w"))
    return f"{tag}/{method}: {len(records)}/{n_target}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tasks = [(tag, rg, vh, m, ne, dl) for tag, rg, vh, ms, ne, dls in JOBS
             for m in ms for dl in dls]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_condition, tasks):
            print("   ", r)
    from exp3_planning import summarize
    out = {}
    for tag, rg, vh, ms, ne, dls in JOBS:
        for m in ms:
            for dl in dls:
                p = os.path.join(VEH, f"{tag}_{m}_{dl}.json")
                if not os.path.exists(p):
                    continue
                recs = json.load(open(p))["records"]
                if recs:
                    out[f"{tag}|{m}|{dl}"] = summarize(recs)
    for k, s in sorted(out.items()):
        print(f"  {k:20s} n={s['n_episodes']:4d} cert={s['cert_window_frac']:.2f}"
              f" wviol={s['window_viol_rate']} succ={s['success_rate']:.3f}"
              f" r={s['mean_radius']} coll/1k={1000*s['coll_step_frac']:.2f}"
              f" fb={s['fallback_step_frac']:.3f}")
    json.dump({"conditions": out, "latency_steps": LATENCY,
               "act_noise": ACT_NOISE, "rho_tot": 12.0}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
