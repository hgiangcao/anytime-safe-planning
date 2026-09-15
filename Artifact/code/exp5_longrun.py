"""exp5: LONG-HORIZON continuous deployment -- the regime the theory is about.

exp3 runs 8-window (17 s) episodes, which is far below the horizons at which the
fixed-T union bound is claimed to break down.  This experiment closes that gap by
running episodes of W_LONG = 120 scored windows (~4 min of continuous
navigation), which is beyond the certifiable horizon delta(n+1) = 105 of a
static pool of n = 2094 calibration scores at delta = 0.05.

Regimes (both calibrated from the SAME n = 2094 sf calibration pool, so the
Prop. 1 / Prop. 3 thresholds apply directly):
  long_replay : agent-only recorded rollouts replayed open loop (robot invisible).
                This preserves score generation but not iid windows: windows from
                one trajectory remain dependent, so its validity evidence is
                empirical and uncertainty must cluster whole trajectories.
  long_sf     : closed-loop social-force crowd, patrol task (the goal flips on
                arrival so the episode is not cut short) -> exchangeability
                stress test at long horizon.
  long_real   : long contiguous stretches of the five ETH/UCY scenes (60 windows
                ~ 2 min each), each calibrated from its own leave-one-scene-out
                pool (n ~ 600-730) -> the same certifiability question on real
                pedestrians, where delta(n+1) ~ 33 < 60 windows.

Every episode starts from a FRESH calibrator warm-started on the same
calibration pool.  This reset makes rank-based certifiability comparable across
records, but it does not make replay trajectories or their windows iid.  The
certifiable-horizon curves measure "what can be computed from n scores", while
dependence-aware inference clusters complete episodes and source scenes.

Run (repeat until 'ALL LONG CONDITIONS COMPLETE'):
    .venv/bin/python code/exp5_longrun.py --minutes 10 --workers 4
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
LONG = os.path.join(ROOT, "results", "long")
CACHE = os.path.join(ROOT, "results", "cache")

W_LONG = 120                      # scored windows per episode (~4 min)
W_REAL_LONG = 60                  # real scenes are shorter: 60 windows (~2 min)
N_EP = 30                         # episodes per condition (sim regimes)
REAL_SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
REGIMES = ["long_replay", "long_sf", "long_real"]
METHODS = ["cs", "auc", "aucb", "union", "aci", "aci_ep", "gauss", "uncal"]
DELTAS = [0.05, 0.1]

_cache = {}


def _load(regime):
    """(warm_scores, pool, predictor).  For 'long_real' the pool entries carry
    their own (predictor, warm pool) because each episode belongs to a different
    leave-one-scene-out fold; warm/predictor are then None."""
    if regime in _cache:
        return _cache[regime]
    import torch
    torch.set_num_threads(1)
    from predictor import Predictor
    if regime == "long_real":
        pool = []
        for sc in REAL_SCENES:
            pr = Predictor(os.path.join(MODELS, f"gru_real_{sc}.pt"))
            wm = np.load(os.path.join(DATA, f"cal_real_{sc}.npy"))
            with open(os.path.join(DATA, f"real_{sc}.pkl"), "rb") as f:
                for P, ev in pickle.load(f)["long"]:
                    pool.append((P, ev, pr, wm, sc))
        _cache[regime] = (None, pool, None)
        return _cache[regime]
    pred = Predictor(os.path.join(MODELS, "gru_sf.pt"))
    warm = np.load(os.path.join(DATA, "cal_sf.npy"))
    pool = None
    if regime == "long_replay":
        with open(os.path.join(DATA, "replay_long_sf.pkl"), "rb") as f:
            pool = [(P, ev, pred) for P, ev in pickle.load(f)]
    _cache[regime] = (warm, pool, pred)
    return _cache[regime]


def n_episodes(regime):
    if regime != "long_real":
        return N_EP
    return len(_load("long_real")[1])


def windows(regime):
    return W_REAL_LONG if regime == "long_real" else W_LONG


def per_window(records, w_max=W_LONG):
    """Certified fraction / violation rate / mean radius as a function of the
    window index within the episode."""
    cert = np.zeros(w_max)
    viol = np.zeros(w_max)
    seen = np.zeros(w_max)
    rsum = np.zeros(w_max)
    for r in records:
        for i, (s, rad) in enumerate(r["windows"][:w_max]):
            seen[i] += 1
            if rad is not None:
                cert[i] += 1
                rsum[i] += rad
                viol[i] += float(s > rad)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "n_seen": seen.tolist(),
            "cert_frac": np.where(seen > 0, cert / np.maximum(seen, 1), np.nan).tolist(),
            "viol_rate": np.where(cert > 0, viol / np.maximum(cert, 1), np.nan).tolist(),
            "mean_radius": np.where(cert > 0, rsum / np.maximum(cert, 1), np.nan).tolist(),
        }


def summarize(records, w_max=W_LONG):
    from exp3_planning import summarize as base_summarize
    s = base_summarize(records)
    n_first_uncert = []
    for r in records:
        idx = [i for i, (_, rad) in enumerate(r["windows"]) if rad is None]
        n_first_uncert.append(idx[0] + 1 if idx else len(r["windows"]) + 1)
    s["first_uncertified_window_median"] = float(np.median(n_first_uncert))
    s["mean_goals_per_episode"] = float(np.mean([r["goals"] for r in records]))
    s["per_window"] = per_window(records, w_max)
    return s


def run_condition(regime, method, delta, deadline):
    import torch
    torch.set_num_threads(1)
    from envloop import make_calibrator, run_episode
    path = os.path.join(LONG, f"{regime}_{method}_{delta}.json")
    ck = json.load(open(path)) if os.path.exists(path) else {"records": []}
    records = ck["records"]
    n_target, W = n_episodes(regime), windows(regime)
    if len(records) >= n_target:
        return f"{regime}/{method}/{delta}: complete ({len(records)})"
    warm, pool, predictor = _load(regime)
    sim_regime = "sf" if regime == "long_sf" else "replay"
    t0 = time.time()
    while len(records) < n_target:
        if time.time() > deadline:
            break
        e = len(records)
        rng = np.random.default_rng(zlib.crc32(f"long|{regime}|{e}".encode()))
        if regime == "long_real":
            P, ev, pr, wm, _sc = pool[e % len(pool)]
            item, w_use, pred_use = (P, ev, pr), wm, pr
        else:
            item = pool[e % len(pool)] if pool is not None else None
            w_use, pred_use = warm, predictor
        # fresh calibrator per episode: independent replicates of one deployment
        cal = make_calibrator(method, delta, w_use, cache_dir=CACHE, T=W)
        m = run_episode(sim_regime, cal, method, pred_use, rng, replay_item=item,
                        n_windows=W, patrol=True)
        m.pop("traj", None)
        records.append(m)
        if len(records) % 5 == 0 or len(records) == n_target:
            with open(path, "w") as f:
                json.dump({"records": records, "summary": summarize(records, W)}, f)
    with open(path, "w") as f:
        json.dump({"records": records, "summary": summarize(records, W)}, f)
    st = "complete" if len(records) >= n_target else "partial"
    return (f"{regime}/{method}/{delta}: {st} ({len(records)}/{n_target}, "
            f"{time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(LONG, exist_ok=True)
    deadline = time.time() + args.minutes * 60
    conds = []
    for reg in REGIMES:
        for d in DELTAS:
            for meth in METHODS:
                p = os.path.join(LONG, f"{reg}_{meth}_{d}.json")
                if (os.path.exists(p)
                        and len(json.load(open(p))["records"]) >= n_episodes(reg)):
                    continue
                conds.append((reg, meth, d))
    if not conds:
        print("ALL LONG CONDITIONS COMPLETE")
        return
    print(f"{len(conds)} pending long conditions; deadline in {args.minutes} min")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_condition, r, m, d, deadline) for r, m, d in conds]
        for f in futs:
            print(f.result())


if __name__ == "__main__":
    main()
