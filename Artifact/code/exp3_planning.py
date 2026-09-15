"""exp3: main closed-loop / replay planning experiment.

Conditions = regime x method x delta:
  regimes : replay (held-out recorded sim rollouts, robot invisible -> clean
            exchangeable regime), sf + orca (closed loop, robot visible ->
            interaction-induced distribution shift, honest stress test),
            eth (real BIWI-ETH pedestrians replayed with the sf predictor and
            only n=72 calibration scores; an extreme small-n pilot), and
            real_{eth,hotel,univ,zara1,zara2} (leave-one-scene-out ETH/UCY:
            predictor + calibration pool from the other four scenes, deployment
            on the held-out scene)
  methods : cs (ours, time-uniform quantile CS), auc (ours, anytime-union
            episode guarantee, delta/(w(w+1)) spending), aucb (ours,
            budget-aware anytime-union: same guarantee; lambda=0 implements the
            exact harmonic-spending cutoff), union (Lindemann-style fixed-T
            union-bound CP), aci
            (adaptive conformal, window-level target), aci_ep (adaptive
            conformal, episode-targeted alpha = delta/W -- the fair
            episode-level ACI arm), gauss (Rayleigh chance constraint),
            uncal (raw predictions)
  delta   : 0.05, 0.1
200 episodes per condition (11 for eth); episode seeds shared across methods and
deltas (paired crowds).  Checkpoints per condition in results/main/*.json; safe
to re-run until everything reports complete.  Accreting calibrators are rebuilt
exactly from the recorded score/err stream on resume.

Run (repeat until 'ALL CONDITIONS COMPLETE'):
    .venv/bin/python code/exp3_planning.py --minutes 8.5 --workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa: F401  (numpy>=2 pickles under numpy 1.x)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
MODELS = os.path.join(ROOT, "results", "models")
MAIN = os.path.join(ROOT, "results", "main")
CACHE = os.path.join(ROOT, "results", "cache")

# real_* = leave-one-scene-out ETH/UCY folds: the predictor and the calibration
# pool for fold <scene> are built on the OTHER four scenes, and deployment
# replays <scene> (real pedestrians, real cross-scene distribution shift).
REAL_SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
REGIMES = ["replay", "sf", "orca", "eth"] + [f"real_{s}" for s in REAL_SCENES]
METHODS = ["cs", "auc", "aucb", "union", "aci", "aci_ep", "gauss", "uncal"]
DELTAS = [0.05, 0.1]
N_EP = {"replay": 200, "sf": 200, "orca": 200, "eth": 11}


def _n_real(scene):
    with open(os.path.join(DATA, f"real_{scene}.pkl"), "rb") as f:
        return len(pickle.load(f)["short"])


for _s in REAL_SCENES:
    try:
        N_EP[f"real_{_s}"] = _n_real(_s)
    except FileNotFoundError:
        REGIMES = [r for r in REGIMES if r != f"real_{_s}"]

_cache = {}


def _load(regime):
    """(warm_scores, replay_pool_or_None, predictor_fn) for a regime."""
    if regime in _cache:
        return _cache[regime]
    import torch
    torch.set_num_threads(1)
    from predictor import Predictor
    preds = {k: Predictor(os.path.join(MODELS, f"gru_{k}.pt")) for k in ("sf", "orca")}
    if regime in ("sf", "orca"):
        warm = np.load(os.path.join(DATA, f"cal_{regime}.npy"))
        out = (warm, None, preds[regime])
    elif regime == "replay":
        warm = np.concatenate([np.load(os.path.join(DATA, "cal_sf.npy")),
                               np.load(os.path.join(DATA, "cal_orca.npy"))])
        pool = []
        for k in ("sf", "orca"):
            with open(os.path.join(DATA, f"replay_{k}.pkl"), "rb") as f:
                pool.extend((P, ev, preds[k]) for P, ev in pickle.load(f))
        rng = np.random.default_rng(555)
        pool = [pool[i] for i in rng.permutation(len(pool))]
        out = (warm, pool, None)
    elif regime == "eth":
        warm = np.load(os.path.join(DATA, "cal_eth.npy"))
        with open(os.path.join(DATA, "replay_eth.pkl"), "rb") as f:
            pool = [(P, ev, preds["sf"]) for P, ev in pickle.load(f)]
        out = (warm, pool, None)
    elif regime.startswith("real_"):
        scene = regime[5:]
        warm = np.load(os.path.join(DATA, f"cal_real_{scene}.npy"))
        pr = Predictor(os.path.join(MODELS, f"gru_real_{scene}.pt"))
        with open(os.path.join(DATA, f"real_{scene}.pkl"), "rb") as f:
            pool = [(P, ev, pr) for P, ev in pickle.load(f)["short"]]
        out = (warm, pool, None)
    _cache[regime] = out
    return out


def _rebuild(method, delta, warm, records):
    from envloop import ACI_LIKE, EPISODIC, ONLINE, make_calibrator
    cal = make_calibrator(method, delta, warm, cache_dir=CACHE)
    for rec in records:
        if method in EPISODIC:
            cal.new_episode()
        for s, err in rec["cal_stream"]:
            if method in ACI_LIKE:
                cal.update_err(err)
            if ONLINE[method]:
                cal.add(s)
    return cal


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial rate (used for every rate we report)."""
    if n == 0:
        return (None, None)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def summarize(records):
    """Per-condition summary.

    Reporting conventions (tightened after the certifiability audit):
      * `cert_window_frac` = fraction of scored windows for which the calibrator
        returned a FINITE radius, i.e. for which a certificate exists at all.
      * `window_viol_rate` is computed over CERTIFIED windows only -- a window
        with r = +inf cannot be "violated", and counting it as a success would
        reward a calibrator for refusing to certify.  `window_viol_rate_all`
        keeps the old all-windows convention for comparison.
      * every rate carries a Wilson 95% interval.
    """
    tw = sum(r["n_windows"] for r in records)          # scored windows
    tu = sum(r["n_uncert"] for r in records)           # of which uncertified
    tc = tw - tu                                       # certified windows
    tv = sum(r["n_viol"] for r in records)             # violations (certified only)
    reached = [r for r in records if r["reached"]]
    steps_all = sum(r["steps"] for r in records)
    ne = len(records)
    n_ev = sum(bool(r["episode_viol"]) for r in records)
    n_coll = sum(bool(r["collision"]) for r in records)
    n_succ = len(reached)
    return {
        "n_episodes": ne,
        "total_windows": tw,
        "cert_window_frac": tc / max(tw, 1),
        "window_viol_rate": tv / max(tc, 1) if tc else None,
        "window_viol_ci": wilson(tv, tc) if tc else (None, None),
        "window_viol_rate_all": tv / max(tw, 1),
        "episode_viol_rate": n_ev / max(ne, 1),
        "episode_viol_ci": wilson(n_ev, ne),
        "uncert_window_frac": tu / max(tw, 1),
        "collision_rate": n_coll / max(ne, 1),
        "collision_ci": wilson(n_coll, ne),
        "coll_step_frac": sum(r["n_coll_steps"] for r in records) / max(steps_all, 1),
        "success_rate": n_succ / max(ne, 1),
        "success_ci": wilson(n_succ, ne),
        "timeout_rate": 1.0 - n_succ / max(ne, 1),
        "mean_task_time_s": float(np.mean([r["steps"] * 0.25 for r in reached])) if reached else None,
        "fallback_step_frac": sum(r["fallback_steps"] for r in records) / max(steps_all, 1),
        "mean_min_dist": float(np.mean([r["min_dist"] for r in records])),
        "mean_goals": float(np.mean([r["goals"] for r in records])),
        "mean_radius": float(np.mean([r["mean_radius"] for r in records if r["mean_radius"] is not None])) if any(r["mean_radius"] is not None for r in records) else None,
    }


def run_condition(regime, method, delta, deadline):
    import torch
    torch.set_num_threads(1)
    from envloop import run_episode
    path = os.path.join(MAIN, f"{regime}_{method}_{delta}.json")
    n_target = N_EP[regime]
    ck = json.load(open(path)) if os.path.exists(path) else {"records": []}
    records = ck["records"]
    if len(records) >= n_target:
        return f"{regime}/{method}/{delta}: complete ({len(records)})"
    warm, pool, predictor = _load(regime)
    cal = _rebuild(method, delta, warm, records)
    t_start = time.time()
    while len(records) < n_target:
        if time.time() > deadline:
            break
        e = len(records)
        import zlib
        seed = zlib.crc32(f"{regime}|{e}".encode())  # deterministic across runs
        rng = np.random.default_rng(seed)
        item = pool[e % len(pool)] if pool is not None else None
        rg = "replay" if regime.startswith("real_") else regime
        m = run_episode(rg, cal, method, predictor, rng, replay_item=item)
        # store the calibrator-relevant stream for exact resume
        m["cal_stream"] = [
            [float(s), 1.0 if (r is not None and s > r) else 0.0]
            for s, r in m["windows"]]  # r=None (uncertified) is not a violation
        records.append(m)
        if len(records) % 20 == 0 or len(records) == n_target:
            with open(path, "w") as f:
                json.dump({"records": records, "summary": summarize(records)}, f)
    with open(path, "w") as f:
        json.dump({"records": records, "summary": summarize(records)}, f)
    st = "complete" if len(records) >= n_target else "partial"
    return (f"{regime}/{method}/{delta}: {st} ({len(records)}/{n_target}, "
            f"{time.time()-t_start:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.5)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(MAIN, exist_ok=True)
    deadline = time.time() + args.minutes * 60
    conds = []
    for reg in REGIMES:
        for d in DELTAS:
            for meth in METHODS:
                path = os.path.join(MAIN, f"{reg}_{meth}_{d}.json")
                if os.path.exists(path):
                    ck = json.load(open(path))
                    if len(ck["records"]) >= N_EP[reg]:
                        continue
                conds.append((reg, meth, d))
    if not conds:
        print("ALL CONDITIONS COMPLETE")
        return
    print(f"{len(conds)} pending conditions; deadline in {args.minutes} min")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_condition, r, m, d, deadline) for r, m, d in conds]
        for f in futs:
            print(f.result())


if __name__ == "__main__":
    main()
