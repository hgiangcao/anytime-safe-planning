"""exp9: (a) a rank-wall check plus a Lagrangian/grid allocation heuristic,
(b) the lookahead-axis wall, and (c) a MATCHED-CALIBRATION long-horizon arm.

Why this experiment exists.  Table II compares the CS with a fixed-T union bound
at T = 120 windows calibrated from n = 2094 scores, i.e. exactly where Prop. 1
says no episode-level scheme can certify anything.  That comparison is a
consequence of the proposition, not independent evidence for it, and it is not
the comparison a practitioner faces once they have read the proposition.  This
experiment supplies the two things that are missing.

(a) Prop. 1 as a prediction.  We check analytically, over a grid of
    (n, delta, T), that the largest certifiable T is exactly
    floor(delta(n+1)) whatever the allocation.  A separate Lagrangian/grid
    heuristic (`csq.allocation_grid_heuristic`) illustrates attainable radius
    reductions on heterogeneous windows.  Because empirical quantile curves are
    nonconvex step functions, these reductions are not claimed globally optimal.

(b) The lookahead axis.  Allocating alpha across the H lookaheads inside a
    window -- the axis where optimised allocation genuinely pays -- multiplies
    the data requirement by H, moving the wall DOWN to delta(n+1)/H.

(c) Matched calibration.  We give the union bound the calibration set Prop. 1
    says it needs (n >= T/delta - 1) and rerun the 120-window deployment.  Every
    method then certifies every window and the comparison becomes one of
    conservatism and throughput rather than of existence -- the honest version of
    the headline.

Run:  .venv/bin/python code/exp9_matched.py --minutes 25 --workers 4
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
from csq import (allocation_grid_heuristic, cert_horizon_fixed_T,  # noqa: E402
                 cert_horizon_lookahead,
                 split_cp_quantile)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
MODELS = os.path.join(ROOT, "results", "models")
CACHE = os.path.join(ROOT, "results", "cache")
MATCH = os.path.join(ROOT, "results", "matched")
OUT = os.path.join(ROOT, "results", "exp9_matched.json")

W_LONG = 120
N_EP = 30
METHODS = ["cs", "union", "aucb", "aci_ep", "uncal"]
DELTA = 0.05


# ------------------------------------------------------------------ (a) Prop. 1
def prop1_test():
    rows = []
    pools = {"sf": np.sort(np.load(os.path.join(DATA, "cal_sf.npy"))),
             "orca": np.sort(np.load(os.path.join(DATA, "cal_orca.npy"))),
             "real_univ": np.sort(np.load(os.path.join(DATA, "cal_real_univ.npy")))}
    for name, s in pools.items():
        n = len(s)
        for delta in (0.05, 0.10):
            pred = cert_horizon_fixed_T(n, delta)
            sl = s.tolist()          # hoist the O(n) conversion out of the loop
            q = lambda w, lev, sl=sl: split_cp_quantile(sl, lev)
            found = None
            for T in range(max(1, pred - 3), pred + 4):
                if not allocation_grid_heuristic(q, T, delta, n)[2]:
                    found = T - 1
                    break
            rows.append({"pool": name, "n": n, "delta": delta,
                         "predicted_max_T": pred, "heuristic_feasible_max_T": found,
                         "match": found == pred})
            print(f"  {name:10s} n={n:5d} delta={delta}: predicted {pred},"
                  f" grid-heuristic feasibility {found}"
                  f"  {'MATCH' if found == pred else 'MISMATCH'}")
    # Descriptive heuristic reductions; not global optima for stepwise quantiles.
    rng = np.random.default_rng(7)
    n = 2094
    het = []
    for T in (20, 60, 104, 105):
        pl = [np.sort(rng.gamma(9.0, 0.1, n)) if w % 2 == 0
              else np.sort(rng.pareto(1.6, n) * 0.3 + 0.4) for w in range(T)]
        pll = [x.tolist() for x in pl]
        qf = lambda w, lev, pll=pll: split_cp_quantile(pll[w], lev)
        eps, r, feas = allocation_grid_heuristic(qf, T, 0.05, n)
        u = sum(qf(w, 1 - 0.05 / T) for w in range(T)) if feas else None
        het.append({"T": T, "feasible": feas,
                    "radius_gain_pct": None if not feas else 100 * (1 - r.sum() / u)})
        print(f"  heterogeneous T={T:4d}: feasible={feas}"
              + ("" if not feas else f" radius gain {100*(1-r.sum()/u):.1f}%"))
    return {
        "method_note": (
            "The wall is an analytic rank-feasibility check. Radius allocations "
            "use a Lagrangian/grid heuristic and are not certified global optima."
        ),
        "homogeneous_wall": rows,
        "heterogeneous_control": het,
    }


# --------------------------------------------------------------- (b) lookahead
def lookahead_wall():
    rows = []
    for name in ("sf", "orca"):
        n = len(np.load(os.path.join(DATA, f"cal_{name}.npy")))
        for delta in (0.05, 0.10):
            rows.append({"pool": name, "n": n, "delta": delta, "H": 8,
                         "wall_scalar": cert_horizon_fixed_T(n, delta),
                         "wall_per_lookahead": cert_horizon_lookahead(n, delta, 8)})
            print(f"  {name} n={n} delta={delta}: scalar wall"
                  f" {rows[-1]['wall_scalar']} -> per-lookahead wall"
                  f" {rows[-1]['wall_per_lookahead']} (factor 8)")
    return rows


# ----------------------------------------------------------- (c) matched pool
def build_matched_pool(target_n):
    import np2compat  # noqa: F401
    """Extra sf calibration scores so that delta(n+1) >= W_LONG at delta=0.05.
    Same generator, same predictor, fresh seeds -> a strictly larger sample of
    the SAME score distribution, which is the resource Prop. 1 identifies."""
    path = os.path.join(DATA, "cal_sf_matched.npy")
    if os.path.exists(path) and len(np.load(path)) >= target_n:
        return np.load(path)
    import torch
    torch.set_num_threads(1)
    from envloop import window_scores_from_rollout
    from exp0_data import gen_rollouts, T_ROLL
    from predictor import Predictor
    pred = Predictor(os.path.join(MODELS, "gru_sf.pt"))
    scores = list(np.load(os.path.join(DATA, "cal_sf.npy")))
    seed = 900000
    while len(scores) < target_n:
        for P, ev in gen_rollouts("sf", seed, 12, T_ROLL):
            scores.extend(window_scores_from_rollout(P, ev, pred))
        seed += 12
        print(f"    matched pool: {len(scores)}/{target_n}")
    np.save(path, np.array(scores))
    return np.array(scores)


def run_condition(method, deadline, n_target=N_EP):
    import np2compat  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from envloop import make_calibrator, run_episode
    from predictor import Predictor
    os.makedirs(MATCH, exist_ok=True)
    path = os.path.join(MATCH, f"matched_{method}_{DELTA}.json")
    ck = json.load(open(path)) if os.path.exists(path) else {"records": []}
    records = ck["records"]
    if len(records) >= n_target:
        return f"matched/{method}: complete ({len(records)})"
    warm = np.load(os.path.join(DATA, "cal_sf_matched.npy"))
    pred = Predictor(os.path.join(MODELS, "gru_sf.pt"))
    with open(os.path.join(DATA, "replay_long_sf.pkl"), "rb") as f:
        pool = pickle.load(f)
    while len(records) < n_target and time.time() < deadline:
        e = len(records)
        rng = np.random.default_rng(zlib.crc32(f"long_replay|{e}".encode()))
        P, ev = pool[e % len(pool)]
        cal = make_calibrator(method, DELTA, warm, cache_dir=CACHE, T=W_LONG)
        out = run_episode("replay", cal, method, pred, rng,
                          replay_item=(P, ev, pred), n_windows=W_LONG, patrol=True)
        out.pop("traj", None)
        records.append(out)
        json.dump({"records": records}, open(path, "w"))
    return f"matched/{method}: {len(records)}/{n_target}"


def summarise_matched():
    from exp3_planning import summarize
    out = {}
    for m in METHODS:
        p = os.path.join(MATCH, f"matched_{m}_{DELTA}.json")
        if not os.path.exists(p):
            continue
        recs = json.load(open(p))["records"]
        if recs:
            out[m] = summarize(recs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=N_EP)
    a = ap.parse_args()
    print("[exp9a] Prop. 1 rank wall plus allocation grid heuristic")
    p1 = prop1_test()
    print("[exp9b] the lookahead-axis wall")
    lw = lookahead_wall()
    need = int(np.ceil(W_LONG / DELTA)) - 1
    print(f"[exp9c] matched calibration pool: need n >= {need}")
    pool = build_matched_pool(need + 50)
    print(f"    matched pool n = {len(pool)}; delta(n+1) ="
          f" {cert_horizon_fixed_T(len(pool), DELTA)} >= {W_LONG} windows")
    deadline = time.time() + a.minutes * 60
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_condition, METHODS, [deadline] * len(METHODS),
                        [a.episodes] * len(METHODS)):
            print("   ", r)
    summ = summarise_matched()
    for m, s in summ.items():
        print(f"  {m:8s} cert={s['cert_window_frac']:.2f} wviol={s['window_viol_rate']}"
              f" r={s['mean_radius']} goals={s['mean_goals']:.1f}"
              f" coll/1k={1000*s['coll_step_frac']:.1f} fb={s['fallback_step_frac']:.2f}")
    json.dump({
        "artifact_schema_version": 2,
        "allocation_method": (
            "Lagrangian/grid heuristic, not a certified global optimum for "
            "nonconvex empirical-quantile steps"
        ),
        "rank_wall_status": (
            "analytic feasibility result independent of the allocation heuristic"
        ),
        "planning_execution_contract": "legacy_fresh_prediction_recentered_v0",
        "prop1_test": p1,
        "lookahead_wall": lw,
        "matched_pool_n": int(len(pool)),
        "matched_wall": cert_horizon_fixed_T(len(pool), DELTA),
        "matched": summ,
    }, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
