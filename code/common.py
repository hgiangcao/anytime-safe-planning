"""Shared experiment infrastructure: checkpointed condition runner.

Every simulation run is one *condition* dict; its result is saved immediately
to results/raw/<exp>/<key>.json. Re-running any experiment script skips
already-completed conditions, so repeated invocations accumulate results
(designed around the 10-minute shell cap). Multiprocessing capped at 3
workers (3 projects share this machine).
"""
import json
import os
import sys
import time
import hashlib
import multiprocessing as mp

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(CODE)
RESULTS = os.path.join(PROJ, "results")
RAW = os.path.join(RESULTS, "raw")
CACHE = os.path.join(RESULTS, "cache")
FIGS = os.path.join(RESULTS, "figures")
for d in (RESULTS, RAW, CACHE, FIGS):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, CODE)

from mapf import maps as mapmod            # noqa: E402
from mapf import flow as flowmod           # noqa: E402
from mapf import guidance as guid          # noqa: E402
from mapf.pibt import dist_table           # noqa: E402
from mapf.sim import run_lifelong          # noqa: E402

N_WORKERS = 3
STEPS = 1000
SEEDS = list(range(10))
ONLINE_CFG = dict(period=200, window=6000, K=100000, fw_iters=30)

# Standard MAPF benchmark maps (movingai.com, Stern et al. 2019) with
# density-matched fleet sizes (~10% / 25% / 40% of free cells).
BENCH_GRID = {
    "empty-32-32":            (1024, [100, 250, 400]),
    "random-32-32-20":        (819,  [100, 200, 320]),
    "room-32-32-4":           (682,  [70, 170, 270]),
    "maze-32-32-4":           (790,  [80, 200, 310]),
    "warehouse-10-20-10-2-1": (5699, [400, 800, 1600]),
}
BENCH_MAPS = list(BENCH_GRID)
MU = 1.0                 # vertex-capacity coefficient (cell exclusion)
GAMMA = 2.0              # head-on coupling
TAU = 20                 # PIBT stall-escape threshold
# Tie-break tolerance for PIBT's candidate ordering, in free-flow timesteps.
# Unweighted PIBT compares INTEGER distances, so candidate moves tie constantly
# and PIBT's random tie-break makes the rule stochastic -- which is what lets a
# jammed configuration dissolve. Real-valued guidance weights destroy those
# ties, the rule becomes deterministic, and a jam with frozen priorities is an
# absorbing state. Comparing candidates at the resolution of one timestep --
# the resolution at which the planner actually acts -- restores them.
EPS_TIE = 1.0


# ---- checkpoint store -----------------------------------------------------
def cond_key(cond):
    s = json.dumps({k: v for k, v in sorted(cond.items())}, sort_keys=True)
    h = hashlib.md5(s.encode()).hexdigest()[:10]
    return f"{cond['map']}_{cond['method']}_N{cond['N']}_s{cond['seed']}_{h}"


def raw_dir(exp):
    d = os.path.join(RAW, exp)
    os.makedirs(d, exist_ok=True)
    return d


def is_done(exp, cond):
    return os.path.exists(os.path.join(raw_dir(exp), cond_key(cond) + ".json"))


def save_result(exp, cond, rec):
    rec = dict(rec)
    rec["cond"] = cond
    path = os.path.join(raw_dir(exp), cond_key(cond) + ".json")
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)


def load_results(exp):
    out = []
    d = raw_dir(exp)
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                out.append(json.load(f))
    return out


# ---- guidance construction ------------------------------------------------
def es_best_path():
    return os.path.join(CACHE, "es_best.npz")


def load_es_theta():
    z = np.load(es_best_path(), allow_pickle=True)
    return z["theta"]


def precompute_tolls(map_names_Ns, **kw):
    """Serial toll precomputation so parallel workers hit a warm cache.
    map_names_Ns: iterable of (map_name, N)."""
    for name, N in map_names_Ns:
        m = mapmod.get_map(name)
        w, info = guid.wardrop_tolls(m, N, cache_dir=CACHE, **kw)
        print(f"  tolls[{name}, N={N}] solve_time={info['solve_time']:.2f}s "
              f"lam={info['lam']:.2f} gap={info['gap']:.1e}")


def build_weights(m, cond):
    """Return (edge_weights or None, build_info). Tolls must be pre-cached."""
    method = cond["method"]
    if method == "unweighted":
        return None, {}
    if method == "lanes":
        return guid.crisscross_lanes(m, penalty=5.0), {}
    if method == "tolls" or method == "tolls-online":
        kw = {k: cond[k] for k in ("alpha", "beta", "gamma", "mu", "form")
              if k in cond}
        kw.setdefault("gamma", GAMMA)
        kw.setdefault("mu", MU)
        w, info = guid.wardrop_tolls(m, cond.get("N_solve", cond["N"]),
                                     seed_lanes=cond.get("seed_lanes", True),
                                     cache_dir=CACHE, **kw)
        return (None, info) if method == "tolls-online" else (w, info)
    if method == "es":
        return guid.es_weights(m, load_es_theta()), {}
    if method == "randw":
        # a fixed random real-valued field with the same weight range as the
        # tolls: real-valued but carrying no traffic information
        rng = np.random.default_rng(12345)
        edges, _ = mapmod.directed_edges(m)
        return 1.0 + 2.0 * rng.random(len(edges)), {}
    if method.startswith("es-strong-"):
        tag = method[len("es-strong-"):]
        z = np.load(os.path.join(CACHE, f"es_strong_best_{tag}.npz"),
                    allow_pickle=True)
        param = str(z["param"]) if "param" in z.files else "grid"
        if param == "full":       # one log-weight per directed edge
            return np.exp(np.clip(z["theta"].astype(float), -2.0, 2.0)), {}
        return guid.es_weights_bilinear(m, z["theta"], K=int(z["K"])), {}
    raise ValueError(method)


def run_condition(cond):
    """Worker: run one lifelong simulation described by cond. Returns record."""
    m = mapmod.get_map(cond["map"])
    method = cond["method"]
    online = None
    if method == "tolls-online":
        weights = None       # cold start: no prior demand knowledge
        binfo = {}
        kw = {k: cond[k] for k in ("alpha", "beta", "gamma", "mu", "form")
              if k in cond}
        kw.setdefault("gamma", GAMMA)
        kw.setdefault("mu", MU)
        net = flowmod.FlowNet(m, **kw)
        online = dict(ONLINE_CFG)
        online["period"] = cond.get("period", online["period"])
        online["net"] = net
        if cond.get("seed_lanes", True):
            online["init_bias"] = mapmod.crisscross_weights(
                m, guid.SEED_BIAS_PENALTY)
    else:
        weights, binfo = build_weights(m, cond)
    r = run_lifelong(m, cond["N"], cond.get("steps", STEPS), seed=cond["seed"],
                     edge_weights=weights, online=online,
                     record_edges=cond.get("record_edges", False),
                     swap=cond.get("swap", True), tau=cond.get("tau", TAU),
                     eps_tie=cond.get("eps_tie", EPS_TIE),
                     planner=cond.get("planner", "pibt"),
                     planner_kw=cond.get("planner_kw"),
                     hotspot=cond.get("hotspot", 0.0))
    rec = dict(throughput=r["throughput"], completed=r["completed"],
               steps=r["steps"], wall_time=r["wall_time"],
               resolve_time=r["resolve_time"], n_resolves=r["n_resolves"],
               series=r["series"])
    if binfo:
        rec["weight_build"] = {k: binfo[k] for k in
                               ("solve_time", "lam", "gap", "iters") if k in binfo}
    if cond.get("record_edges"):
        ec = r["edge_counts"]
        np.savez_compressed(os.path.join(CACHE,
                            f"edges_{cond_key(cond)}.npz"), edge_counts=ec)
        rec["edge_counts_file"] = f"cache/edges_{cond_key(cond)}.npz"
    return rec


def run_pending(exp, conds, max_minutes=9.0, workers=N_WORKERS):
    """Run all not-yet-done conditions with a worker pool; checkpoint each.
    Stops dispatching after max_minutes (results so far are saved)."""
    pending = [c for c in conds if not is_done(exp, c)]
    print(f"[{exp}] {len(conds)} conditions, {len(pending)} pending")
    if not pending:
        return True
    t0 = time.time()
    done_ct = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        it = pool.imap_unordered(_worker, [(exp, c) for c in pending])
        for exp_, cond, rec in it:
            save_result(exp_, cond, rec)
            done_ct += 1
            el = time.time() - t0
            if done_ct % 10 == 0 or el > max_minutes * 60:
                print(f"  [{exp}] {done_ct}/{len(pending)} done, {el:.0f}s")
            if el > max_minutes * 60:
                print(f"  [{exp}] time budget hit, exiting cleanly (re-run to resume)")
                pool.terminate()
                return False
    print(f"[{exp}] complete ({done_ct} new)")
    return True


def _worker(arg):
    exp, cond = arg
    rec = run_condition(cond)
    return exp, cond, rec
