"""Exp 2b — larger-budget CMA-ES points for the compute-throughput curve.

Continues the exp2 CMA-ES search (resuming from cache/es_state.npz, identical
update equations, parameterization, and per-eval protocol) from 300 up to
3,000 rollout evaluations, snapshotting the best-ever theta at ~1,000 and
3,000 evals. Each snapshot is then evaluated with the full protocol
(10 seeds x 1,000 steps) so Fig. 1's search curve is a trend, not a point.

State is kept in cache/es_state_ext.npz; exp2's own files are not touched.
"""
import json
import os
import time

import numpy as np

import common
from mapf import maps as mapmod
from mapf import guidance as guid
from mapf.pibt import dist_table
from mapf.sim import run_lifelong

from exp2_es import CMAES, eval_theta, MAP, N_AGENTS, LAM, TILE

STATE0 = os.path.join(common.CACHE, "es_state.npz")        # exp2 (read-only)
STATE = os.path.join(common.CACHE, "es_state_ext.npz")     # ours
MILESTONES = {996: "es1000", 3000: "es3000"}               # n_evals -> label
GENS_MAX = 250                                             # 250*12 = 3000 evals


def snap_path(label):
    return os.path.join(common.CACHE, f"es_best_{label}.npz")


def save_state(es, best, history, total_time, n_evals):
    np.savez_compressed(STATE + ".tmp.npz", mean=es.mean, sigma=es.sigma,
                        C=es.C, pc=es.pc, ps=es.ps, gen=es.gen,
                        best_theta=best[0], best_fit=best[1],
                        history=json.dumps(history),
                        total_time=total_time, n_evals=n_evals)
    os.replace(STATE + ".tmp.npz", STATE)


def load_state(path, es):
    z = np.load(path, allow_pickle=True)
    es.mean, es.sigma = z["mean"], float(z["sigma"])
    es.C, es.pc, es.ps, es.gen = z["C"], z["pc"], z["ps"], int(z["gen"])
    return ((np.asarray(z["best_theta"]), float(z["best_fit"])),
            json.loads(str(z["history"])), float(z["total_time"]),
            int(z["n_evals"]))


def full_eval(label):
    """Full-protocol eval (10 seeds x 1000 steps) of a snapshot theta."""
    z = np.load(snap_path(label), allow_pickle=True)
    m = mapmod.get_map(MAP)
    w = guid.es_weights(m, np.asarray(z["theta"]), tile=TILE)
    for s in common.SEEDS:
        cond = dict(map=MAP, N=N_AGENTS, method=label, seed=s)
        if common.is_done("exp2b_eval", cond):
            continue
        r = run_lifelong(m, N_AGENTS, common.STEPS, seed=s, edge_weights=w)
        common.save_result("exp2b_eval", cond, dict(
            throughput=r["throughput"], completed=r["completed"],
            steps=r["steps"], wall_time=r["wall_time"],
            resolve_time=r["resolve_time"], n_resolves=r["n_resolves"],
            series=r["series"]))
    print(f"[exp2b] full eval done: {label}")


def main():
    import multiprocessing as mp
    es = CMAES(np.zeros(guid.es_dim(TILE)), 0.5, LAM, seed=1)
    if os.path.exists(STATE):
        best, history, total_time, n_evals = load_state(STATE, es)
        print(f"[exp2b] resumed ext state at gen {es.gen}, "
              f"{n_evals} evals, best {-best[1]:.3f}")
    else:
        best, history, total_time, n_evals = load_state(STATE0, es)
        print(f"[exp2b] starting from exp2 state: gen {es.gen}, "
              f"{n_evals} evals, best {-best[1]:.3f}, {total_time:.0f}s")
    ctx = mp.get_context("fork")
    with ctx.Pool(common.N_WORKERS) as pool:
        while es.gen < GENS_MAX:
            tg = time.time()
            X, Y = es.ask()
            out = pool.map(eval_theta, list(X))
            fits = np.array([o[0] for o in out])
            n_evals += len(fits)
            i = int(np.argmin(fits))
            if fits[i] < best[1]:
                best = (X[i].copy(), float(fits[i]))
            es.tell(Y, fits)
            total_time += time.time() - tg
            history.append(dict(gen=es.gen, best=-float(fits.min()),
                                mean=-float(fits.mean()),
                                best_ever=-best[1], sigma=float(es.sigma),
                                cum_time=total_time, n_evals=n_evals))
            save_state(es, best, history, total_time, n_evals)
            if es.gen % 10 == 0:
                print(f"[exp2b] gen {es.gen}: best-ever {-best[1]:.3f} "
                      f"sigma {es.sigma:.3f} ({total_time:.0f}s, {n_evals} evals)")
            for me, label in MILESTONES.items():
                if n_evals >= me and not os.path.exists(snap_path(label)):
                    m = mapmod.get_map(MAP)
                    np.savez_compressed(snap_path(label), theta=best[0],
                                        fit=best[1],
                                        weights=guid.es_weights(m, best[0], TILE),
                                        search_time=total_time, n_evals=n_evals,
                                        map=MAP, N=N_AGENTS)
                    print(f"[exp2b] milestone {label}: {n_evals} evals, "
                          f"{total_time:.0f}s, train fitness {-best[1]:.3f}")
    for label in MILESTONES.values():
        full_eval(label)
    print("[exp2b] all done")


if __name__ == "__main__":
    main()
