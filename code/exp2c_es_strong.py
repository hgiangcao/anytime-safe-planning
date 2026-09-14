"""Exp 2c - STRENGTHENED black-box guidance search baseline.

Fixes two handicaps of exp2's baseline that made "derivation beats search"
untestable:

 1. PARAMETERIZATION. exp2 searched a 4x4 tiled field indexed by `uy % tile`,
    which is spatially PERIODIC: it cannot represent boundary-aware or
    map-specific structure, so no budget could reach the toll solution. Here
    CMA-ES searches a non-periodic bilinear control-point field
    (K x K x 4 = 256 dims at K=8) that CAN represent it.
 2. WARM START. `--warm` initialises the CMA-ES mean at the least-squares
    projection of the derived log-tolls onto the control-point basis, i.e.
    search on top of derivation (the hybrid).

Everything else matches exp2: standard Hansen CMA-ES, common random numbers,
per-generation checkpointing, and a final full-protocol (10 seeds x 1000
steps) evaluation of the frozen best weights.  Review-remediation note: new
runs explicitly evaluate and retain x0 before sampling.  The frozen 2026-09-03
states predate that fix and are labeled as legacy/unretained by the audit; they
must not be presented as incumbent-preserving search.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import MU, GAMMA
from exp2_es import CMAES
from mapf import maps as mapmod
from mapf import guidance as guid
from mapf.pibt import dist_table
from mapf.sim import run_lifelong

MAP = "empty-32-32"
N_AGENTS = 400
K = 8
LAM = 16
EVAL_SEEDS = (10007, 10009)
EVAL_STEPS = 500

# Two search spaces, because expressiveness turns out to be the crux:
#   "grid" : K x K x 4 bilinear control points (256 dims). Smooth, and
#            therefore CANNOT represent the cell-scale alternating lane
#            structure of the toll field -- projecting the tolls onto it costs
#            ~16% relative error and about 1.2 tasks/step.
#   "full" : one log-weight per directed edge (|E| = 3968 on empty-32-32).
#            Fully expressive by construction, so "search cannot find it" is
#            about the search, not the parameterisation. Full-covariance
#            CMA-ES is O(d^2) memory / O(d^3) per update and infeasible here,
#            so this uses separable CMA-ES (Ros & Hansen, PPSN-08), the
#            standard high-dimensional variant.
PARAM = "grid"


class SepCMAES:
    """Separable (diagonal-covariance) CMA-ES for high-dimensional search."""

    def __init__(self, x0, sigma0, lam, seed=0):
        self.d = d = len(x0)
        self.lam = lam
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / d) / (d + 4 + 2 * self.mueff / d)
        self.cs = (self.mueff + 2) / (d + self.mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + self.mueff)
        cmu = min(1 - c1, 2 * (self.mueff - 2 + 1 / self.mueff)
                  / ((d + 2) ** 2 + self.mueff))
        scale = (d + 2) / 3.0            # separable speed-up (Ros & Hansen)
        self.c1, self.cmu = min(1.0, c1 * scale), min(1 - c1 * scale, cmu * scale)
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (d + 1)) - 1) + self.cs
        self.chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        self.mean = np.asarray(x0, float)
        self.sigma = float(sigma0)
        self.C = np.ones(d)              # diagonal of the covariance
        self.pc = np.zeros(d)
        self.ps = np.zeros(d)
        self.gen = 0
        self.seed = seed

    def ask(self):
        rng = np.random.default_rng(self.seed * 100003 + self.gen)
        self._D = np.sqrt(self.C)
        z = rng.standard_normal((self.lam, self.d))
        y = z * self._D
        return self.mean + self.sigma * y, y

    def tell(self, ys, fits):
        idx = np.argsort(fits)
        ysel = ys[idx[:self.mu]]
        ybar = self.w @ ysel
        self.mean = self.mean + self.sigma * ybar
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * ybar / self._D)
        hs = (np.linalg.norm(self.ps)
              / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN
              < 1.4 + 2 / (self.d + 1))
        self.pc = ((1 - self.cc) * self.pc
                   + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ybar)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2
                               + (not hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (self.w @ (ysel ** 2)))
        self.C = np.maximum(self.C, 1e-12)
        self.sigma *= np.exp((self.cs / self.damps)
                             * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.gen += 1


def weights_from(m, theta):
    if PARAM == "full":
        return np.exp(np.clip(np.asarray(theta, float), -2.0, 2.0))
    return guid.es_weights_bilinear(m, theta, K=K)


def n_params(m):
    if PARAM == "full":
        return len(mapmod.directed_edges(m)[0])
    return guid.es_dim_bilinear(K)


def state_path(tag):
    return os.path.join(common.CACHE, f"es_strong_{tag}.npz")


def best_path(tag):
    return os.path.join(common.CACHE, f"es_strong_best_{tag}.npz")


def eval_theta(theta):
    m = mapmod.get_map(MAP)
    w = weights_from(m, theta)
    t0 = time.time()
    D = dist_table(m, w)
    tputs = [run_lifelong(m, N_AGENTS, EVAL_STEPS, seed=s, D=D)["throughput"]
             for s in EVAL_SEEDS]
    return -float(np.mean(tputs)), time.time() - t0


def toll_projection():
    """Least-squares projection of the derived log-tolls onto the bilinear
    control-point basis (so the warm start is the best representable
    approximation of the toll field within the search space)."""
    m = mapmod.get_map(MAP)
    w, _ = guid.wardrop_tolls(m, N_AGENTS, gamma=GAMMA, mu=MU,
                              cache_dir=common.CACHE)
    target = np.log(np.asarray(w, float))
    if PARAM == "full":
        return target.copy(), 0.0
    d = guid.es_dim_bilinear(K)
    # the map theta -> log-weights is linear before the exp/clip, so build its
    # matrix column by column (256 cheap evaluations)
    cols = []
    for j in range(d):
        e = np.zeros(d); e[j] = 1.0
        cols.append(np.log(guid.es_weights_bilinear(m, e * 1.0, K=K)))
    A = np.stack(cols, axis=1)
    zero = np.log(guid.es_weights_bilinear(m, np.zeros(d), K=K))
    A = A - zero[:, None]
    theta, *_ = np.linalg.lstsq(A, target - zero, rcond=None)
    approx = guid.es_weights_bilinear(m, theta, K=K)
    err = float(np.abs(approx - w).mean() / np.abs(w).mean())
    print(f"  toll projection onto {d}-dim basis: mean rel. error {err:.3%}")
    return theta, err


def main(tag="cold", gens=25, max_minutes=600.0):
    m0 = mapmod.get_map(MAP)
    d = n_params(m0)
    if tag.endswith("warm"):        # "warm" and "fullwarm"
        x0, proj_err = toll_projection()
        sigma0 = 0.25
    else:
        x0, proj_err = np.zeros(d), None
        sigma0 = 0.5
    lam = LAM if PARAM == "grid" else int(4 + 3 * np.log(d))
    es = (CMAES(x0, sigma0, lam, seed=1) if PARAM == "grid"
          else SepCMAES(x0, sigma0, lam, seed=1))
    print(f"[{tag}] parameterisation={PARAM} dim={d} population={lam}")
    ST = state_path(tag)
    best, history, total_time, n_evals = (x0.copy(), np.inf), [], 0.0, 0
    initial_fit = np.nan
    initial_eval_time = 0.0
    initial_candidate_retained = False
    if os.path.exists(ST):
        z = np.load(ST, allow_pickle=True)
        es.mean, es.sigma = z["mean"], float(z["sigma"])
        es.C, es.pc, es.ps, es.gen = z["C"], z["pc"], z["ps"], int(z["gen"])
        best = (z["best_theta"], float(z["best_fit"]))
        history = json.loads(str(z["history"]))
        total_time, n_evals = float(z["total_time"]), int(z["n_evals"])
        if "initial_fit" in z.files:
            initial_fit = float(z["initial_fit"])
            initial_eval_time = float(z["initial_eval_time"])
            initial_candidate_retained = bool(z["initial_candidate_retained"])
        else:
            print("WARNING: legacy frozen state has no evaluated/retained x0; "
                  "do not interpret it as incumbent-preserving search")
        print(f"resumed {tag} at gen {es.gen}, best={-best[1]:.3f}, "
              f"{n_evals} evals, {total_time:.0f}s")
    else:
        # Evaluate x0 under exactly the training objective before any sample.
        # This protects a useful warm start and makes starting, retained-best,
        # and terminal candidates separately auditable.
        initial_fit, initial_eval_time = eval_theta(x0)
        best = (x0.copy(), float(initial_fit))
        total_time = float(initial_eval_time)
        n_evals = 1
        initial_candidate_retained = True
        history.append(dict(gen=0, best=-float(initial_fit),
                            best_ever=-float(initial_fit), sigma=float(es.sigma),
                            cum_time=total_time, n_evals=n_evals,
                            candidate="initial_mean"))
        print(f"[{tag}] evaluated and retained initial mean: "
              f"{-initial_fit:.3f}", flush=True)

    import multiprocessing as mp
    t_start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(common.N_WORKERS) as pool:
        while es.gen < gens:
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
                                best_ever=-best[1], sigma=float(es.sigma),
                                cum_time=total_time, n_evals=n_evals))
            np.savez_compressed(ST + ".tmp.npz", mean=es.mean, sigma=es.sigma,
                                C=es.C, pc=es.pc, ps=es.ps, gen=es.gen,
                                best_theta=best[0], best_fit=best[1],
                                history=json.dumps(history),
                                total_time=total_time, n_evals=n_evals,
                                initial_theta=x0, initial_fit=initial_fit,
                                initial_eval_time=initial_eval_time,
                                initial_candidate_retained=initial_candidate_retained)
            os.replace(ST + ".tmp.npz", ST)
            print(f"[{tag}] gen {es.gen}: best {-fits.min():.3f} "
                  f"best-ever {-best[1]:.3f} sigma {es.sigma:.3f} "
                  f"({total_time:.0f}s, {n_evals} evals)", flush=True)
            if time.time() - t_start > max_minutes * 60:
                print("time budget hit; re-run to resume")
                return
    m = mapmod.get_map(MAP)
    np.savez_compressed(best_path(tag), theta=best[0], fit=best[1],
                        weights=weights_from(m, best[0]), param=PARAM, dim=d,
                        search_time=total_time, n_evals=n_evals, map=MAP,
                        N=N_AGENTS, K=K, proj_err=proj_err if proj_err else -1.0,
                        history=json.dumps(history), initial_theta=x0,
                        initial_fit=initial_fit,
                        initial_eval_time=initial_eval_time,
                        initial_candidate_retained=initial_candidate_retained,
                        terminal_mean=es.mean)
    print(f"[{tag}] search done: {n_evals} evals, {total_time:.0f}s, "
          f"train fitness {-best[1]:.3f}. Full-protocol eval...")
    conds = [dict(map=MAP, N=N_AGENTS, method=f"es-strong-{tag}", seed=s,
                  steps=common.STEPS) for s in common.SEEDS]
    common.run_pending(f"exp2c_eval_{tag}", conds)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        PARAM = sys.argv[3]
    main(tag=sys.argv[1] if len(sys.argv) > 1 else "cold",
         gens=int(sys.argv[2]) if len(sys.argv) > 2 else 25)
