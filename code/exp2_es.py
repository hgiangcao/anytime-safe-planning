"""Exp 2 — black-box guidance-weight search baseline (CMA-ES), GGO-style.

Self-implemented standard CMA-ES (Hansen's update equations, full covariance)
over a 64-dim tiled log-weight parameterization (4x4 tile x 4 directions) of
the guidance graph — an honest laptop-scale stand-in for GGO's black-box
search (GGO uses CMA-ES over larger parameterizations with far larger rollout
budgets on clusters).

Budget: lam=12, 25 generations = 300 rollout evaluations; each evaluation =
mean throughput over 2 fixed seeds x 500 steps on empty-32x32 at N=400
(the training condition). Total search compute is recorded and reported in
the compute-vs-throughput figure.

Checkpointed per generation; final best theta is then evaluated with the
full protocol (10 seeds x 1000 steps) and saved as cache/es_best.npz.
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

MAP = "empty-32x32"
N_AGENTS = 400
LAM = 12
GENS = 25
EVAL_SEEDS = (10007, 10009)
EVAL_STEPS = 500
TILE = 4
STATE = os.path.join(common.CACHE, "es_state.npz")


class CMAES:
    """Standard (mu/mu_w, lambda)-CMA-ES (Hansen), minimization."""

    def __init__(self, x0, sigma0, lam, seed=0):
        self.d = d = len(x0)
        self.lam = lam
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / d) / (d + 4 + 2 * self.mueff / d)
        self.cs = (self.mueff + 2) / (d + self.mueff + 5)
        self.c1 = 2 / ((d + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((d + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (d + 1)) - 1) + self.cs
        self.chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        self.mean = np.asarray(x0, float)
        self.sigma = float(sigma0)
        self.C = np.eye(d)
        self.pc = np.zeros(d)
        self.ps = np.zeros(d)
        self.gen = 0
        self.seed = seed

    def ask(self):
        rng = np.random.default_rng(self.seed * 100003 + self.gen)
        ev, B = np.linalg.eigh(self.C)
        Dd = np.sqrt(np.maximum(ev, 1e-20))
        self._B, self._D = B, Dd
        z = rng.standard_normal((self.lam, self.d))
        y = z @ (B * Dd).T          # y_i = B D z_i
        return self.mean + self.sigma * y, y

    def tell(self, ys, fits):
        idx = np.argsort(fits)
        ysel = ys[idx[:self.mu]]
        ybar = self.w @ ysel
        self.mean = self.mean + self.sigma * ybar
        B, Dd = self._B, self._D
        invsqrtC_y = B @ ((B.T @ ybar) / Dd)
        self.ps = ((1 - self.cs) * self.ps +
                   np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_y)
        hs = (np.linalg.norm(self.ps) /
              np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN
              < 1.4 + 2 / (self.d + 1))
        self.pc = ((1 - self.cc) * self.pc +
                   hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ybar)
        artmp = ysel
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) +
                             (not hs) * self.cc * (2 - self.cc) * self.C) +
                  self.cmu * (artmp.T * self.w) @ artmp)
        self.C = (self.C + self.C.T) / 2
        self.sigma = self.sigma * np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.gen += 1


def eval_theta(theta):
    """Fitness = -mean throughput over fixed seeds (common random numbers)."""
    t0 = time.time()
    m = mapmod.get_map(MAP)
    w = guid.es_weights(m, theta, tile=TILE)
    D = dist_table(m, w)
    tputs = [run_lifelong(m, N_AGENTS, EVAL_STEPS, seed=s, D=D)["throughput"]
             for s in EVAL_SEEDS]
    return -float(np.mean(tputs)), time.time() - t0


def save_state(es, best, history, total_time, n_evals):
    np.savez_compressed(STATE + ".tmp.npz", mean=es.mean, sigma=es.sigma,
                        C=es.C, pc=es.pc, ps=es.ps, gen=es.gen,
                        best_theta=best[0], best_fit=best[1],
                        history=json.dumps(history),
                        total_time=total_time, n_evals=n_evals)
    os.replace(STATE + ".tmp.npz", STATE)


def load_state(es):
    z = np.load(STATE, allow_pickle=True)
    es.mean, es.sigma = z["mean"], float(z["sigma"])
    es.C, es.pc, es.ps, es.gen = z["C"], z["pc"], z["ps"], int(z["gen"])
    return ((z["best_theta"], float(z["best_fit"])),
            json.loads(str(z["history"])), float(z["total_time"]), int(z["n_evals"]))


def main(max_minutes=8.5):
    import multiprocessing as mp
    es = CMAES(np.zeros(guid.es_dim(TILE)), 0.5, LAM, seed=1)
    best, history, total_time, n_evals = (None, np.inf), [], 0.0, 0
    if os.path.exists(STATE):
        best, history, total_time, n_evals = load_state(es)
        print(f"resumed at gen {es.gen}, best={-best[1]:.3f}, "
              f"{n_evals} evals, {total_time:.0f}s")
    t_start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(common.N_WORKERS) as pool:
        while es.gen < GENS:
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
            print(f"gen {es.gen}: best {-fits.min():.3f} mean {-fits.mean():.3f} "
                  f"best-ever {-best[1]:.3f} sigma {es.sigma:.3f} "
                  f"({total_time:.0f}s, {n_evals} evals)")
            if time.time() - t_start > max_minutes * 60:
                print("time budget hit; re-run to resume")
                return
    # search finished -> freeze best theta
    m = mapmod.get_map(MAP)
    np.savez_compressed(common.es_best_path(), theta=best[0],
                        fit=best[1], weights=guid.es_weights(m, best[0], TILE),
                        search_time=total_time, n_evals=n_evals,
                        map=MAP, N=N_AGENTS, history=json.dumps(history))
    print(f"ES search done: {n_evals} evals, {total_time:.0f}s, "
          f"train fitness {-best[1]:.3f}. Final full-protocol eval...")
    conds = [dict(map=MAP, N=N_AGENTS, method="es", seed=s) for s in common.SEEDS]
    common.run_pending("exp2_eval", conds)


if __name__ == "__main__":
    main()
