"""exp2: certified inflation radius vs horizon/time - the conservatism separation.

Curves (written to results/exp2_radius.csv, plotted by exp4):
  1. union-bound CP: radius needed for a T-window episode = split-CP quantile at
     level 1 - delta/T of the n calibration scores; grows with T and becomes +inf
     (uncertifiable) once delta/T < 1/(n+1), i.e. T > delta (n+1).
  2. AUC (ours-episode): radius at window w of an episode = quantile at level
     1 - delta/(w(w+1)) (pool accretes one score per window); the SAME episode
     guarantee without knowing T, but its per-window level also decays -- and,
     crucially, it becomes UNCERTIFIABLE after only ~sqrt(delta n) windows.
  2b. AUC-B (ours-episode, budget-aware): same episode guarantee, spending
     eps_w = max(1/(n_w+1), lam * budget_remaining), whose lam=0 endpoint reaches
     the exact harmonic cutoff max{W: sum_{w<=W} 1/(n+w) <= delta}.
  2c. plug-in split-CP at level 1-eps: the LIKE-FOR-LIKE per-window comparator
     for the CS (same semantics, but only fixed-time valid -- it is the object
     whose repeated re-evaluation fails in ~96% of runs, cf. exp1).
  3. CS (ours): time-uniform upper confidence bound on the (1-eps) quantile after
     t scores; the displayed uniform mixture contracts at sqrt(log(t)/t), not at
     the LIL-optimal rate; every window
     simultaneously carries per-window risk <= eps with confidence 1 - delta.
  4. Rayleigh chance constraint at level 1 - delta/T (parametric; no finite-sample
     guarantee).
Score sources: the real sf calibration scores (n = ~2100) and a long synthetic
Rayleigh stream (to show flattening as t grows).

Run: .venv/bin/python code/exp2_radius.py   (~1 min)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csq import (BudgetedAUC, QuantileCS, cert_horizon_fixed_T,
                 cert_horizon_minimal, cert_horizon_schedule, cs_boundary,
                 split_cp_quantile)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
CACHE = os.path.join(ROOT, "results", "cache")

def _aucb_horizon(n0, delta, lam, w_max=10**6):
    """Certifiable horizon of BudgetedAUC (number of leading windows with a
    finite radius).  Mirrors BudgetedAUC._advance exactly but tracks only the
    budget arithmetic, so it is O(W) instead of O(W n); test_core.py checks the
    two agree."""
    d_rem, w = float(delta), 0
    while w < w_max:
        floor = 1.0 / (n0 + w + 1.0)
        if d_rem < floor:
            return w
        d_rem -= max(floor, lam * d_rem)
        w += 1
    return w_max


rows = []
cal = np.sort(np.load(os.path.join(DATA, "cal_sf.npy")))
n = len(cal)
rng = np.random.default_rng(7)

for delta in [0.05, 0.1]:
    # 1 & 4: union-bound CP and Rayleigh CC radius as a function of horizon T
    sigma = float(np.sqrt(np.mean(cal**2) / 2.0))
    # include the exact blow-up threshold T* = min{T : delta/T < 1/(n+1)} in the
    # grid so the reported "uncertifiable beyond T" is exact, not grid-rounded
    # (verification fix 2026-09-02: the geomspace grid alone reported 111/218
    # instead of the true 105/210)
    T_star = int(np.floor(delta * (n + 1))) + 1
    Tgrid = np.unique(np.concatenate([
        np.round(np.geomspace(1, 3000, 120)).astype(int), [T_star - 1, T_star]]))
    for T in Tgrid:
        r_u = split_cp_quantile(cal, 1.0 - delta / T)
        r_g = sigma * np.sqrt(-2.0 * np.log(delta / T))
        rows.append(dict(curve="union", delta=delta, x=int(T), radius=float(r_u)))
        rows.append(dict(curve="gauss", delta=delta, x=int(T), radius=float(r_g)))
    # 2: AUC per-window radius (pool accretes 1 score per window, drawn from cal
    #    with replacement as a surrogate deployment stream)
    pool = list(cal)
    pool_sorted = np.sort(pool)
    for w in range(1, 61):
        eps_w = delta / (w * (w + 1.0))
        r = split_cp_quantile(pool_sorted, 1.0 - eps_w)
        rows.append(dict(curve="auc", delta=delta, x=w, radius=float(r)))
        pool_sorted = np.sort(np.append(pool_sorted, rng.choice(cal)))
    # 2b: AUC-B per-window radius, same surrogate deployment stream
    ab = BudgetedAUC(delta, cal, lam=0.05)
    for w in range(1, 61):
        rows.append(dict(curve="aucb", delta=delta, x=w, radius=float(ab.radius())))
        ab.add(float(rng.choice(cal)))
    # 2c: plug-in split-CP at the per-window level 1-eps (eps = delta): the
    #     like-for-like comparator for the CS; horizon-independent by definition
    r_pi = float(split_cp_quantile(cal, 1.0 - delta))
    for T in Tgrid:
        rows.append(dict(curve="plugin", delta=delta, x=int(T), radius=r_pi))
    # 3: CS upper bound vs number of scores t (real scores, shuffled stream)
    stream = rng.permutation(cal)
    beta = 1.0 - delta  # eps = delta
    u_bnd = cs_boundary(n + 10, beta, delta, cache_dir=CACHE)
    srt = []
    import bisect as _b
    for t, s in enumerate(stream, start=1):
        _b.insort(srt, float(s))
        u = int(u_bnd[t - 1])
        r = srt[u - 1] if u <= t else np.inf
        if t in (5, 10, 20, 50) or t % 50 == 0:
            rows.append(dict(curve="cs", delta=delta, x=t, radius=float(r)))
    rows.append(dict(curve="split_q", delta=delta, x=n,
                     radius=float(split_cp_quantile(cal, 1.0 - delta))))

# synthetic long-stream flattening (Rayleigh sigma=0.35 ~ sf scores)
SIG = 0.35
TL = 100000
for delta in [0.05]:
    beta = 1.0 - delta
    q_true = SIG * np.sqrt(-2 * np.log(delta))
    u_bnd = cs_boundary(TL, beta, delta, cache_dir=CACHE)
    s = SIG * np.sqrt(-2 * np.log(rng.random(TL)))
    order = np.argsort(s)
    # U_t for checkpoints via full sort of prefix (vectorized enough at checkpoints)
    for t in np.unique(np.round(np.geomspace(20, TL, 60)).astype(int)):
        u = int(u_bnd[t - 1])
        pre = np.sort(s[:t])
        r = pre[u - 1] if u <= t else np.inf
        rows.append(dict(curve="cs_synth", delta=delta, x=int(t), radius=float(r)))
    rows.append(dict(curve="q_true_synth", delta=delta, x=TL, radius=float(q_true)))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "exp2_radius.csv"), index=False)

# ---------------------------------------------- certifiable horizon vs pool size
# How many consecutive windows of ONE episode can each episode-level calibrator
# certify at all (finite radius) from a pool of n scores?  Prop. 1 makes the
# fixed-T answer allocation-independent; Prop. 3 gives the horizon-free optimum.
hrows = []
for delta in [0.05, 0.1]:
    for nn in np.unique(np.round(np.geomspace(50, 200000, 60)).astype(int)):
        nn = int(nn)
        hrows.append(dict(delta=delta, n=nn, method="union_anyalloc",
                          horizon=cert_horizon_fixed_T(nn, delta)))
        hrows.append(dict(delta=delta, n=nn, method="auc",
                          horizon=cert_horizon_schedule(
                              (lambda d: (lambda w: d / (w * (w + 1.0))))(delta),
                              nn, delta)))
        hrows.append(dict(delta=delta, n=nn, method="optimal",
                          horizon=cert_horizon_minimal(nn, delta)))
        for lam in (0.05, 0.01):
            hrows.append(dict(delta=delta, n=nn, method=f"aucb_lam{lam}",
                              horizon=_aucb_horizon(nn, delta, lam)))
        hrows.append(dict(delta=delta, n=nn, method="cs", horizon=-1))  # -1 = unbounded
hdf = pd.DataFrame(hrows)
hdf.to_csv(os.path.join(ROOT, "results", "exp2_horizon.csv"), index=False)

# ------------------------------------------- eps vs delta sensitivity of the CS
# Theorem 1 keeps the per-window risk eps and the CS confidence delta distinct;
# the experiments tie eps = delta.  This sweep shows what that costs.
srows = []
for delta_cs in [0.01, 0.05, 0.1, 0.2]:
    for eps in [0.02, 0.05, 0.1, 0.2]:
        u_b = cs_boundary(n, 1.0 - eps, delta_cs, cache_dir=CACHE)
        u = int(u_b[n - 1])
        r_cs = float(cal[u - 1]) if u <= n else float("inf")
        srows.append(dict(delta=delta_cs, eps=eps, n=n, cs_radius=r_cs,
                          plugin_radius=float(split_cp_quantile(cal, 1.0 - eps))))
pd.DataFrame(srows).to_csv(os.path.join(ROOT, "results", "exp2_epsdelta.csv"),
                           index=False)
# headline numbers
hl = {}
for delta in [0.05, 0.1]:
    d = df[(df.curve == "union") & (df.delta == delta)]
    Tinf = int(d[~np.isfinite(d.radius)].x.min()) if (~np.isfinite(d.radius)).any() else None
    cs_end = float(df[(df.curve == "cs") & (df.delta == delta)].radius.iloc[-1])
    u50 = d[d.x <= 50].radius.max()
    hl[f"delta_{delta}"] = {
        "union_uncertifiable_beyond_T": Tinf,
        "union_radius_T50": float(u50),
        "cs_radius_final_t~2100": cs_end,
        "split_quantile_1-delta": float(split_cp_quantile(cal, 1 - delta)),
        "cs_over_plugin_pct": 100.0 * (cs_end / float(split_cp_quantile(cal, 1 - delta)) - 1.0),
        "cert_horizon_union_any_allocation": cert_horizon_fixed_T(n, delta),
        "cert_horizon_auc_wsq": cert_horizon_schedule(
            (lambda d: (lambda w: d / (w * (w + 1.0))))(delta), n, delta),
        "cert_horizon_aucb_lam0.05": _aucb_horizon(n, delta, 0.05),
        "cert_horizon_aucb_lam0.01": _aucb_horizon(n, delta, 0.01),
        "cert_horizon_optimal": cert_horizon_minimal(n, delta),
        "cert_horizon_cs": "unbounded"}
with open(os.path.join(ROOT, "results", "exp2_headline.json"), "w") as f:
    json.dump(hl, f, indent=1)
print(json.dumps(hl, indent=1))
print("exp2 complete")
