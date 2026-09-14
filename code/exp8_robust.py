"""exp8: shift-robust confidence sequence (Prop. 4) -- validity, price, and use.

Theorem 1 assumes deployment scores are exchangeable with the pool.  Every
closed-loop and cross-scene regime in this paper violates that, which is exactly
the objection an adaptive-conformal advocate raises.  Prop. 4 makes the
degradation explicit: under a TOTAL shift budget sum_t (p_t - beta)_+ <= rho the
time-uniform guarantee is restored by inflating the log-threshold by rho/beta,
i.e. by running the CS at an effective confidence delta * exp(-rho/beta).

Parts:
  (a) synthetic: plain CS vs robust CS under persistent and transient departures
      from exchangeability -- the plain CS loses its guarantee, the robust one
      keeps it, and the required budget matches the theory;
  (b) the price of robustness: radius as a function of rho on the real sf
      calibration pool;
  (c) a descriptive marginal-frequency diagnostic for closed-loop (orca, sf)
      and cross-scene regimes.  It is not the theorem's unobserved pathwise sum
      of conditional probabilities; exp12 adds episode-cluster uncertainty and
      held-out transfer.

Run:  .venv/bin/python code/exp8_robust.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csq import RobustQuantileCS, cs_boundary, split_cp_quantile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
MAIN = os.path.join(ROOT, "results", "main")
CACHE = os.path.join(ROOT, "results", "cache")
OUT = os.path.join(ROOT, "results", "exp8_robust.json")


def anytime_fail(beta, delta, T, R, p_vec, rho_tot=0.0, rho_step=0.0, seed=1):
    """P(exists t<=T: U_t < q_beta) for a stream whose conditional exceedance
    probability is p_vec[t].  Distribution-free: only the indicator matters."""
    u = cs_boundary(T, beta, delta, cache_dir=CACHE, rho_tot=rho_tot, rho_step=rho_step)
    rng = np.random.default_rng(seed)
    out = np.zeros(R, dtype=bool)
    for c in range(0, R, 2000):
        m = min(2000, R - c)
        Z = np.cumsum(rng.random((m, T)) < p_vec[None, :], axis=1)
        out[c:c + m] = (Z >= u[None, :]).any(axis=1)
    return float(out.mean())


def part_a():
    beta, delta, T, R = 0.95, 0.05, 3000, 8000
    rows = []
    scen = {
        "exchangeable": np.full(T, beta),
        "persistent_0.01": np.full(T, beta + 0.01),          # budget 30 over 3000
        "persistent_0.003": np.full(T, beta + 0.003),        # budget 9
        "transient_0.04_first500": np.where(np.arange(1, T + 1) <= 500, beta + 0.04, beta),
    }
    for name, p in scen.items():
        used = float(np.maximum(p - beta, 0).sum())
        row = {"scenario": name, "true_shift_budget": used,
               "plain_cs_fail": anytime_fail(beta, delta, T, R, p),
               "robust_cs_fail": anytime_fail(beta, delta, T, R, p, rho_tot=used),
               "target": delta}
        rows.append(row)
        print(f"  {name:26s} rho={used:6.2f}  plain {row['plain_cs_fail']:.4f}"
              f"  robust {row['robust_cs_fail']:.4f}  (target {delta})")
    return rows


def part_b():
    """Price of robustness: radius vs rho on the real sf calibration pool."""
    s = np.sort(np.load(os.path.join(DATA, "cal_sf.npy")))
    n = len(s)
    rows = []
    for delta in (0.05, 0.10):
        beta = 1.0 - delta
        plug = split_cp_quantile(list(s), beta)
        base = None
        for rho in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
            cs = RobustQuantileCS(beta, delta, s, rho_tot=rho, t_max=20000, cache_dir=CACHE)
            r = cs.radius()
            base = r if base is None else base
            rows.append({"delta": delta, "rho_tot": rho, "n": n, "radius": float(r),
                         "plugin_quantile": float(plug),
                         "premium_vs_plugin_pct": 100.0 * (r / plug - 1.0),
                         "premium_vs_cs_pct": 100.0 * (r / base - 1.0),
                         "eff_delta": float(delta * np.exp(-rho / beta))})
            print(f"  delta={delta} rho={rho:5.1f}: r={r:.3f} m  (+{100*(r/plug-1):.1f}% vs plug-in,"
                  f" +{100*(r/base-1):.1f}% vs exchangeable CS)")
    return rows


def part_c():
    """Empirical proxy for, not a certificate of, the conditional shift budget.

    We report W*(phat-beta)_+ from recorded labels against a reference quantile.
    This estimates a marginal frequency.  Recorded labels cannot identify
    p_t=P(s_t<q_beta|F_{t-1}) or prove the pathwise premise of Proposition 4.
    """
    rows = []
    for regime, pool in [("replay", "cal_sf.npy"), ("sf", "cal_sf.npy"),
                         ("orca", "cal_orca.npy")] + \
                        [(f"real_{s}", f"cal_real_{s}.npy") for s in
                         ("eth", "hotel", "univ", "zara1", "zara2")]:
        pf = os.path.join(DATA, pool)
        if not os.path.exists(pf):
            continue
        cal = np.sort(np.load(pf))
        for delta in (0.05, 0.10):
            beta = 1.0 - delta
            q = float(np.quantile(cal, beta))
            f = os.path.join(MAIN, f"{regime}_cs_{delta}.json")
            if not os.path.exists(f):
                continue
            recs = json.load(open(f))["records"]
            sc = np.array([w[0] for r in recs for w in r["windows"]])
            if sc.size == 0:
                continue
            below = float((sc < q).mean())          # realised p
            W = int(sc.size)
            rows.append({"regime": regime, "delta": delta, "n_windows": W,
                         "beta": beta, "p_hat": below,
                         "excess_per_window": max(below - beta, 0.0),
                         "budget_over_120_windows": 120.0 * max(below - beta, 0.0),
                         "budget_over_deployment": W * max(below - beta, 0.0),
                         "formal_conditional_shift_budget": False})
            print(f"  {regime:11s} delta={delta}: p_hat={below:.4f} vs beta={beta}"
                  f" -> rho(120 windows)={120*max(below-beta,0):.3f}")
    return rows


def main():
    print("[exp8a] validity of the robust CS under bounded shift")
    a = part_a()
    print("[exp8b] price of robustness (sf pool)")
    b = part_b()
    print("[exp8c] descriptive marginal-frequency shift diagnostic")
    c = part_c()
    json.dump({"validity": a, "price": b, "measured_budget": c}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
