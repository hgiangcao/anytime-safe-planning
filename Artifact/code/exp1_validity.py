"""exp1: statistical validity of the core machinery on synthetic iid data
(the regime where Theorem 1's assumptions hold exactly).

(a) Time-uniform quantile coverage of the CS: fraction of runs in which the CS
    upper bound EVER (t <= T) falls below the true quantile.  Target <= delta.
    Also the same event for the naive online plug-in split-CP radius (recomputed
    every step at level 1-eps) - the 'replanning/peeking loophole': it has NO
    time-uniform guarantee and undercovers in almost every run.
(b) Episode-level validity of AUC / AUC-B (ours-episode) and fixed-T union-bound
    CP on iid scores.  ACI appears in TWO tunings, because judging a
    window-targeted ACI against an episode target is arithmetic, not evidence:
      - aci      : window-level target alpha = delta (its usual planner tuning);
                   its episode rate is then ~1-(1-delta)^W by construction, and
                   the SAME arithmetic applies to any per-window method (incl.
                   our own CS -- reported here as cs_episode_viol).
      - aci_ep   : episode-targeted alpha = delta/W (Bonferroni), the fair
                   episode-level ACI arm.  It lands near delta on i.i.d. scores;
                   the substantive gap is that it has no finite-sample guarantee
                   at ANY tuning, which the closed-loop regimes then probe.

Run: .venv/bin/python code/exp1_validity.py    (~1-2 min, all seeded)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csq import (ACI, ACIEpisode, AnytimeUnionCP, BudgetedAUC, UnionBoundCP,
                 cs_boundary, split_cp_quantile)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "exp1_validity.json")
CACHE = os.path.join(ROOT, "results", "cache")

T = 5000
NRUN = 20000
W = 8

res = {"config": {"T": T, "n_runs": NRUN, "windows_per_episode": W, "seed": 123},
       "cs_time_uniform": [], "plugin_time_uniform": [], "episode_level": []}
rng = np.random.default_rng(123)

# ---------------------------------------------------- (a) time-uniform coverage
for beta in [0.9, 0.95]:
    for delta in [0.05, 0.1]:
        u_bnd = cs_boundary(T, beta, delta, cache_dir=CACHE)
        U = rng.random((NRUN, T))
        Z = np.cumsum(U < beta, axis=1)
        ever = (Z >= u_bnd[None, :]).any(axis=1)
        # cumulative profile for the figure
        firsts = np.where(ever, (Z >= u_bnd[None, :]).argmax(axis=1), T + 1)
        ts = np.unique(np.round(np.geomspace(10, T, 40)).astype(int))
        profile = [(int(t), float((firsts < t).mean())) for t in ts]
        res["cs_time_uniform"].append({
            "beta": beta, "delta": delta,
            "anytime_undercoverage_rate": float(ever.mean()),
            "mc_3sigma": float(3 * np.sqrt(delta * (1 - delta) / NRUN)),
            "profile_t_rate": profile})
        # naive online plug-in split-CP at level beta (recomputed each step):
        # undercoverage event r_t < q_beta  <=>  #(first t samples < beta) < k_t
        n1 = np.arange(1, T + 1)
        k_t = np.ceil((n1 + 1) * beta)  # order statistic index used at time t
        valid = k_t <= n1  # radius finite
        # r_t = s_(k_t); r_t < q_beta  <=>  at least k_t of first t samples < q_beta
        under = (Z >= k_t[None, :]) & valid[None, :]
        ever_pi = under.any(axis=1)
        res["plugin_time_uniform"].append({
            "beta": beta, "delta_nominal": delta,
            "anytime_undercoverage_rate": float(ever_pi.mean())})
        print(f"beta={beta} delta={delta}: CS anytime-undercov "
              f"{ever.mean():.4f} (<= {delta}); plug-in {ever_pi.mean():.4f}")

# ------------------------------------------- (b) episode-level guarantees, iid
NEP = 20000
for delta in [0.05, 0.1]:
    rng2 = np.random.default_rng(1000 + int(delta * 100))
    warm = rng2.standard_normal(2000)

    def _episode_rate(make, n_ep, w=W, aci_like=False):
        """Fraction of episodes with at least one violated window."""
        c = make()
        bad = 0
        for _ in range(n_ep):
            if hasattr(c, "new_episode"):
                c.new_episode()
            v = False
            for _w in range(w):
                r = c.radius()
                s = rng2.standard_normal()
                # uncertified (r = +inf) is not a violation
                err = 1.0 if (np.isfinite(r) and s > r) else 0.0
                v |= bool(err)
                if aci_like:
                    c.update_err(err)
                c.add(s)
            bad += v
        return bad / n_ep

    auc_r = _episode_rate(lambda: AnytimeUnionCP(delta, warm), NEP)
    aucb_r = _episode_rate(lambda: BudgetedAUC(delta, warm, lam=0.05), NEP)
    # Union CP: fixed calibration, fresh per run to include calibration randomness
    bad_u = 0
    for e in range(NEP):
        ub = UnionBoundCP(delta, W, rng2.standard_normal(2000))
        bad_u += bool((rng2.standard_normal(W) > ub.radius()).any())
    # ACI, window-targeted (its planner tuning) and episode-targeted (fair arm)
    aci = ACI(delta, warm, gamma=0.01)
    bad_a, err_a = 0, 0
    for e in range(NEP // 4):
        v = False
        for w in range(W):
            r = aci.radius()
            s = rng2.standard_normal()
            err = 1.0 if (np.isfinite(r) and s > r) else 0.0
            v |= bool(err)
            err_a += err
            aci.update_err(err)
            aci.add(s)
        bad_a += v
    acie = ACIEpisode(delta, W, warm, gamma=0.01)
    bad_ae = 0
    for e in range(NEP // 4):
        v = False
        for w in range(W):
            r = acie.radius()
            s = rng2.standard_normal()
            err = 1.0 if (np.isfinite(r) and s > r) else 0.0
            v |= bool(err)
            acie.update_err(err)
            acie.add(s)
        bad_ae += v
    # the CS is a PER-WINDOW method: its episode rate compounds the same way ACI's
    # does.  Reported so that the ACI comparison cannot be read as special pleading.
    from csq import QuantileCS
    cs_r = _episode_rate(lambda: QuantileCS(1 - delta, delta, warm, t_max=NEP * W + 3000,
                                            cache_dir=CACHE), NEP // 4)
    res["episode_level"].append({
        "delta": delta,
        "auc_episode_viol": auc_r,
        "aucb_episode_viol": aucb_r,
        "union_episode_viol": bad_u / NEP,
        "aci_episode_viol": bad_a / (NEP // 4),
        "aci_window_err": err_a / (NEP // 4 * W),
        "aci_ep_episode_viol": bad_ae / (NEP // 4),
        "cs_episode_viol": cs_r,
        "compounding_ceiling_1m(1md)^W": 1 - (1 - delta) ** W,
        "mc_3sigma": float(3 * np.sqrt(delta * (1 - delta) / NEP))})
    print(f"delta={delta}: AUC ep {auc_r:.4f}, AUC-B ep {aucb_r:.4f}, "
          f"union ep {bad_u/NEP:.4f}, ACI(window-target) ep {bad_a/(NEP//4):.4f} "
          f"(window err {err_a/(NEP//4*W):.4f}), ACI(episode-target) ep "
          f"{bad_ae/(NEP//4):.4f}, CS(per-window) ep {cs_r:.4f}, "
          f"compounding ceiling {1-(1-delta)**W:.4f}")

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("exp1 complete ->", OUT)
