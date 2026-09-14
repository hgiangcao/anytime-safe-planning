"""Dependence-aware audit of the empirical shift-budget diagnostic.

Proposition 4 assumes a pathwise budget on conditional probabilities,

    sum_t (P(s_t < q_beta | F_{t-1}) - beta)_+ <= rho.

Recorded score labels do not identify those conditional probabilities.  This
script therefore makes no theorem-level claim.  It estimates the *marginal*
below-quantile frequency, preserves every episode as a cluster when bootstrapping
uncertainty, and reports leave-one-scene-out transfer including failures.

Run:  ../.venv/bin/python code/exp12_rho_loso.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csq import split_cp_quantile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
MAIN = os.path.join(ROOT, "results", "main")
OUT = os.path.join(ROOT, "results", "exp12_rho_loso.json")
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
SIM = [("replay", "cal_sf.npy"), ("sf", "cal_sf.npy"),
       ("orca", "cal_orca.npy")]
BOOTSTRAPS = 10000


def empirical_budget(regime, pool, delta, windows=120):
    """Episode-cluster bootstrap for the marginal below-quantile diagnostic."""
    cal = np.sort(np.load(os.path.join(DATA, pool)))
    beta = 1.0 - delta
    q = float(split_cp_quantile(cal, beta))
    path = os.path.join(MAIN, f"{regime}_cs_{delta}.json")
    if not os.path.exists(path):
        return None
    records = json.load(open(path))["records"]
    clusters = []
    for record in records:
        scores = np.asarray([w[0] for w in record["windows"]], dtype=float)
        if scores.size:
            clusters.append((int(np.sum(scores < q)), int(scores.size)))
    if not clusters:
        return None
    below = np.asarray([x[0] for x in clusters], dtype=float)
    counts = np.asarray([x[1] for x in clusters], dtype=float)
    p_hat = float(below.sum() / counts.sum())
    episode_rates = below / counts
    p_episode = float(episode_rates.mean())
    rng = np.random.default_rng(zlib.crc32(f"rho|{regime}|{delta}".encode()))
    # Resample whole episodes; windows within an episode are never treated as
    # independent observations.
    draw = rng.integers(0, len(clusters), size=(BOOTSTRAPS, len(clusters)))
    p_boot = below[draw].sum(axis=1) / counts[draw].sum(axis=1)
    p_lo, p_hi = np.quantile(p_boot, [0.025, 0.975])
    # A nonparametric bootstrap is degenerate when every observed cluster is all
    # below q.  A bounded-cluster Hoeffding interval remains nonzero and is the
    # conservative dependence-aware uncertainty summary used for transfer.
    hoeffding_half = float(np.sqrt(np.log(2.0 / 0.05) / (2.0 * len(clusters))))
    h_lo = max(0.0, p_episode - hoeffding_half)
    h_hi = min(1.0, p_episode + hoeffding_half)

    def rho(p):
        return float(windows * max(float(p) - beta, 0.0))

    return {
        "regime": regime,
        "delta": delta,
        "beta": beta,
        "reference_quantile": q,
        "n_episode_clusters": len(clusters),
        "n_windows": int(counts.sum()),
        "p_hat_marginal": p_hat,
        "p_hat_cluster_bootstrap_ci95": [float(p_lo), float(p_hi)],
        "p_hat_episode_mean": p_episode,
        "p_hat_episode_hoeffding_ci95": [h_lo, h_hi],
        "rho_hat_over_120_windows": rho(p_hat),
        "rho_cluster_bootstrap_ci95": [rho(p_lo), rho(p_hi)],
        "rho_episode_hoeffding_ci95": [rho(h_lo), rho(h_hi)],
        "formal_conditional_shift_budget": False,
    }


def main():
    estimates = []
    folds = []
    for delta in (0.05, 0.10):
        per = {r: empirical_budget(r, p, delta) for r, p in SIM}
        for scene in SCENES:
            key = f"real_{scene}"
            per[key] = empirical_budget(key, f"cal_real_{scene}.npy", delta)
        per = {k: v for k, v in per.items() if v is not None}
        estimates.extend(per.values())
        for scene in SCENES:
            held = f"real_{scene}"
            held_est = per[held]
            others = [v for k, v in per.items() if k != held]
            source_point = max(v["rho_hat_over_120_windows"] for v in others)
            source_hi = max(v["rho_episode_hoeffding_ci95"][1] for v in others)
            held_point = held_est["rho_hat_over_120_windows"]
            held_hi = held_est["rho_episode_hoeffding_ci95"][1]
            row = {
                "delta": delta,
                "held_out": held,
                "rho_from_other_regimes_point": source_point,
                "rho_from_other_regimes_episode_hoeffding_hi95": source_hi,
                "rho_needed_by_held_out_point": held_point,
                "rho_needed_by_held_out_episode_hoeffding_ci95":
                    held_est["rho_episode_hoeffding_ci95"],
                "point_transfer_covered": bool(source_point >= held_point),
                "source_hi95_covers_held_point": bool(source_hi >= held_point),
                "source_hi95_covers_held_hi95": bool(source_hi >= held_hi),
            }
            folds.append(row)
            print(
                f"delta={delta:.2f} held {held:10s}: source point {source_point:5.2f} "
                f"[upper {source_hi:5.2f}], held {held_point:5.2f} "
                f"[upper {held_hi:5.2f}] -> "
                f"{'covers' if row['point_transfer_covered'] else 'POINT FAILURE'}")

    n_point = sum(r["point_transfer_covered"] for r in folds)
    n_hi_point = sum(r["source_hi95_covers_held_point"] for r in folds)
    n_hi_hi = sum(r["source_hi95_covers_held_hi95"] for r in folds)
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "source_execution_contract": "legacy_fresh_prediction_recentered_v0",
        "estimand": (
            "episode-clustered marginal frequency of s<q_beta, scaled to 120 windows; "
            "this is not the theorem's pathwise sum of conditional probabilities"),
        "formal_conditional_shift_budget_certified": False,
        "uncertainty": {
            "cluster_bootstrap": {"unit": "episode", "replicates": BOOTSTRAPS,
                                  "seed_rule": "crc32('rho|regime|delta')",
                                  "interval": "percentile 95%"},
            "transfer_interval": (
                "95% Hoeffding interval over bounded episode means; conservative "
                "and non-degenerate when all observed windows are below q_beta"),
        },
        "estimates": estimates,
        "folds": folds,
        "point_transfers_covered": n_point,
        "source_hi95_covers_held_point": n_hi_point,
        "source_hi95_covers_held_hi95": n_hi_hi,
        "n_total": len(folds),
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\npoint transfer: {n_point}/{len(folds)}; wrote {OUT}")


if __name__ == "__main__":
    main()
