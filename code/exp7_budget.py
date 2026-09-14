"""exp7: guarantee G2 -- an anytime-valid bound on the cumulative violation count.

Theorem 1 certifies a per-window risk eps uniformly over all replans, and Props.
1-2 show that an *episode-level* certificate ceases to exist past a horizon fixed
by the calibration budget.  Between the two lies the quantity a long deployment
actually cares about: how many certified regions have failed so far.  The
ViolationBudgetCS of Sec. III-C bounds it at every horizon, from the SAME
precomputed boundary array that produces the radius:

    P( exists W >= 1 : V_W > B_W )  <=  delta + delta',   B_W = u_W(eps,delta') - 1.

Parts:
  (a) synthetic anytime validity of B_W (worst case: violations exactly
      Bernoulli(eps)), plus the tightness ratio B_W / (eps W);
  (b) the budget applied to the recorded long-horizon deployments in
      results/long/*.json: realised V_W vs B_W per method, and the fraction of
      episodes that ever exceed the displayed envelope.  Contract membership is
      labelled from method semantics, never inferred from observed finite-radius
      fractions.  Infinite G3 regions are valid but operationally uninformative.

Run:  .venv/bin/python code/exp7_budget.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csq import violation_budget_curve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = os.path.join(ROOT, "results", "long")
CACHE = os.path.join(ROOT, "results", "cache")
OUT = os.path.join(ROOT, "results", "exp7_budget.json")

W_MAX = 3000
R_RUNS = 20000
G3_METHODS = {"auc", "aucb", "union"}
G2_METHODS = {"cs", "cs_r"}


def contract_label(method):
    if method in G2_METHODS:
        return "G1_quantile_plus_G2_count"
    if method in G3_METHODS:
        return "G3_episode_any_failure"
    if method == "aci_ep":
        return "episode_targeted_ACI_without_finite_sample_G3"
    if method == "aci":
        return "average_coverage_baseline_without_G1_G2_G3"
    if method == "gauss":
        return "parametric_baseline_without_distribution_free_G1_G2_G3"
    return "no_G1_G2_G3_contract"


def synth_validity():
    """Worst-case check: Yhat_w iid Bernoulli(eps) sits exactly on the H0 boundary,
    so this is the tightest possible test of P(exists W: V_W > B_W) <= delta'."""
    rng = np.random.default_rng(20260905)
    rows = []
    for eps in (0.05, 0.10):
        for dp in (0.05, 0.10):
            B = violation_budget_curve(eps, dp, W_MAX, cache_dir=CACHE)
            fails = np.zeros(R_RUNS, dtype=bool)
            for chunk in range(0, R_RUNS, 2000):
                m = min(2000, R_RUNS - chunk)
                V = np.cumsum(rng.random((m, W_MAX)) < eps, axis=1)
                fails[chunk:chunk + m] = (V > B[None, :]).any(axis=1)
            rows.append({
                "eps": eps, "delta_prime": dp,
                "anytime_exceed_rate": float(fails.mean()),
                "target": dp,
                "valid": bool(fails.mean() <= dp),
                "B": {str(W): int(B[W - 1]) for W in (1, 8, 30, 60, 120, 500, 1000, 3000)},
                "B_over_epsW": {str(W): float(B[W - 1] / (eps * W))
                                for W in (8, 30, 60, 120, 500, 1000, 3000)},
            })
            print(f"  eps={eps} delta'={dp}: exceed {fails.mean():.4f} (target {dp})"
                  f"  B_120={B[119]} (eps*120={eps*120:.1f})")
    return rows


def deployment_budget():
    """Replay the recorded long-horizon window traces through the budget."""
    out = {}
    for fn in sorted(os.listdir(LONG)):
        if not fn.endswith(".json"):
            continue
        stem, dstr = fn[:-5].rsplit("_", 1)
        delta = float(dstr)
        method = next((m for m in ("aci_ep", "aucb", "auc", "cs", "union", "aci",
                                   "gauss", "uncal") if stem.endswith("_" + m)), None)
        if method is None:
            continue
        regime = stem[: -(len(method) + 1)]
        if method == "uncal":
            continue
        d = json.load(open(os.path.join(LONG, fn)))
        recs = d.get("records", [])
        if not recs:
            continue
        eps = delta
        Wmax = max(len(r["windows"]) for r in recs)
        B = violation_budget_curve(eps, delta, Wmax + 1, cache_dir=CACHE)
        ever, finalV, finalB, wstars, cert_frac = [], [], [], [], []
        for r in recs:
            V = 0
            exceeded = False
            wstar = None
            for w, (s, rad) in enumerate(r["windows"], start=1):
                if rad is None:                       # no usable finite radius
                    if wstar is None:
                        wstar = w
                    continue
                V += int(s > rad)
                if V > B[w - 1]:
                    exceeded = True
            ever.append(exceeded)
            finalV.append(V)
            finalB.append(int(B[len(r["windows"]) - 1]))
            wstars.append(wstar)
            nw = len(r["windows"])
            cert_frac.append(sum(x[1] is not None for x in r["windows"]) / max(nw, 1))
        out[f"{regime}|{method}|{delta}"] = {
            "n_episodes": len(recs),
            "cert_frac": float(np.mean(cert_frac)),
            "mean_violations": float(np.mean(finalV)),
            "budget_at_end": int(np.median(finalB)),
            "frac_episodes_budget_exceeded": float(np.mean(ever)),
            "median_first_uncertified_window": (
                None if all(x is None for x in wstars)
                else float(np.median([x for x in wstars if x is not None]))),
            "contract_label": contract_label(method),
            "G2_contract_under_stated_iid_assumptions": method in G2_METHODS,
            "G3_contract_under_stated_exchangeability_assumptions": method in G3_METHODS,
            "finite_radius_throughout_all_records": bool(np.mean(cert_frac) == 1.0),
            "assumptions_established_by_this_legacy_artifact": False,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reuse-synthetic", action="store_true",
        help="reuse the existing synthetic rows while refreshing semantic labels")
    args = ap.parse_args()
    print("[exp7a] synthetic anytime validity of the violation budget")
    prior_hash = None
    if args.reuse_synthetic:
        raw = open(OUT, "rb").read()
        prior_hash = hashlib.sha256(raw).hexdigest()
        synth = json.loads(raw)["synthetic"]
        print("  reused existing synthetic rows; no simulation rerun")
    else:
        synth = synth_validity()
    print("[exp7b] budget on recorded long-horizon deployments")
    dep = deployment_budget()
    for k, v in sorted(dep.items()):
        if abs(float(k.split("|")[2]) - 0.05) < 1e-9:
            print(f"  {k:38s} cert={v['cert_frac']:.2f} V={v['mean_violations']:5.2f}"
                  f" B={v['budget_at_end']:3d} exceed={v['frac_episodes_budget_exceeded']:.2f}")
    payload = {
        "artifact_schema_version": 2,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "purpose": "G2_count_diagnostic_with_contract_semantics",
        "source_execution_contract": "legacy_fresh_prediction_recentered_v0",
        "warning": (
            "Replay dependence means this artifact does not establish iid or "
            "exchangeability assumptions; planning outcomes use the legacy controller."
        ),
        "synthetic_rows_reused_from_prior_sha256": prior_hash,
        "synthetic": synth,
        "deployment": dep,
    }
    json.dump(payload, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
