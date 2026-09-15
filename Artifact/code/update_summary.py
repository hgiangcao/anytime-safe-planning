"""Fold the revision's experiments (exp7-exp10) into results/summary.json."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa
from csq import violation_budget_curve, cert_horizon_fixed_T, cert_horizon_lookahead

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
S = json.load(open(os.path.join(R, "summary.json")))
e7 = json.load(open(os.path.join(R, "exp7_budget.json")))
e8 = json.load(open(os.path.join(R, "exp8_robust.json")))
e9 = json.load(open(os.path.join(R, "exp9_matched.json")))
e10 = json.load(open(os.path.join(R, "exp10_vehicle.json")))
B = violation_budget_curve(0.05, 0.05, 2000, cache_dir=os.path.join(R, "cache"))

tot = sum(len(json.load(open(f)).get("records", []))
          for d in ("main", "long", "matched", "vehicle")
          for f in glob.glob(os.path.join(R, d, "*.json")))
cond = sum(len(glob.glob(os.path.join(R, d, "*.json")))
           for d in ("main", "long", "matched", "vehicle"))

S["headline"].pop("violation_budget_G3", None)
S["headline"]["violation_budget_G2"] = {
    "what": "Theorem 2 / G2: for the CS under its stated iid assumptions, B_W is a "
            "time-uniform bound on the cumulative number of violated windows. Other "
            "methods are descriptive traces against the same envelope, not G2 claims. "
            "No legacy replay artifact establishes iid or exchangeability assumptions.",
    "B_at": {"8": int(B[7]), "120": int(B[119]), "1000": int(B[999])},
    "synthetic_anytime_exceed": {f"eps{r['eps']}_dp{r['delta_prime']}":
                                 r["anytime_exceed_rate"] for r in e7["synthetic"]},
    "deployment": {k: {kk: v[kk] for kk in
                       ("cert_frac", "mean_violations", "budget_at_end",
                        "frac_episodes_budget_exceeded", "contract_label",
                        "G2_contract_under_stated_iid_assumptions",
                        "G3_contract_under_stated_exchangeability_assumptions",
                        "assumptions_established_by_this_legacy_artifact")}
                   for k, v in e7["deployment"].items() if k.endswith("|0.05")},
}
S["headline"]["shift_robust_CS"] = {
    "what": "Prop. 4: under a TOTAL shift budget sum_t (p_t-beta)_+ <= rho the CS stays "
            "time-uniformly valid at threshold log(1/delta) + rho/beta -- a CONSTANT "
            "degradation, equivalently delta -> delta*exp(-rho/beta).",
    "synthetic_validity": e8["validity"],
    "legacy_marginal_proxy_per_120_windows": {f"{r['regime']}_{r['delta']}":
                                               r["budget_over_120_windows"]
                                               for r in e8["measured_budget"]},
    "rho_used": 12.0,
    "in_loop": {k: {"window_viol_rate": v["window_viol_rate"],
                    "mean_radius": v["mean_radius"], "success_rate": v["success_rate"]}
                for k, v in e10["conditions"].items() if k.startswith("rob_")},
}
S["headline"].pop("prop1_tested_against_optimised_allocation", None)
S["headline"]["prop1_rank_wall_and_allocation_grid_heuristic"] = e9["prop1_test"]
S["headline"]["lookahead_wall"] = e9["lookahead_wall"]
S["headline"]["matched_calibration"] = {
    "what": "LEGACY RECENTERED CONTROLLER. Radii are cross-contract descriptive context; "
            "throughput, fallback, and collision fields are not evidence for window_anchor_v1.",
    "pool_n": e9["matched_pool_n"], "wall": e9["matched_wall"],
    "per_method": {k: {kk: v[kk] for kk in
                       ("cert_window_frac", "window_viol_rate", "mean_radius",
                        "mean_goals", "coll_step_frac", "fallback_step_frac")}
                   for k, v in e9["matched"].items()},
}
S["headline"]["harder_vehicle"] = {
    "what": "LEGACY RECENTERED CONTROLLER with stale latency and unbounded Gaussian noise; "
            "empirical provenance only, with no containment-to-collision certificate.",
    "per_condition": {k: {kk: v[kk] for kk in
                          ("window_viol_rate", "success_rate", "mean_radius",
                           "coll_step_frac", "fallback_step_frac")}
                      for k, v in e10["conditions"].items() if k.startswith("uni_")},
}
S["headline"]["totals"] = {"episodes": tot, "conditions": cond}
S["revision_status_2026_09_07"] = {
    "execution_contract": "legacy_fresh_prediction_recentered_v0",
    "warning": "This historical aggregate retains old planning fields for provenance. "
               "Do not use them as evidence for window_anchor_v1; see the paper, "
               "PROVENANCE.md, and EXTERNAL_REVIEW_RESPONSE.md."
}
S["caveats"] = [
    c for c in S["caveats"]
    if "robust e-value mixing" not in c.lower()
    and "optimised-allocation baseline" not in c.lower()
] + [
    "Robust e-value mixing IS now implemented (Prop. 4 / RobustQuantileCS / cs_r arm); the "
    "The robust theorem requires an externally justified conditional pathwise shift budget. "
    "Recorded marginal frequencies do not certify it; the held-out point diagnostic succeeds "
    "in 9/10 cases and fails for Zara1 at delta=0.1.",
    "The Prop. 1 rank wall is analytic and independent of radius optimisation. The released "
    "allocation routine is only a Lagrangian/grid heuristic: empirical quantile curves are "
    "nonconvex steps, so its 29-70% heterogeneous-window reductions are attainable examples, "
    "not certified global optima.",
    "The archived unicycle study used stale/recentered predictions and unbounded Gaussian "
    "actuation noise. It is empirical provenance, not revised-controller or hardware evidence.",
    "G2 is the CS cumulative-violation budget. G3 methods use an episode any-failure contract; "
    "when their radius is infinite it is valid but operationally uninformative.",
]
json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)
print("summary.json updated;", tot, "episodes across", cond, "conditions")
