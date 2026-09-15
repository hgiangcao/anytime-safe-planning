"""Integrity-check stored statistics, including provenance-only legacy fields.

Passing a legacy planning check means only that the summary matches the stored
bytes. It does not promote a recentered-controller throughput/collision value
to evidence for ``window_anchor_v1``. Current claims and residual limitations
are enumerated in ``EXTERNAL_REVIEW_RESPONSE.md``.
"""
import json, os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa
from csq import (violation_budget_curve, cert_horizon_fixed_T,
                 cert_horizon_lookahead, cs_boundary)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ok, bad = 0, []
def chk(name, got, want, tol=1e-9):
    global ok
    g = float(got); w = float(want)
    if abs(g - w) <= tol * max(1.0, abs(w)): ok += 1; print(f"  OK   {name}: {g}")
    else: bad.append(f"{name}: expected={w} data={g}")

B = violation_budget_curve(0.05, 0.05, 2000, cache_dir=os.path.join(R,"cache"))
chk("B_8", B[7], 2); chk("B_120", B[119], 15); chk("B_1000", B[999], 77)
B10 = violation_budget_curve(0.10, 0.10, 200, cache_dir=os.path.join(R,"cache"))

e7 = json.load(open(os.path.join(R, "exp7_budget.json")))
sv = {(r["eps"], r["delta_prime"]): r["anytime_exceed_rate"] for r in e7["synthetic"]}
chk("synth 0.05/0.05", round(sv[(0.05,0.05)],3), 0.021, 1e-6)
chk("synth 0.05/0.1",  round(sv[(0.05,0.10)],3), 0.073, 1e-6)
chk("synth 0.1/0.05",  round(sv[(0.10,0.05)],3), 0.030, 1e-6)
chk("synth 0.1/0.1",   round(sv[(0.10,0.10)],3), 0.060, 1e-6)
dp = e7["deployment"]
chk("replay V_120", round(dp["long_replay|cs|0.05"]["mean_violations"],1), 3.9, 1e-6)
chk("sf V_120",     round(dp["long_sf|cs|0.05"]["mean_violations"],1), 4.1, 1e-6)
chk("real V_60",    round(dp["long_real|cs|0.05"]["mean_violations"],1), 1.5, 1e-6)
chk("real cs exceed", round(dp["long_real|cs|0.05"]["frac_episodes_budget_exceeded"],2), 0.07, 1e-6)
chk("real aci exceed",round(dp["long_real|aci|0.05"]["frac_episodes_budget_exceeded"],2), 0.14, 1e-6)
chk("auc cert",  round(dp["long_replay|auc|0.05"]["cert_frac"],2), 0.08, 1e-6)
chk("aucb cert", round(dp["long_replay|aucb|0.05"]["cert_frac"],2), 0.43, 1e-6)

e8 = json.load(open(os.path.join(R, "exp8_robust.json")))
va = {r["scenario"]: r for r in e8["validity"]}
chk("plain persistent", round(va["persistent_0.01"]["plain_cs_fail"],2), 0.51, 1e-6)
chk("plain transient",  round(va["transient_0.04_first500"]["plain_cs_fail"],4), 0.9996, 1e-6)
chk("rob persistent",   round(va["persistent_0.01"]["robust_cs_fail"],3), 0.000, 1e-6)
chk("rob transient",    round(va["transient_0.04_first500"]["robust_cs_fail"],3), 0.001, 1e-6)
mb = {(r["regime"], r["delta"]): r for r in e8["measured_budget"]}
worst = max(r["budget_over_120_windows"] for r in e8["measured_budget"])
chk("max measured rho", round(worst), 12, 1e-6)
chk("sf rho lo", round(mb[("sf",0.05)]["budget_over_120_windows"],1), 1.1, 1e-6)
chk("sf rho hi", round(mb[("sf",0.10)]["budget_over_120_windows"],1), 1.5, 1e-6)

e9 = json.load(open(os.path.join(R, "exp9_matched.json")))
print("  NOTE legacy planning fields below are integrity-only, not paper claims")
chk("matched n", e9["matched_pool_n"], 2597); chk("matched wall", e9["matched_wall"], 129)
m = e9["matched"]
chk("cs radius", round(m["cs"]["mean_radius"],3), 1.711, 1e-6)
chk("union radius", round(m["union"]["mean_radius"],3), 2.746, 1e-6)
chk("aucb radius", round(m["aucb"]["mean_radius"],3), 2.663, 1e-6)
chk("cs goals", round(m["cs"]["mean_goals"],1), 17.8, 1e-6)
chk("union goals", round(m["union"]["mean_goals"],1), 7.0, 1e-6)
chk("aucb goals", round(m["aucb"]["mean_goals"],1), 4.7, 1e-6)
chk("cs coll/1k", round(1000*m["cs"]["coll_step_frac"],1), 2.0, 1e-6)
chk("union coll/1k", round(1000*m["union"]["coll_step_frac"],1), 3.7, 1e-6)
chk("aci_ep coll/1k", round(1000*m["aci_ep"]["coll_step_frac"],1), 3.6, 1e-6)
chk("aucb coll/1k", round(1000*m["aucb"]["coll_step_frac"],1), 3.2, 1e-6)
chk("cs fallback", round(m["cs"]["fallback_step_frac"],2), 0.06, 1e-6)
chk("union fallback", round(m["union"]["fallback_step_frac"],2), 0.32, 1e-6)
chk("aci_ep cert", round(m["aci_ep"]["cert_window_frac"],2), 0.96, 1e-6)
chk("aucb cert", round(m["aucb"]["cert_window_frac"],2), 0.47, 1e-6)
chk("radius reduction %", round(100*(1-m["cs"]["mean_radius"]/m["union"]["mean_radius"])), 38, 1e-6)
chk("throughput ratio", round(m["cs"]["mean_goals"]/m["union"]["mean_goals"],1), 2.5, 1e-6)
p1 = e9["prop1_test"]["homogeneous_wall"]
assert all(r["match"] for r in p1), "Prop1 mismatch"
print("  OK   Prop.1 analytic wall matches feasibility check in all", len(p1), "cases:",
      sorted({r["heuristic_feasible_max_T"] for r in p1}))
het = [h for h in e9["prop1_test"]["heterogeneous_control"] if h["feasible"]]
g = [round(h["radius_gain_pct"]) for h in het if h["radius_gain_pct"] > 1]
chk("het gain lo", min(g), 29, 1e-6); chk("het gain hi", max(g), 70, 1e-6)
lw = e9["lookahead_wall"][0]
chk("lookahead wall", lw["wall_per_lookahead"], 13); chk("scalar wall", lw["wall_scalar"], 104)

e10 = json.load(open(os.path.join(R, "exp10_vehicle.json")))["conditions"]
chk("orca cs .1",  round(e10["rob_orca|cs|0.1"]["window_viol_rate"],3), 0.122, 1e-6)
chk("orca csr .1", round(e10["rob_orca|cs_r|0.1"]["window_viol_rate"],3), 0.089, 1e-6)
chk("real cs .1",  round(e10["rob_real|cs|0.1"]["window_viol_rate"],3), 0.129, 1e-6)
chk("real csr .1", round(e10["rob_real|cs_r|0.1"]["window_viol_rate"],3), 0.073, 1e-6)
chk("orca cs .05", round(e10["rob_orca|cs|0.05"]["window_viol_rate"],3), 0.043, 1e-6)
chk("orca csr .05",round(e10["rob_orca|cs_r|0.05"]["window_viol_rate"],3), 0.031, 1e-6)
chk("real cs .05", round(e10["rob_real|cs|0.05"]["window_viol_rate"],3), 0.045, 1e-6)
chk("real csr .05",round(e10["rob_real|cs_r|0.05"]["window_viol_rate"],3), 0.016, 1e-6)
chk("orca r plain", round(e10["rob_orca|cs|0.1"]["mean_radius"],3), 1.586, 1e-6)
chk("orca r rob",   round(e10["rob_orca|cs_r|0.1"]["mean_radius"],3), 1.756, 1e-6)
chk("real r plain", round(e10["rob_real|cs|0.1"]["mean_radius"],3), 1.471, 1e-6)
chk("real r rob",   round(e10["rob_real|cs_r|0.1"]["mean_radius"],3), 1.680, 1e-6)
chk("orca prem %",  round(100*(e10["rob_orca|cs_r|0.1"]["mean_radius"]/e10["rob_orca|cs|0.1"]["mean_radius"]-1),1), 10.7, 1e-6)
chk("real prem %",  round(100*(e10["rob_real|cs_r|0.1"]["mean_radius"]/e10["rob_real|cs|0.1"]["mean_radius"]-1),1), 14.2, 1e-6)
chk("uni sf cs succ",   round(100*e10["uni_sf|cs|0.05"]["success_rate"],1), 46.7, 1e-6)
chk("uni sf union succ",round(100*e10["uni_sf|union|0.05"]["success_rate"],1), 30.7, 1e-6)
chk("uni sf aucb succ", round(100*e10["uni_sf|aucb|0.05"]["success_rate"],1), 21.3, 1e-6)
chk("uni rp cs succ",   round(100*e10["uni_replay|cs|0.05"]["success_rate"],1), 39.3, 1e-6)
chk("uni rp union succ",round(100*e10["uni_replay|union|0.05"]["success_rate"],1), 16.0, 1e-6)
chk("uni rp aucb succ", round(100*e10["uni_replay|aucb|0.05"]["success_rate"],1), 3.3, 1e-6)
chk("uni rp aucb fb",   round(100*e10["uni_replay|aucb|0.05"]["fallback_step_frac"],1), 58.5, 1e-6)
chk("uni sf cs viol",   round(e10["uni_sf|cs|0.05"]["window_viol_rate"],3), 0.034, 1e-6)
chk("uni rp cs viol",   round(e10["uni_replay|cs|0.05"]["window_viol_rate"],3), 0.034, 1e-6)
chk("uni sf cs coll",   round(1000*e10["uni_sf|cs|0.05"]["coll_step_frac"],1), 3.5, 1e-6)
chk("uni rp cs coll",   round(1000*e10["uni_replay|cs|0.05"]["coll_step_frac"],1), 5.1, 1e-6)
chk("uni sf uncal coll",round(1000*e10["uni_sf|uncal|0.05"]["coll_step_frac"],1), 27.3, 1e-6)
chk("uni rp uncal coll",round(1000*e10["uni_replay|uncal|0.05"]["coll_step_frac"],1), 51.3, 1e-6)

# Uniform-mixture rate audit.  The displayed uniform mixture pays sqrt(log t/t),
# not the LIL-optimal sqrt(log log t/t): the log-rate-normalised rank gap stays
# roughly constant over four decades.  This is an implementation check, not an
# asymptotic proof.
u = cs_boundary(2_000_000, 0.95, 0.05, cache_dir=os.path.join(R,"cache"))
ts = np.arange(1, 2_000_001)
log_scaled = (u/ts - 0.95) * np.sqrt(ts / np.log(np.maximum(ts, 2)))
if not (0.20 <= log_scaled[9_999] <= 0.26 and 0.20 <= log_scaled[-1] <= 0.26):
    bad.append(("uniform-mixture log-rate", log_scaled[9_999], log_scaled[-1]))
else:
    ok += 1

# episode totals
import glob
tot = sum(len(json.load(open(f)).get("records",[]))
          for d in ("main","long","matched","vehicle") for f in glob.glob(os.path.join(R,d,"*.json")))
cond = sum(len(glob.glob(os.path.join(R,d,"*.json"))) for d in ("main","long","matched","vehicle"))
chk("total episodes", tot, 15758); chk("total conditions", cond, 217)

print(f"\n{ok} checks OK, {len(bad)} MISMATCH")
for b in bad: print("  MISMATCH", b)
sys.exit(1 if bad else 0)
