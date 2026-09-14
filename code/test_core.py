"""Unit tests for the statistical/optimization core.  Run:
    .venv/bin/python code/test_core.py
Exits nonzero on failure.  Monte-Carlo assertions use 3-sigma slack.
"""
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from csq import (ACI, ACIEpisode, AlphaSpendingQuantileCS, AnytimeUnionCP,
                 BudgetedAUC, QuantileCS, alpha_spending_rank,
                 RayleighCC, UnionBoundCP, cert_horizon_fixed_T,
                 cert_horizon_minimal, cert_horizon_schedule, cs_boundary,
                 harmonic_spend,
                 log_evalue, split_cp_quantile)

FAIL = 0


def check(name, cond, msg=""):
    global FAIL
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL += 1
    print(f"[{status}] {name} {msg}")


def mc_slack(p, n):
    return 3.0 * np.sqrt(max(p * (1 - p), 1e-6) / n)


# 1. e-value: EXACT expectation = 1 at the null (sum over binomial pmf), and
#    monotone in Z (over the numerically representable range)
from scipy.stats import binom
rng = np.random.default_rng(0)
beta = 0.9
for t in [20, 200, 2000]:
    zs = np.arange(0, t + 1)
    lv = log_evalue(zs, t, beta)
    pmf = binom.pmf(zs, t, beta)
    finite = np.isfinite(lv)
    Em = float(np.sum(pmf[finite] * np.exp(np.minimum(lv[finite], 700))))
    check(f"evalue_exact_mean_t{t}", abs(Em - 1.0) < 1e-6, f"E[M]={Em:.8f}")
    # monotonicity in Z matters only near/above the Ville threshold log(1/delta)
    # (>= log(1/0.5) covers every delta <= 0.5); far below it, betainc's
    # underflow-transition region has ~1e-1 float noise at lv ~ -6, which never
    # affects the boundary bisection because the predicate is constant there.
    band = finite & (lv > np.log(1 / 0.5) - 2.0)
    dv = np.diff(lv[band])
    check(f"evalue_monotone_in_Z_t{t}", bool(np.all(dv > -1e-7)))

# The implemented uniform alternative mixture has a sqrt(log(t)/t) rank gap.
# This guards against reintroducing the incorrect LIL-rate claim.
ub_rate = cs_boundary(100000, 0.95, 0.05)
scaled = []
for tt in (10000, 100000):
    scaled.append((ub_rate[tt - 1] / tt - 0.95) * np.sqrt(tt / np.log(tt)))
check("uniform_mixture_log_rate", all(0.18 < x < 0.28 for x in scaled),
      f"sqrt(t/log t)-scaled gaps={scaled}")

# 2. Time-uniform coverage of the quantile CS (the paper's Theorem-1 event)
t0 = time.time()
T = 2000
NRUN = 5000
for beta, delta in [(0.9, 0.05), (0.9, 0.1), (0.95, 0.05)]:
    u_bnd = cs_boundary(T, beta, delta)
    U = rng.random((NRUN, T))
    Zc = np.cumsum(U < beta, axis=1)  # count strictly below true quantile (=beta)
    viol = (Zc >= u_bnd[None, :]).any(axis=1)
    rate = viol.mean()
    check(f"cs_time_uniform_b{beta}_d{delta}", rate <= delta + mc_slack(delta, NRUN),
          f"any-time undercoverage rate={rate:.4f} target<={delta}")
print(f"  (cs coverage tests {time.time()-t0:.1f}s)")

# 3. QuantileCS.radius agrees with brute-force boundary scan on a small stream
qcs = QuantileCS(0.9, 0.1, warm_scores=rng.normal(size=50), t_max=500)
for s in rng.normal(size=100):
    qcs.add(s)
tcur = qcs.t
lvals = log_evalue(np.arange(tcur + 1), tcur, 0.9)
us = np.where(lvals >= np.log(1 / 0.1))[0]
expect = np.inf if len(us) == 0 else qcs.sorted[us[0] - 1]
check("cs_radius_bruteforce", np.isclose(qcs.radius(), expect), f"{qcs.radius():.3f} vs {expect:.3f}")
qgrow = QuantileCS(0.9, 0.1, warm_scores=[-1.0, 0.0, 1.0], t_max=2)
qgrow.radius()
check("quantile_boundary_auto_extends", len(qgrow._u) >= qgrow.t,
      f"capacity={len(qgrow._u)} t={qgrow.t}")

# 3b. Standard exact-binomial alpha-spending baseline.  Its spending clock is
# the deployment query k, not the absolute sample count inside the warm pool.
for nn, kk in [(25, 1), (200, 2), (2094, 8)]:
    lev = 0.1 / (kk * (kk + 1.0))
    uu = alpha_spending_rank(nn, 0.9, lev)
    tail = float(binom.sf(uu - 1, nn, 0.9))
    prev = float(binom.sf(uu - 2, nn, 0.9)) if uu > int(np.ceil(.9 * nn)) else 1.0
    check(f"alpha_spending_exact_tail_n{nn}_k{kk}",
          tail <= lev + 1e-14 and (uu == int(np.ceil(.9 * nn)) or prev > lev),
          f"rank={uu} tail={tail:.3g} level={lev:.3g}")

warm_a = np.linspace(-2.0, 2.0, 200)
warm_b = np.linspace(10.0, 14.0, 200)
sp_a = AlphaSpendingQuantileCS(.9, .1, warm_a)
sp_b = AlphaSpendingQuantileCS(.9, .1, warm_b)
_ = sp_a.radius(); _ = sp_b.radius()
same_schedule = (sp_a.last_query_index == sp_b.last_query_index == 1
                 and sp_a.last_rank == sp_b.last_rank
                 and sp_a.last_alpha == sp_b.last_alpha)
sp_a.add(-100.0); sp_b.add(100.0)
_ = sp_a.radius(); _ = sp_b.radius()
same_schedule &= (sp_a.last_query_index == sp_b.last_query_index == 2
                  and sp_a.last_rank == sp_b.last_rank
                  and sp_a.last_alpha == sp_b.last_alpha)
check("alpha_spending_schedule_score_independent", same_schedule,
      f"k={sp_a.last_query_index} rank={sp_a.last_rank} alpha={sp_a.last_alpha}")

NRUN_SP, N0_SP, Q_SP = 4000, 200, 100
u_sp = np.array([
    alpha_spending_rank(N0_SP + k - 1, .9, .1 / (k * (k + 1.0)))
    for k in range(1, Q_SP + 1)])
draws_sp = rng.random((NRUN_SP, N0_SP + Q_SP - 1))
zc_sp = np.cumsum(draws_sp < .9, axis=1)[:, N0_SP - 1:]
bad_sp = (zc_sp >= u_sp[None, :]).any(axis=1)
rate_sp = float(bad_sp.mean())
check("alpha_spending_same_G1_time_uniform", rate_sp <= .1 + mc_slack(.1, NRUN_SP),
      f"queried-prefix undercoverage={rate_sp:.4f} target<=0.1")

# 4. split-CP marginal coverage
n = 999
NREP = 20000
cal = rng.normal(size=(NREP, n))
new = rng.normal(size=NREP)
for eps in [0.05, 0.1]:
    k = int(np.ceil((n + 1) * (1 - eps)))
    q = np.sort(cal, axis=1)[:, k - 1]
    r = (new > q).mean()
    check(f"splitcp_marginal_{eps}", r <= eps + mc_slack(eps, NREP), f"exceed={r:.4f}")

# 5. AUC per-episode validity (iid scores, accreting pool, W=8 windows)
NEP = 4000
W = 8
for delta in [0.05, 0.1]:
    rng2 = np.random.default_rng(7)
    auc = AnytimeUnionCP(delta, rng2.normal(size=2000))
    bad = 0
    for e in range(NEP):
        auc.new_episode()
        ep_viol = False
        for w in range(W):
            r = auc.radius()
            s = rng2.normal()
            if np.isfinite(r) and s > r:
                ep_viol = True
            auc.add(s)
        bad += ep_viol
    rate = bad / NEP
    check(f"auc_episode_valid_d{delta}", rate <= delta + mc_slack(delta, NEP),
          f"episode viol rate={rate:.4f}")

# 6. Union-bound CP per-episode validity (fixed calibration)
for delta in [0.05, 0.1]:
    rng3 = np.random.default_rng(11)
    bad = 0
    NEP2 = 4000
    for e in range(NEP2):
        ub = UnionBoundCP(delta, W, rng3.normal(size=2000))
        s = rng3.normal(size=W)
        bad += bool((s > ub.radius()).any())
    rate = bad / NEP2
    check(f"union_episode_valid_d{delta}", rate <= delta + mc_slack(delta, NEP2),
          f"episode viol rate={rate:.4f}")

# 7. Rayleigh CC recovers the right quantile on true Rayleigh scores
sig = 0.7
s = sig * np.sqrt(-2 * np.log(rng.random(200000)))
rc = RayleighCC(0.05, 8, s)
true_q = sig * np.sqrt(-2 * np.log(0.05 / 8))
check("rayleigh_quantile", abs(rc.radius() - true_q) / true_q < 0.02,
      f"{rc.radius():.3f} vs {true_q:.3f}")

# 8. ACI long-run coverage tracks the target
rng4 = np.random.default_rng(3)
aci = ACI(0.1, rng4.normal(size=500), gamma=0.01)
errs = []
for i in range(20000):
    r = aci.radius()
    s = rng4.normal()
    e = 1.0 if (not np.isfinite(r) or s > r) else 0.0
    errs.append(e)
    aci.update_err(e)
    aci.add(s)
rate = np.mean(errs[2000:])
check("aci_longrun", abs(rate - 0.1) < 0.02, f"err rate={rate:.4f}")

# 8b. BudgetedAUC: same episode-level guarantee, much longer certifiable horizon
for delta in [0.05, 0.1]:
    rngB = np.random.default_rng(23)
    warm = rngB.normal(size=2000)
    ab = BudgetedAUC(delta, warm, lam=0.05)
    NEPB, WB = 4000, 8
    bad, spent_max = 0, 0.0
    for e in range(NEPB):
        ab.new_episode()
        ep = False
        for w in range(WB):
            r = ab.radius()
            s = rngB.normal()
            if np.isfinite(r) and s > r:
                ep = True
            ab.add(s)
        spent_max = max(spent_max, delta - ab.d_rem)
        bad += ep
    rate = bad / NEPB
    check(f"aucb_episode_valid_d{delta}", rate <= delta + mc_slack(delta, NEPB),
          f"episode viol rate={rate:.4f}")
    check(f"aucb_budget_respected_d{delta}", spent_max <= delta + 1e-12,
          f"max spend={spent_max:.6f} <= {delta}")

# Episode spending is determined by sizes/index/remaining budget, never values.
ba = BudgetedAUC(.1, np.arange(200, dtype=float), lam=.05)
bb = BudgetedAUC(.1, np.arange(200, dtype=float) + 1000.0, lam=.05)
schedule_a, schedule_b = [], []
for j in range(8):
    _ = ba.radius(); _ = bb.radius()
    schedule_a.append((ba.w, ba.d_rem, ba.exhausted))
    schedule_b.append((bb.w, bb.d_rem, bb.exhausted))
    ba.add(float(j)); bb.add(float(1000 + j))
check("aucb_schedule_score_value_independent", schedule_a == schedule_b,
      f"terminal={schedule_a[-1]}")

# 8c. certifiable-horizon arithmetic in the empirical-rank allocation family:
#     fixed-T <= delta(n+1); delta/(w(w+1)) dies at ~sqrt(delta n); and the
#     horizon-free optimum is the exact harmonic cutoff (not n(e^delta-1)).
for n, delta in [(2094, 0.05), (2094, 0.1), (500, 0.05)]:
    h_fixed = cert_horizon_fixed_T(n, delta)
    h_sq = cert_horizon_schedule(lambda w: delta / (w * (w + 1.0)), n, delta)
    h_opt = cert_horizon_minimal(n, delta)
    check(f"horizon_fixedT_n{n}_d{delta}", h_fixed == int(np.floor(delta * (n + 1))),
          f"T*={h_fixed}")
    check(f"horizon_sqrt_law_n{n}_d{delta}", abs(h_sq - np.sqrt(delta * n)) <= 2.0,
          f"W*={h_sq} vs sqrt(delta n)={np.sqrt(delta*n):.1f}")
    exact = (harmonic_spend(n, h_opt) <= delta + 1e-15
             and harmonic_spend(n, h_opt + 1) > delta)
    lo, hi = n * np.expm1(delta), (n + 1) * np.expm1(delta)
    check(f"horizon_exact_harmonic_n{n}_d{delta}", exact and h_opt >= h_fixed,
          f"W*={h_opt}, spend={harmonic_spend(n,h_opt):.8f}, next={harmonic_spend(n,h_opt+1):.8f}")
    check(f"horizon_integral_bracket_n{n}_d{delta}", h_opt + 1 > lo and h_opt <= hi,
          f"{lo:.3f} sufficient-real cutoff, {hi:.3f} upper bound, W*={h_opt}")
    # lambda=0 is precisely the minimal-spend schedule and must stop at W*.
    ab = BudgetedAUC(delta, np.arange(n, dtype=float), lam=0.0)
    hb = 0
    for w in range(1, h_opt + 5):
        if not np.isfinite(ab.radius()):
            break
        hb = w
        ab.add(0.0)
    rem_after_stop = ab.d_rem
    _ = ab.radius()
    for _ in range(200):
        ab.add(0.0)
        assert not np.isfinite(ab.radius())
    check(f"aucb_reaches_exact_horizon_n{n}_d{delta}", hb == h_opt,
          f"aucb horizon={hb} vs optimal {h_opt}")
    check(f"aucb_stops_spending_n{n}_d{delta}",
          ab.exhausted and ab.d_rem == rem_after_stop,
          f"remaining={ab.d_rem:.8g}")

# Reviewer counterexample to the old reversed integral inequality.
check("harmonic_cutoff_counterexample_n1_dhalf",
      cert_horizon_minimal(1, 0.5) == 1
      and harmonic_spend(1, 1) == 0.5
      and harmonic_spend(1, 2) > 0.5,
      f"W*={cert_horizon_minimal(1,0.5)}")

# 8c2. the analytic AUC-B horizon used by exp2 matches the calibrator itself
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_e2", __import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)), "exp2_radius.py"))
for n0, dd, lm in [(500, 0.05, 0.05), (2094, 0.1, 0.01)]:
    ab = BudgetedAUC(dd, np.zeros(n0), lam=lm)
    h = 0
    while np.isfinite(ab.radius()) and h < 3000:
        h += 1
        ab.add(0.0)
    d_rem, w = dd, 0
    while w < 10**6:
        fl = 1.0 / (n0 + w + 1.0)
        if d_rem < fl:
            break
        d_rem -= max(fl, lm * d_rem)
        w += 1
    check(f"aucb_horizon_formula_n{n0}_d{dd}_lam{lm}", h == w, f"{h} vs {w}")

# 8d. episode-targeted ACI is just ACI at the Bonferroni window level
ae = ACIEpisode(0.08, 8, np.arange(100, dtype=float))
check("aci_episode_target", abs(ae.alpha_target - 0.01) < 1e-12,
      f"alpha_target={ae.alpha_target}")

# 8e. planner fallback is evasive and carries an exact reason label
from planner import plan_step
p_r = np.array([0.0, 0.0]); v_r = np.array([0.0, 0.0])
close = np.tile(np.array([[0.35, 0.0]]), (8, 1, 1))     # agent right on top of us
a_inf, fb_inf = plan_step(p_r, v_r, np.array([0.0, -4.0]), close, np.inf, 8)
a_big, fb_big = plan_step(p_r, v_r, np.array([0.0, -4.0]), close, 5.0, 8)
check("fallback_flagged_when_uncertified", fb_inf and fb_big)
check("fallback_moves_away", float(a_inf[0]) < -1e-6 and np.linalg.norm(a_inf) > 0.5,
      f"a={a_inf}")
check("fallback_increases_clearance",
      np.linalg.norm(a_big) > 0.5 and float(a_big[0]) < -1e-6, f"a={a_big}")
_, why_inf = plan_step(p_r, v_r, np.array([0.0, -4.0]), close, np.inf, 8,
                       return_status=True)
check("fallback_reason_no_certificate", why_inf == "fallback_no_certificate", why_inf)

# Absolute window scales must survive suffix replanning.  At distance 1.2 m,
# COLL_DIST + .5*r is feasible but COLL_DIST + 1.0*r is not.
one = np.array([[[1.2, 0.0]]])
_, anchored_ok = plan_step(p_r, v_r, np.array([0.0, -4.0]), one, 1.0, 1,
                           region_scales=np.array([0.5]), return_status=True)
_, recentered_bad = plan_step(p_r, v_r, np.array([0.0, -4.0]), one, 1.0, 1,
                              return_status=True)
_, tracked_bad = plan_step(p_r, v_r, np.array([0.0, -4.0]), one, 1.0, 1,
                           region_scales=np.array([0.5]), tracking_margin=0.3,
                           return_status=True)
check("absolute_window_scale_preserved", anchored_ok == "certified"
      and recentered_bad == "fallback_no_feasible_action",
      f"anchored={anchored_ok}, recentered={recentered_bad}")
check("tracking_margin_is_separate", tracked_bad == "fallback_no_feasible_action",
      tracked_bad)

# 9. Planner + sims smoke test (one quick episode per regime, uncalibrated)
from envloop import run_episode, window_scores_from_rollout, make_calibrator
from predictor import Predictor
from sims import SocialForce, record_rollout

pred = Predictor(cv_only=True)
rngE = np.random.default_rng(42)
m = run_episode("sf", make_calibrator("uncal", 0.1, []), "uncal", pred, rngE)
check("episode_sf_runs", m["n_windows"] >= 1 and m["min_dist"] > 0, str({k: m[k] for k in ("n_windows", "reached", "steps", "collision")}))
check("window_anchor_contract_logged",
      m["execution_contract"] == "window_anchor_v1"
      and len(m["predictor_update_steps"]) == m["steps"]
      and all(a is None or (a >= 5 and (a - 5) % 8 == 0)
              for a in m["constraint_anchor_steps"]),
      m["execution_contract"])
check("fallback_collision_labels_complete",
      len(m["fallback_labels"]) == m["steps"]
      and len(m["collision_step_labels"]) == m["steps"]
      and len(m["containment_labels_by_control_step"]) == m["steps"]
      and len(m["selected_agent_endpoint_clearance_implication_labels"]) == m["steps"]
      and sum(m["fallback_labels"]) == m["fallback_steps"],
      f"steps={m['steps']} fallback={m['fallback_steps']}")
check("physical_collision_is_separate_label",
      m["collision"] == any(m["collision_step_labels"])
      and all((x is None) or isinstance(x, bool)
              for x in m["containment_labels_by_control_step"]),
      "observed substep collisions are not aliases for tube containment")
P, ev = record_rollout(SocialForce(8, np.random.default_rng(1)), 70)
m2 = run_episode("replay", make_calibrator("cs", 0.1, np.abs(rngE.normal(size=500))), "cs", None, rngE, replay_item=(P, ev, pred))
check("episode_replay_runs", m2["n_windows"] >= 1, str({k: m2[k] for k in ("n_windows", "n_viol", "mean_radius")}))
sc = window_scores_from_rollout(P, ev, pred)
check("calib_scores", len(sc) >= 5 and all(np.isfinite(sc)), f"n={len(sc)} mean={np.mean(sc):.3f}")
# A respawned identity is excluded from the score event.  The artifact stores the
# surviving identity explicitly and names clearance as a selected-agent implication.
P_edge = np.empty((14, 2, 2), dtype=float)
P_edge[:, 0, :] = np.array([5.0, 5.0])
P_edge[:, 1, :] = np.array([-5.0, 5.0])
ev_edge = [np.array([], dtype=int) for _ in range(13)]
ev_edge[6] = np.array([0], dtype=int)
edge = run_episode(
    "replay", make_calibrator("cs", 0.1, np.abs(rngE.normal(size=500))),
    "cs", pred, np.random.default_rng(43), replay_item=(P_edge, ev_edge, pred),
    n_windows=1, record_audit_trace=True)
edge_audit = edge["window_audit"][0]
check("selected_agent_clearance_scope_logged",
      edge_audit["score_selected_agent_indices"] == [1]
      and edge_audit["score_selected_agent_uids"] == ["1:0"]
      and edge_audit["clean_agent_count"] == 1
      and edge_audit["all_window_agents_in_score_event"] is False
      and "discrete_clearance_implication_labels" not in edge,
      str(edge_audit))
check("slot_generation_identity_transition_logged",
      edge_audit["identity_transition_records"] == [{
          "slot": 0, "from_uid": "0:0", "to_uid": "0:1", "observed_step": 7}]
      and edge_audit["identity_policy_triggered"], str(edge_audit))
edge_ctl = edge["control_audit_trace"]
check("identity_fallback_starts_after_observed_transition",
      not edge_ctl[6]["identity_policy_fallback"]
      and edge_ctl[7]["identity_policy_fallback"]
      and edge_ctl[7]["identity_policy_input"].startswith("current_observation")
      and not any(edge["selected_agent_endpoint_clearance_implication_labels"][5:13]),
      f"statuses={[r['plan_status'] for r in edge_ctl[5:]]}")

tracked = run_episode(
    "sf", make_calibrator("cs", .1, np.abs(rngE.normal(size=500))),
    "cs", pred, np.random.default_rng(44), n_windows=1,
    tracking_error_bound=.1, tracking_margin=.1, tracking_error_phase=.37,
    record_audit_trace=True)
check("projected_acceleration_tracking_bound",
      max(tracked["tracking_error_norm_labels"]) <= .1 + 1e-10
      and all(tracked["tracking_bound_labels"])
      and tracked["tracking_stress"]["phase_rad"] == .37
      and "projected_acceleration" in tracked["tracking_contract"],
      f"max={max(tracked['tracking_error_norm_labels']):.6f}")
check("warmup_and_active_fallback_separated",
      tracked["fallback_steps"] == (tracked["fallback_steps_warmup"]
                                      + tracked["fallback_steps_active_window"])
      and tracked["fallback_steps_warmup"] == 5,
      f"warm={tracked['fallback_steps_warmup']} active={tracked['fallback_steps_active_window']}")
check("patrol_progress_fields_present",
      tracked["path_length_m"] >= 0.0
      and np.isfinite(tracked["signed_waypoint_progress_m"])
      and tracked["waypoint_completion_count"] == tracked["goals"])
try:
    run_episode("sf", make_calibrator("cs", .1, np.abs(rngE.normal(size=500))),
                "cs", pred, np.random.default_rng(45), n_windows=1,
                tracking_error_bound=.1, tracking_margin=.099)
except ValueError:
    uncovered_tracking_rejected = True
else:
    uncovered_tracking_rejected = False
check("uncovered_tracking_stress_rejected", uncovered_tracking_rejected)


# 10. G2: anytime-valid bound on the cumulative violation count (Sec. III-C)
from csq import ViolationBudgetCS, violation_budget_curve

B = violation_budget_curve(0.05, 0.05, 3000)
check("budget_monotone", bool(np.all(np.diff(B) >= 0)), f"B_1={B[0]} B_3000={B[-1]}")
check("budget_rate_to_eps",
      B[119] / 120 > 0.05 and abs(B[2999] / 3000 - 0.05) < 0.02,
      f"B_120/120={B[119]/120:.3f} B_3000/3000={B[2999]/3000:.4f}")
# worst case under H0: violations exactly Bernoulli(eps)
rngB = np.random.default_rng(11)
V = np.cumsum(rngB.random((6000, 1500)) < 0.05, axis=1)
ex = float((V > B[None, :1500]).any(axis=1).mean())
check("budget_anytime_valid", ex <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 6000),
      f"anytime exceed {ex:.4f} <= 0.05")
# a stream that violates 3x as often must be caught
V2 = np.cumsum(rngB.random((400, 1500)) < 0.15, axis=1)
check("budget_detects_excess", float((V2 > B[None, :1500]).any(axis=1).mean()) > 0.99,
      "3x violation rate is rejected")
vb = ViolationBudgetCS(0.05, 0.05, w_max=500)
for _ in range(200):
    vb.add(False)
check("budget_object_tracks", vb.W == 200 and vb.count == 0 and not vb.exceeded,
      f"W={vb.W} B={vb.budget()}")
vbgrow = ViolationBudgetCS(0.05, 0.05, w_max=2)
for _ in range(3):
    vbgrow.add(False)
check("violation_boundary_auto_extends", len(vbgrow._u) >= vbgrow.W,
      f"capacity={len(vbgrow._u)} W={vbgrow.W}")

# 11. Prop. 4: shift-robust CS
from csq import RobustQuantileCS, cs_boundary

u0 = cs_boundary(4000, 0.95, 0.05)
u1 = cs_boundary(4000, 0.95, 0.05, rho_tot=12.0)
check("robust_boundary_wider", bool(np.all(u1 >= u0)) and u1[3999] > u0[3999],
      f"u_4000: exact {u0[-1]} -> robust {u1[-1]}")
check("robust_reduces_to_exact",
      bool(np.array_equal(cs_boundary(2000, 0.95, 0.05, rho_tot=0.0),
                          cs_boundary(2000, 0.95, 0.05))), "rho=0 is the exact CS")
rngR = np.random.default_rng(5)
T = 2000
p_shift = np.full(T, 0.95 + 0.004)                 # total budget 8
Zs = np.cumsum(rngR.random((3000, T)) < p_shift[None, :], axis=1)
fail_plain = float((Zs >= u0[None, :T]).any(axis=1).mean())
u_rob = cs_boundary(T, 0.95, 0.05, rho_tot=8.0)
fail_rob = float((Zs >= u_rob[None, :]).any(axis=1).mean())
check("robust_restores_validity_under_shift", fail_plain > 0.05 and fail_rob <= 0.05,
      f"plain {fail_plain:.3f} vs robust {fail_rob:.3f} (target 0.05)")
sM = np.sort(np.abs(rngR.normal(size=3000)))
r_ex = RobustQuantileCS(0.95, 0.05, sM, rho_tot=0.0).radius()
r_rb = RobustQuantileCS(0.95, 0.05, sM, rho_tot=12.0).radius()
check("robust_price_is_modest", r_rb >= r_ex and r_rb / r_ex < 1.25,
      f"radius {r_ex:.3f} -> {r_rb:.3f} (+{100*(r_rb/r_ex-1):.1f}%)")
rgrow = RobustQuantileCS(0.9, 0.1, [-1.0, 0.0, 1.0], rho_tot=0.2, t_max=2)
rgrow.radius()
check("robust_boundary_auto_extends", len(rgrow._u) >= rgrow.t,
      f"capacity={len(rgrow._u)} t={rgrow.t}")

# 12. Prop. 1 feasibility is analytic; allocation quality is only heuristic.
from csq import allocation_grid_heuristic, cert_horizon_lookahead

nP = 800
sP = np.sort(np.random.default_rng(3).gamma(2.0, 0.4, nP)).tolist()
qP = lambda w, lev: split_cp_quantile(sP, lev)
wall = cert_horizon_fixed_T(nP, 0.05)
check("allocation_heuristic_respects_rank_wall",
      allocation_grid_heuristic(qP, wall, 0.05, nP)[2]
      and not allocation_grid_heuristic(qP, wall + 1, 0.05, nP)[2],
      f"feasible at T={wall}, infeasible at T={wall+1}")
check("allocation_heuristic_uniform_when_homogeneous",
      abs(allocation_grid_heuristic(qP, 20, 0.05, nP)[1].sum()
          / (20 * qP(0, 1 - 0.05 / 20)) - 1) < 1e-9,
      "heuristic returns uniform on identical curves")
# The heuristic can find gains, but empirical quantile steps can create a duality gap.
rngH = np.random.default_rng(4)
pl = [np.sort(rngH.gamma(9.0, 0.1, nP)).tolist() if w % 2 == 0
      else np.sort(rngH.pareto(1.6, nP) * 0.3 + 0.4).tolist() for w in range(20)]
qH = lambda w, lev: split_cp_quantile(pl[w], lev)
gain = 1 - allocation_grid_heuristic(qH, 20, 0.05, nP)[1].sum() / sum(
    qH(w, 1 - 0.05 / 20) for w in range(20))
check("allocation_heuristic_finds_example_gain", gain > 0.1,
      f"attained radius gain {100*gain:.1f}%")
# Regression for a concrete nonconvex-step duality gap: heuristic total 6, while
# the feasible allocation (.1,.3,.3) has total radius 5.
gap_pools = [[0.0] * 6 + [5.0] * 3,
             [0.0] * 7 + [3.0] * 2,
             [0.0] * 7 + [3.0] * 2]
q_gap = lambda w, lev: split_cp_quantile(gap_pools[w], lev)
_, gap_r, gap_feas = allocation_grid_heuristic(q_gap, 3, 0.7, 9)
known_r = np.array([q_gap(w, 1 - e) for w, e in enumerate((.1, .3, .3))])
check("allocation_heuristic_not_mislabeled_optimal",
      gap_feas and abs(gap_r.sum() - 6.0) < 1e-12
      and abs(known_r.sum() - 5.0) < 1e-12,
      f"heuristic={gap_r.sum():.1f}, known feasible={known_r.sum():.1f}")
check("lookahead_wall_divides_by_H",
      cert_horizon_lookahead(nP, 0.05, 8) == cert_horizon_fixed_T(nP, 0.05) // 8
      or abs(cert_horizon_lookahead(nP, 0.05, 8) - wall / 8) <= 1,
      f"scalar {wall} -> per-lookahead {cert_horizon_lookahead(nP, 0.05, 8)}")

# 13. The unicycle vehicle honours the same certified constraint
from planner import plan_step_uni, robot_step_uni

st = np.array([0.0, 4.6, -np.pi / 2, 0.0])
pr_ = np.zeros((8, 1, 2))
u_free, fb_free = plan_step_uni(st, np.array([0.0, -4.6]), pr_, 0.0, 8)
u_inf, fb_inf = plan_step_uni(st, np.array([0.0, -4.6]), pr_, np.inf, 8)
check("uni_no_certificate_is_fallback", (not fb_free) and fb_inf,
      f"r=0 fallback={fb_free}, r=inf fallback={fb_inf}")
stn = robot_step_uni(st, u_free)
check("uni_dynamics_nonholonomic", stn[3] >= 0.0 and abs(stn[2] - st[2]) <= 2.5 * 0.25 + 1e-9,
      f"v={stn[3]:.3f} dtheta={abs(stn[2]-st[2]):.3f}")
sf_lat = run_episode("sf", make_calibrator("cs", 0.1, np.abs(rngE.normal(size=500))),
                     "cs", pred, np.random.default_rng(9), vehicle="uni",
                     latency=0, act_noise=0.05)
check("uni_episode_runs", sf_lat["n_windows"] >= 1 and sf_lat["min_dist"] > 0,
      str({k: sf_lat[k] for k in ("n_windows", "steps", "fallback_steps")}))
check("no_false_tracking_certificate",
      sf_lat["tracking_contract"] == "empirical_only_unbounded_gaussian_actuation_noise"
      and not any(sf_lat["tracking_bound_labels"])
      and not any(sf_lat["selected_agent_endpoint_clearance_implication_labels"]),
      sf_lat["tracking_contract"])
try:
    run_episode("sf", make_calibrator("cs", 0.1, np.abs(rngE.normal(size=500))),
                "cs", pred, np.random.default_rng(9), vehicle="uni", latency=1)
except ValueError:
    stale_rejected = True
else:
    stale_rejected = False
check("stale_prediction_contract_rejected", stale_rejected)

print(f"\n{'ALL TESTS PASSED' if FAIL == 0 else str(FAIL) + ' TESTS FAILED'}")
sys.exit(1 if FAIL else 0)
