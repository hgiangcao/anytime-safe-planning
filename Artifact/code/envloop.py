"""Closed-loop / replay episode runner + the shared scoring protocol.

Scoring protocol (identical timestamp convention for calibration and deployment;
that equality is necessary but does not itself make replay windows exchangeable):
  - Non-overlapping windows of H control steps starting at t = L, L+H, L+2H, ...
  - At the window start the (frozen) predictor issues positions for j = 1..H.
  - Window nonconformity score (lookahead-normalized, cf. the k-step residual in
    the spec): s = max over CLEAN agents, max over j of
    (H/j) * || actual(t+j) - predicted(t+j | t) ||,
    so the certified region at lookahead j is a ball of radius r * j/H around the
    prediction: s <= r  <=>  every selected clean agent stays inside its region
    at every j.
    An agent is clean iff it was not respawned (recycled through the arena edge)
    during [t-L+1, t+H]; teleports would otherwise contaminate the score.
  - The inflation radius r used throughout a window is read from the calibrator
    BEFORE the window's score is revealed; a window is violated iff s > r.

The planner recomputes the robot action every control step, but every hard
pedestrian constraint within a scored window is the remaining suffix of the
prediction issued at that window's start.  New predictions are timestamped for
audit but never replace the certified tube.  This makes the implication
``window score <= radius -> selected-clean-agent containment`` literal in the
implementation; excluded/respawned identities receive no clearance implication.

Containment alone is not collision avoidance.  Selected-agent discrete endpoint
clearance also needs a valid robot tracking/reachability margin, and fallback
steps carry no certificate.  Continuous-time collision avoidance would additionally require
an intersample swept-volume bound, which this score does not provide.  The
simulator's double-integrator execution is exact at control times (zero tracking
margin); Gaussian actuation noise has no deterministic bound, so the
nonholonomic-noise arm is explicitly empirical rather than certified.
"""
from __future__ import annotations

import hashlib

import numpy as np

from csq import (ACI, ACIEpisode, AlphaSpendingQuantileCS, AnytimeUnionCP,
                 BudgetedAUC, QuantileCS, RayleighCC, RobustQuantileCS,
                 Uncalibrated, UnionBoundCP, ViolationBudgetCS)
from planner import A_MAX, plan_step, plan_step_uni, robot_step, robot_step_uni
from predictor import H, L
from sims import COLL_DIST, DT, SUBSTEPS, OrcaLite, Replay, SocialForce

W_EP = 8                    # scored windows per (full-length) episode
T_MAX = L + W_EP * H        # 69 control steps ~ 17 s
JSCALE = (np.arange(1, H + 1) / H)          # region scale at lookahead j
JW = JSCALE[:, None]                        # broadcast over agents
ROBOT_START = np.array([0.0, 4.6])
ROBOT_GOAL = np.array([0.0, -4.6])
GOAL_TOL = 0.5


RHO_DEFAULT = 12.0   # Legacy empirical stress-test setting.  A held-out
                     # marginal diagnostic does not certify the conditional
                     # pathwise shift budget required by RobustQuantileCS.


def make_calibrator(method, delta, warm_scores, cache_dir=None, T=None, lam=0.05,
                    rho_tot=RHO_DEFAULT):
    """T = episode horizon in scored windows made available to the fixed-T
    baselines (union, gauss).  Horizon-free methods (cs, auc, aucb) ignore it."""
    T = W_EP if T is None else int(T)
    if method == "cs":
        return QuantileCS(1.0 - delta, delta, warm_scores, cache_dir=cache_dir)
    if method == "spending_cs":
        return AlphaSpendingQuantileCS(
            1.0 - delta, delta, warm_scores, cache_dir=cache_dir)
    if method == "cs_r":
        return RobustQuantileCS(1.0 - delta, delta, warm_scores,
                                rho_tot=rho_tot, cache_dir=cache_dir)
    if method == "auc":
        return AnytimeUnionCP(delta, warm_scores)
    if method == "aucb":
        return BudgetedAUC(delta, warm_scores, lam=lam)
    if method == "union":
        return UnionBoundCP(delta, T, warm_scores)
    if method == "aci":
        return ACI(delta, warm_scores, gamma=0.01)
    if method == "aci_ep":
        return ACIEpisode(delta, T, warm_scores, gamma=0.01)
    if method == "gauss":
        return RayleighCC(delta, T, warm_scores)
    if method == "uncal":
        return Uncalibrated()
    raise ValueError(method)


ONLINE = {"cs": True, "spending_cs": True, "cs_r": True, "auc": True,
          "aucb": True, "aci": True, "aci_ep": True, "union": False,
          "gauss": False, "uncal": False}
EPISODIC = ("auc", "aucb")          # calibrators with per-episode internal state
ACI_LIKE = ("aci", "aci_ep")
FINITE_SAMPLE_METHODS = {"cs", "spending_cs", "cs_r", "auc", "aucb", "union"}
EXECUTION_CONTRACT = "window_anchor_v1"
TRACKING_STRESS_CONTRACT = "projected_acceleration_endpoint_bound_v1"
IDENTITY_POLICY = "current_observation_speed_envelope_v2"
IDENTITY_SPEED_CAP_MPS = {"sf": 2.2, "orca": 2.52}


def window_scores_from_rollout(P, ev, predictor):
    """Calibration scores from a recorded agent-only rollout (positions P
    [T+1,n,2], respawn events ev).  Same windows/cleanliness as deployment."""
    T1, n, _ = P.shape
    V = np.diff(P, axis=0) / DT
    resp = np.zeros((T1, n), dtype=bool)
    for t, ids in enumerate(ev):
        if len(ids):
            resp[t + 1, ids] = True
    out = []
    t = L
    while t + H < T1:
        clean = ~resp[t - L + 1 : t + H + 1].any(axis=0)
        if clean.any():
            vhist = V[t - L : t].transpose(1, 0, 2)  # [n,L,2]
            pred = predictor.predict(P[t], np.ascontiguousarray(vhist))  # [H,n,2]
            err = np.linalg.norm(P[t + 1 : t + H + 1] - pred, axis=2) / JW  # [H,n]
            out.append(float(err[:, clean].max()))
        t += H
    return out


def run_episode(regime, calibrator, method, predictor, rng, replay_item=None,
                record_traj=False, n_windows=None, patrol=False,
                vehicle="di", latency=0, act_noise=0.0, budget_cs=None,
                tracking_margin=0.0, shift_budget_certified=False,
                tracking_error_bound=0.0, tracking_error_phase=0.0,
                identity_policy=IDENTITY_POLICY,
                identity_agent_speed_bound=None,
                record_audit_trace=False):
    """One episode; returns a metrics dict.  replay_item = (P, ev, predictor).

    n_windows: scored windows in the episode (default W_EP).  patrol=True runs a
    continuous-deployment episode: reaching the goal flips it back to the start
    and the episode runs for the full window budget, so that long-horizon
    behaviour (certifiability decay, radius drift, throughput) can be measured
    instead of being cut short by the goal-triggered stopping rule.
    record_traj=True additionally returns per-step robot/agent positions, the
    current predictions and radius (for the snapshot figure).

    vehicle: "di" = 2-D double integrator or "uni" = differential-drive
    unicycle.  ``latency`` is retained only to reject legacy calls: stale
    mid-window predictions are not covered by the window-start score.  Gaussian
    actuation noise of relative scale ``act_noise`` is allowed for empirical
    stress testing, but has no finite tracking guarantee.  ``tracking_margin``
    is a separately supplied robot-side reachability allowance.
    ``shift_budget_certified`` may be true for CS-R only when its conditional
    pathwise budget came from an external bound; the held-out marginal proxy in
    this repository is not such a bound.

    ``tracking_error_bound`` is an upper bound on the endpoint deviation caused
    by a deterministic acceleration stress.  The requested acceleration offset
    has norm ``tracking_error_bound / DT**2`` and a fixed phase sequence; the
    executed acceleration is projected back to the declared acceleration ball.
    Projection and the velocity clip are nonexpansive, so the realized endpoint
    error is at most the requested bound.  Position and velocity are both
    advanced by ``robot_step`` and remain dynamically consistent.  A run is
    rejected unless ``tracking_margin`` covers the bound.

    ``identity_policy`` handles entering, respawned, or history-insufficient
    identities.  A recent identity at an anchor, or a transition observed in an
    active window, discards the stale anchored prediction and plans against
    balls grown from the *current observations* at a declared agent-speed cap.
    If the reachability action is infeasible, the planner maximizes clearance
    from those current observations.  This is an explicit conservative
    simulator policy, not a statistical or continuous-time safety theorem; the
    transition step remains observed only.  The score-selected set is resolved
    retrospectively by the same no-respawn rule as calibration.  Slot-generation
    UIDs prevent a recycled simulator slot from masquerading as one identity.

    budget_cs: an optional ViolationBudgetCS; when given, the running violation
    count and its anytime budget B_W are recorded per window (guarantee G2)."""
    if latency:
        raise ValueError(
            "window_anchor_v1 does not certify stale/recentered predictions; "
            "use latency=0 or define and calibrate a latency-aware score")
    if tracking_margin < 0.0:
        raise ValueError("tracking_margin must be nonnegative")
    if tracking_error_bound < 0.0:
        raise ValueError("tracking_error_bound must be nonnegative")
    if tracking_error_bound > tracking_margin + 1e-12:
        raise ValueError("tracking_margin must cover tracking_error_bound")
    if tracking_error_bound and vehicle != "di":
        raise ValueError("bounded endpoint disturbance is implemented for vehicle='di'")
    if identity_policy != IDENTITY_POLICY:
        raise ValueError("unsupported identity policy")
    W = W_EP if n_windows is None else int(n_windows)
    T_end = L + W * H
    if regime == "sf":
        sim, visible = SocialForce(int(rng.integers(5, 12)), rng), True
    elif regime == "orca":
        sim, visible = OrcaLite(int(rng.integers(5, 12)), rng), True
    elif regime in ("replay", "eth"):
        P, ev, predictor = replay_item
        sim, visible = Replay(P, ev), False
        T_end = min(T_end, len(P) - 1)
    else:
        raise ValueError(regime)

    if identity_agent_speed_bound is None:
        identity_agent_speed_bound = IDENTITY_SPEED_CAP_MPS.get(regime)
    if (identity_agent_speed_bound is not None
            and float(identity_agent_speed_bound) <= 0.0):
        raise ValueError("identity_agent_speed_bound must be positive or None")

    n = len(sim.pos)
    p, v = ROBOT_START.copy(), np.zeros(2)
    uni_state = np.array([ROBOT_START[0], ROBOT_START[1], -np.pi / 2, 0.0])
    goal = ROBOT_GOAL.copy()
    budget_trace = []                   # (W, V_W, B_W) per scored window
    vbuf = [np.zeros((n, 2)) for _ in range(L)]
    last_resp = np.full(n, -10**9)
    slot_generation = np.zeros(n, dtype=np.int64)

    if method in EPISODIC:
        calibrator.new_episode()

    windows = []  # open scoring windows: dict(t0, preds, r, pos0)
    wrec = []     # closed, scored windows: [score, radius_or_None] (aligned)
    window_audit = []
    predictor_update_steps = []
    constraint_anchor_steps = []
    fallback_labels = []
    fallback_trace = []
    collision_step_labels = []
    anchored_constraint_feasible_labels = []
    tracking_bound_labels = []
    containment_labels_by_control_step = []
    selected_agent_endpoint_clearance_implication_labels = []
    identity_policy_fallback_labels = []
    tracking_error_norm_labels = []
    control_audit_trace = []
    viols, radii, uncert = [], [], 0
    fallback_steps = 0
    fallback_steps_warmup = 0
    fallback_steps_active_window = 0
    goals = 0
    path_length_m = 0.0
    signed_waypoint_progress_m = 0.0
    positive_waypoint_progress_m = 0.0
    min_dist = np.inf
    collided = False
    n_coll_steps = 0
    reached = False
    steps = T_end

    traj = [] if record_traj else None
    prev_pos = sim.pos.copy()
    for t in range(T_end):
        p_before_step = p.copy()
        vhist = np.ascontiguousarray(np.stack(vbuf, axis=1))  # [n,L,2]
        preds = predictor.predict(sim.pos, vhist)  # fresh [H,n,2], audit only mid-window
        predictor_update_steps.append(t)
        if t >= L and (t - L) % H == 0 and t + H <= T_end:
            r0 = calibrator.radius()  # read exactly once, before this score exists
            anchor_selected = last_resp <= t - L
            pred_blob = np.asarray(preds, dtype="<f8").tobytes()
            windows.append({
                "t0": t,
                "preds": preds.copy(),
                "r": r0,
                "anchor_selected": anchor_selected.copy(),
                "anchor_slot_generation": slot_generation.copy(),
                "identity_policy_triggered": bool(not anchor_selected.all()),
                "identity_transition_ids": set(),
                "identity_transition_steps": [],
                "identity_transition_records": [],
                "anchor_prediction_sha256": hashlib.sha256(pred_blob).hexdigest(),
            })
            radii.append(r0 if np.isfinite(r0) else None)

        # Replan robot motion against the suffix of the *same anchored tube*.
        # A fresh prediction never replaces a certified constraint mid-window.
        if windows:
            active = windows[0]  # scoring windows are non-overlapping
            offset = t - active["t0"]
            if not 0 <= offset < H:
                raise RuntimeError("active scoring window is misaligned")
            preds_ctl = active["preds"][offset:]
            scales_ctl = JSCALE[offset:]
            horizon_ctl = H - offset
            r_ctl = active["r"]
            anchor_step = active["t0"]
            identity_fallback = bool(active["identity_policy_triggered"])
        else:
            # History warm-up has no scored/certified window.  Motion is allowed
            # only as an explicitly logged no-certificate fallback.
            preds_ctl = preds
            scales_ctl = JSCALE
            horizon_ctl = H
            r_ctl = np.inf
            anchor_step = None
            identity_fallback = False
        constraint_anchor_steps.append(anchor_step)

        identity_policy_input = "anchored_prediction_suffix"
        if identity_fallback:
            # Never steer against the stale pre-respawn trajectory.  The speed
            # envelope is centered on the observations available *now*.
            preds_ctl = np.repeat(sim.pos[None, :, :], horizon_ctl, axis=0)
            if identity_agent_speed_bound is None:
                r_ctl = np.inf
                scales_ctl = np.arange(1, horizon_ctl + 1, dtype=float) / horizon_ctl
                identity_policy_input = "current_observations_evasive_no_speed_cap"
            else:
                r_ctl = float(identity_agent_speed_bound) * DT * horizon_ctl
                scales_ctl = np.arange(1, horizon_ctl + 1, dtype=float) / horizon_ctl
                identity_policy_input = "current_observation_speed_envelope"

        if record_traj:
            traj.append({"t": t, "robot": p.copy(), "agents": sim.pos.copy(),
                         "fresh_preds": preds.copy(),
                         "constraint_preds": preds_ctl.copy(),
                         "constraint_region_scales": scales_ctl.copy(),
                         "identity_policy_input": identity_policy_input,
                         "constraint_anchor_step": anchor_step,
                         "r": float(r_ctl)})
        nominal_control = None
        executed_control = None
        if vehicle == "uni":
            u, plan_status = plan_step_uni(
                uni_state, goal, preds_ctl,
                r_ctl, horizon_ctl,
                act_noise=act_noise, rng=rng, region_scales=scales_ctl,
                tracking_margin=tracking_margin, return_status=True)
            uni_new = robot_step_uni(uni_state, u)
            p_planned, v_new = uni_new[:2].copy(), np.array(
                [uni_new[3] * np.cos(uni_new[2]), uni_new[3] * np.sin(uni_new[2])])
            nominal_control = np.asarray(u, dtype=float)
            executed_control = nominal_control.copy()
            p_new = p_planned.copy()
        else:
            a, plan_status = plan_step(
                p, v, goal, preds_ctl,
                r_ctl, horizon_ctl,
                region_scales=scales_ctl, tracking_margin=tracking_margin,
                return_status=True)
            nominal_control = np.asarray(a, dtype=float)
            p_planned, _v_planned = robot_step(p, v, nominal_control)
            executed_control = nominal_control.copy()
            if tracking_error_bound:
                angle = (float(tracking_error_phase)
                         + t * (np.sqrt(5.0) - 1.0) * np.pi)
                da = (float(tracking_error_bound) / (DT * DT)) * np.array(
                    [np.cos(angle), np.sin(angle)])
                executed_control = nominal_control + da
                an = float(np.linalg.norm(executed_control))
                if an > A_MAX:
                    executed_control = executed_control * (A_MAX / an)
            p_new, v_new = robot_step(p, v, executed_control)
        if anchor_step is None:
            plan_status = "fallback_no_active_window"
        elif identity_fallback:
            plan_status = (
                "identity_speed_envelope_feasible_uncertified"
                if plan_status == "certified"
                else "fallback_identity_current_observation_evasive")
        fell_back = plan_status != "certified"
        fallback_steps += int(fell_back)
        fallback_steps_warmup += int(fell_back and anchor_step is None)
        fallback_steps_active_window += int(fell_back and anchor_step is not None)
        fallback_labels.append(bool(fell_back))
        if fell_back:
            fallback_trace.append({"control_step": t, "reason": plan_status})
        tracking_error_norm = float(np.linalg.norm(p_new - p_planned))
        if tracking_error_norm > tracking_error_bound + 1e-10:
            raise RuntimeError("projected acceleration exceeded endpoint-error bound")
        tracking_is_bounded = bool(
            not (vehicle == "uni" and act_noise > 0.0)
            and tracking_error_norm <= tracking_margin + 1e-12)
        anchored_constraint_feasible_labels.append(bool(
            anchor_step is not None and plan_status == "certified"))
        tracking_bound_labels.append(bool(tracking_is_bounded))
        identity_policy_fallback_labels.append(bool(identity_fallback))
        tracking_error_norm_labels.append(tracking_error_norm)
        # These two labels are filled retrospectively when the score window
        # closes.  They are distinct from the observed substep collision label:
        # endpoint containment + tracking proves endpoint clearance only.
        containment_labels_by_control_step.append(None)
        selected_agent_endpoint_clearance_implication_labels.append(False)

        distance_before_step = float(np.linalg.norm(p_before_step - goal))
        uids_at_plan = [f"{i}:{int(g)}" for i, g in enumerate(slot_generation)]
        subs, resp_ids = sim.step_control(robot_pos=p, robot_visible=visible, robot_vel=v)
        step_transition_records = []
        if len(resp_ids):
            ids_arr = np.asarray(resp_ids, dtype=int)
            last_resp[ids_arr] = t + 1
            for i in ids_arr:
                old_generation = int(slot_generation[i])
                slot_generation[i] += 1
                step_transition_records.append({
                    "slot": int(i),
                    "from_uid": f"{int(i)}:{old_generation}",
                    "to_uid": f"{int(i)}:{int(slot_generation[i])}",
                    "observed_step": int(t + 1),
                })
            for wd in windows:
                wd["identity_policy_triggered"] = True
                wd["identity_transition_ids"].update(
                    int(i) for i in ids_arr)
                wd["identity_transition_steps"].append(int(t + 1))
                wd["identity_transition_records"].extend(step_transition_records)
        # collision check on substeps (robot linearly interpolated)
        fr = (np.arange(1, SUBSTEPS + 1) / SUBSTEPS)[:, None]
        rob = p[None] * (1 - fr) + p_new[None] * fr  # [S,2]
        d = np.linalg.norm(subs - rob[:, None, :], axis=2)  # [S,n]
        md = float(d.min())
        post_respawn_endpoint_min = None
        if len(resp_ids):
            post_d = np.linalg.norm(
                sim.pos[np.asarray(resp_ids, dtype=int)] - p_new[None, :], axis=1)
            post_respawn_endpoint_min = float(post_d.min())
        observed_step_min = min(
            md, post_respawn_endpoint_min
            if post_respawn_endpoint_min is not None else np.inf)
        min_dist = min(min_dist, observed_step_min)
        collision_now = observed_step_min < COLL_DIST
        collision_step_labels.append(bool(collision_now))
        if collision_now:
            collided = True
            n_coll_steps += 1

        if record_traj:
            traj[-1].update({
                "planned_robot_endpoint": np.asarray(p_planned).copy(),
                "realized_robot_endpoint": np.asarray(p_new).copy(),
                "agents_after_control": sim.pos.copy(),
                "agent_substep_positions": np.asarray(subs).copy(),
                "observed_step_min_distance_m": float(observed_step_min),
                "observed_substep_collision": bool(collision_now),
            })

        if record_traj:
            traj[-1]["agent_substep_positions"] = subs.copy()
            traj[-1]["agents_after_control"] = sim.pos.copy()
            traj[-1]["planned_robot_endpoint"] = p_planned.copy()
            traj[-1]["realized_robot_endpoint"] = p_new.copy()
            traj[-1]["observed_step_min_distance_m"] = observed_step_min
            traj[-1]["observed_substep_collision"] = bool(collision_now)

        if record_audit_trace:
            control_audit_trace.append({
                "control_step": int(t),
                "predictor_update_step": int(t),
                "constraint_anchor_step": (
                    None if anchor_step is None else int(anchor_step)),
                "plan_status": plan_status,
                "fallback": bool(fell_back),
                "identity_policy_fallback": bool(identity_fallback),
                "tracking_margin_m": float(tracking_margin),
                "tracking_error_norm_m": tracking_error_norm,
                "tracking_bound_satisfied": bool(tracking_is_bounded),
                "tracking_stress_contract": TRACKING_STRESS_CONTRACT,
                "tracking_error_phase_rad": float(tracking_error_phase),
                "nominal_control": nominal_control.tolist(),
                "executed_control": executed_control.tolist(),
                "identity_policy_input": identity_policy_input,
                "identity_agent_speed_bound_mps": (
                    None if identity_agent_speed_bound is None
                    else float(identity_agent_speed_bound)),
                "observed_agent_uids_at_plan": uids_at_plan,
                "identity_transitions_observed_after_control": step_transition_records,
                "fresh_prediction_sha256": hashlib.sha256(
                    np.asarray(preds, dtype="<f8").tobytes()).hexdigest(),
                "constraint_prediction_sha256": hashlib.sha256(
                    np.asarray(preds_ctl, dtype="<f8").tobytes()).hexdigest(),
                "constraint_region_scales": np.asarray(
                    scales_ctl, dtype=float).tolist(),
                "constraint_horizon": int(horizon_ctl),
                "planned_robot_endpoint": np.asarray(p_planned).tolist(),
                "realized_robot_endpoint": np.asarray(p_new).tolist(),
                "min_agent_distance_over_substeps_m": md,
                "post_respawn_endpoint_min_distance_m": post_respawn_endpoint_min,
                "observed_step_min_distance_m": observed_step_min,
                "observed_substep_collision": bool(collision_now),
            })

        if vehicle == "uni":
            uni_state = uni_new
        vbuf.pop(0)
        vnow = (sim.pos - prev_pos) / DT
        if len(resp_ids):
            vnow[np.asarray(resp_ids, dtype=int)] = 0.0
        vbuf.append(vnow)
        prev_pos = sim.pos.copy()
        p, v = p_new, v_new
        step_distance = float(np.linalg.norm(p_new - p_before_step))
        distance_after_step = float(np.linalg.norm(p_new - goal))
        path_length_m += step_distance
        signed_waypoint_progress_m += distance_before_step - distance_after_step
        positive_waypoint_progress_m += max(
            0.0, distance_before_step - distance_after_step)

        # record actual positions (sim.pos is now at time t+1) into open windows
        for wd in windows:
            if "trace" not in wd:
                wd["trace"] = np.empty((H, n, 2))
            wd["trace"][t - wd["t0"]] = sim.pos

        # close any window ending now (its last predicted step is t+1 == t0+H)
        for wd in [w for w in windows if t + 1 == w["t0"] + H]:
            t0 = wd["t0"]
            clean = last_resp <= t0 - L  # no respawn in (t0-L, t0+H]
            if clean.any():
                err = np.linalg.norm(wd["trace"] - wd["preds"], axis=2) / JW
                s = float(err[:, clean].max())
                r0 = wd["r"]
                wrec.append([s, float(r0) if np.isfinite(r0) else None])
                violation = bool(s > r0) if np.isfinite(r0) else None
                window_audit.append({
                    "window_index": len(wrec),
                    "attempted_window_index": len(window_audit) + 1,
                    "prediction_issued_step": int(t0),
                    "score_observed_step": int(t0 + H),
                    "constraint_control_steps": list(range(int(t0), int(t0 + H))),
                    "anchor_prediction_sha256": wd["anchor_prediction_sha256"],
                    "score": s,
                    "radius": float(r0) if np.isfinite(r0) else None,
                    "score_status": "scored",
                    "score_selection_resolution": (
                        "retrospective_no_respawn_rule_identical_to_calibration"),
                    "total_agent_count": int(n),
                    "anchor_eligible_agent_indices": np.flatnonzero(
                        wd["anchor_selected"]).astype(int).tolist(),
                    "anchor_eligible_agent_uids": [
                        f"{int(i)}:{int(wd['anchor_slot_generation'][i])}"
                        for i in np.flatnonzero(wd["anchor_selected"])],
                    "excluded_at_anchor_agent_indices": np.flatnonzero(
                        ~wd["anchor_selected"]).astype(int).tolist(),
                    "excluded_at_anchor_agent_uids": [
                        f"{int(i)}:{int(wd['anchor_slot_generation'][i])}"
                        for i in np.flatnonzero(~wd["anchor_selected"])],
                    "within_window_identity_transition_indices": sorted(
                        wd["identity_transition_ids"]),
                    "identity_transition_steps": wd["identity_transition_steps"],
                    "identity_transition_records": wd["identity_transition_records"],
                    "identity_policy_triggered": bool(
                        wd["identity_policy_triggered"]),
                    "finite_sample_radius_available_under_declared_assumptions": bool(
                        method in FINITE_SAMPLE_METHODS and np.isfinite(r0)
                        and (method != "cs_r" or shift_budget_certified)),
                    "containment_violation": violation,
                    "clean_agent_count": int(clean.sum()),
                    "score_selected_agent_indices": np.flatnonzero(clean).astype(int).tolist(),
                    "score_selected_agent_uids": [
                        f"{int(i)}:{int(wd['anchor_slot_generation'][i])}"
                        for i in np.flatnonzero(clean)],
                    "all_window_agents_in_score_event": bool(clean.all()),
                })
                contained = bool(violation is False)
                for q in range(int(t0), int(t0 + H)):
                    if q >= len(containment_labels_by_control_step):
                        raise RuntimeError("score window closes before labels exist")
                    containment_labels_by_control_step[q] = (
                        contained if np.isfinite(r0) else None)
                    selected_agent_endpoint_clearance_implication_labels[q] = bool(
                        contained
                        and anchored_constraint_feasible_labels[q]
                        and tracking_bound_labels[q]
                        and not wd["identity_policy_triggered"])
                if np.isfinite(r0):
                    viols.append(violation)
                else:
                    uncert += 1
                if budget_cs is not None:
                    budget_cs.add(np.isfinite(r0) and s > r0)
                    budget_trace.append(
                        [budget_cs.W, budget_cs.count, int(budget_cs.budget())])
                if ONLINE[method]:
                    if method in ACI_LIKE:
                        # an infinite radius is an UNCERTIFIED window, not a
                        # violated one: in Gibbs-Candes an alpha <= 0 level means
                        # the prediction set is all of R, which always covers.
                        calibrator.update_err(
                            1.0 if (np.isfinite(r0) and s > r0) else 0.0)
                    calibrator.add(s)
            else:
                window_audit.append({
                    "window_index": None,
                    "attempted_window_index": len(window_audit) + 1,
                    "prediction_issued_step": int(t0),
                    "score_observed_step": int(t0 + H),
                    "constraint_control_steps": list(range(int(t0), int(t0 + H))),
                    "anchor_prediction_sha256": wd["anchor_prediction_sha256"],
                    "score": None,
                    "radius": float(wd["r"]) if np.isfinite(wd["r"]) else None,
                    "score_status": "not_scored_no_stable_identity",
                    "score_selection_resolution": (
                        "retrospective_no_respawn_rule_identical_to_calibration"),
                    "total_agent_count": int(n),
                    "anchor_eligible_agent_indices": np.flatnonzero(
                        wd["anchor_selected"]).astype(int).tolist(),
                    "anchor_eligible_agent_uids": [
                        f"{int(i)}:{int(wd['anchor_slot_generation'][i])}"
                        for i in np.flatnonzero(wd["anchor_selected"])],
                    "excluded_at_anchor_agent_indices": np.flatnonzero(
                        ~wd["anchor_selected"]).astype(int).tolist(),
                    "excluded_at_anchor_agent_uids": [
                        f"{int(i)}:{int(wd['anchor_slot_generation'][i])}"
                        for i in np.flatnonzero(~wd["anchor_selected"])],
                    "within_window_identity_transition_indices": sorted(
                        wd["identity_transition_ids"]),
                    "identity_transition_steps": wd["identity_transition_steps"],
                    "identity_transition_records": wd["identity_transition_records"],
                    "identity_policy_triggered": True,
                    "finite_sample_radius_available_under_declared_assumptions": False,
                    "containment_violation": None,
                    "clean_agent_count": 0,
                    "score_selected_agent_indices": [],
                    "score_selected_agent_uids": [],
                    "all_window_agents_in_score_event": False,
                })
            windows.remove(wd)

        if np.linalg.norm(p - goal) < GOAL_TOL:
            goals += 1
            if patrol:  # continuous deployment: turn around and keep going
                goal = ROBOT_START.copy() if goal[1] < 0 else ROBOT_GOAL.copy()
            else:
                reached = True
                steps = t + 1
                break

    return {
        "artifact_schema_version": 3,
        "execution_contract": EXECUTION_CONTRACT,
        "predictor": {
            "class": type(predictor).__name__,
            "cv_only": bool(getattr(predictor, "cv_only", False)),
            "model_id": getattr(predictor, "model_id", None),
        },
        "tracking_contract": (
            "empirical_only_unbounded_gaussian_actuation_noise"
            if vehicle == "uni" and act_noise > 0.0
            else (f"{TRACKING_STRESS_CONTRACT}_endpoint_error_at_most_"
                  f"{float(tracking_error_bound):.6g}m_with_"
                  f"{float(tracking_margin):.6g}m_margin")),
        "tracking_stress": {
            "contract": TRACKING_STRESS_CONTRACT,
            "requested_endpoint_error_bound_m": float(tracking_error_bound),
            "tracking_margin_m": float(tracking_margin),
            "phase_rad": float(tracking_error_phase),
            "direction_sequence": "phase+t*(sqrt(5)-1)*pi",
            "acceleration_projection_radius_mps2": float(A_MAX),
        },
        "identity_policy": identity_policy,
        "identity_policy_contract": {
            "agent_speed_bound_mps": (
                None if identity_agent_speed_bound is None
                else float(identity_agent_speed_bound)),
            "transition_step_status": "observation_only_fallback_begins_next_control",
            "uid_format": "slot:generation",
            "scope": "simulator_policy_not_statistical_or_continuous_time_theorem",
        },
        "statistical_contract": {
            "method": method,
            "finite_sample_construction": bool(method in FINITE_SAMPLE_METHODS),
            "assumption_status": (
                "externally_certified_conditional_shift_budget"
                if method == "cs_r" and shift_budget_certified
                else "conditional_shift_budget_not_certified"
                if method == "cs_r"
                else "exchangeability_required_not_established_by_this_record"
                if method in FINITE_SAMPLE_METHODS
                else "no_finite_sample_contract"
            ),
        },
        "n_windows": len(wrec),
        "n_attempted_windows": len(window_audit),
        "n_unscored_identity_windows": sum(
            row["score_status"] != "scored" for row in window_audit),
        "n_viol": int(np.sum(viols)) if viols else 0,
        "episode_viol": bool(np.any(viols)) if viols else False,
        "n_uncert": uncert,
        "collision": collided,
        "n_coll_steps": n_coll_steps,
        "min_dist": min_dist,
        "reached": reached,
        "goals": goals,
        "steps": steps,
        "fallback_steps": fallback_steps,
        "fallback_steps_warmup": fallback_steps_warmup,
        "fallback_steps_active_window": fallback_steps_active_window,
        "mean_radius": float(np.mean([x for x in radii if x is not None])) if any(
            x is not None for x in radii) else None,
        "windows": wrec,  # [score, radius_or_None] per scored window, in order
        "window_audit": window_audit,
        "predictor_update_steps": predictor_update_steps,
        "constraint_anchor_steps": constraint_anchor_steps,
        "fallback_labels": fallback_labels,
        "fallback_trace": fallback_trace,
        "collision_step_labels": collision_step_labels,
        "anchored_constraint_feasible_labels": anchored_constraint_feasible_labels,
        "tracking_bound_labels": tracking_bound_labels,
        "containment_labels_by_control_step": containment_labels_by_control_step,
        "selected_agent_endpoint_clearance_implication_labels":
            selected_agent_endpoint_clearance_implication_labels,
        "identity_policy_fallback_labels": identity_policy_fallback_labels,
        "tracking_error_norm_labels": tracking_error_norm_labels,
        "initial_goal_distance_m": float(np.linalg.norm(ROBOT_START - ROBOT_GOAL)),
        "final_goal_distance_m": float(np.linalg.norm(p - goal)),
        "net_goal_progress_m": float(
            np.linalg.norm(ROBOT_START - ROBOT_GOAL) - np.linalg.norm(p - goal)),
        "path_length_m": float(path_length_m),
        "signed_waypoint_progress_m": float(signed_waypoint_progress_m),
        "positive_waypoint_progress_m": float(positive_waypoint_progress_m),
        "waypoint_completion_count": int(goals),
        **({"budget_trace": budget_trace} if budget_cs is not None else {}),
        **({"control_audit_trace": control_audit_trace}
           if record_audit_trace else {}),
        **({"traj": traj} if record_traj else {}),
    }
