"""Sampling-based receding-horizon planner for a 2-D double integrator.

DWA-style candidate sequences are rolled out against the still-relevant suffix
of the prediction made at the current *window start*.  This anchored tube is the
hard certified constraint: replanning the robot motion must not silently
recenter it on a newer prediction.  Feasibility also accepts a separate robot
tracking margin; pedestrian containment and robot tracking are distinct.

If no candidate is feasible, or the calibrator reports no finite radius, the
planner executes a clearance-maximising evasive fallback.  A fallback has no
robot-clearance certificate and is returned with an explicit reason; statistical
containment, tracking status, and observed collision remain separate log fields.
"""
from __future__ import annotations

import numpy as np

from sims import COLL_DIST, DT

V_MAX = 1.6
A_MAX = 2.0
X_LIM = 7.3
Y_LIM = 5.3

def _make_candidates(n_dir=16, mags=(0.7, 1.4, 2.0)):
    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    m = np.asarray(mags, dtype=float)
    return (dirs[:, None, :] * m[None, :, None]).reshape(-1, 2)


_ACC = _make_candidates()  # [48,2]; + coast + brake = 50 candidates


def configure(n_dir=16, mags=(0.7, 1.4, 2.0)):
    """Resize the sampling grid (planner-sensitivity ablation, exp6)."""
    global _ACC
    _ACC = _make_candidates(n_dir, mags)
    return len(_ACC) + 2


def _finish(control, status, return_status):
    """Preserve the historical bool API while supporting exact audit labels."""
    return control, (status if return_status else status != "certified")


def _scales(horizon, preds, region_scales):
    if region_scales is None:
        return np.arange(1, horizon + 1, dtype=float) / preds.shape[0]
    out = np.asarray(region_scales, dtype=float)
    if out.shape != (horizon,):
        raise ValueError("region_scales must have one entry per lookahead")
    if np.any(out <= 0.0) or np.any(out > 1.0) or np.any(np.diff(out) <= 0.0):
        raise ValueError("region_scales must be strictly increasing in (0, 1]")
    return out


def plan_step(p, v, goal, preds, r, horizon, *, region_scales=None,
              tracking_margin=0.0, return_status=False):
    """One receding-horizon planning step.

    p, v: robot position/velocity (2,); goal: (2,); preds: predicted agent
    positions [H, n, 2] for lookahead steps j = 1..H; r: inflation radius (may be
    +inf); horizon: number of remaining lookahead steps to enforce (<= len(preds)).
    ``region_scales`` contains their absolute offsets divided by the original
    window horizon.  For an eight-step window after three executed controls,
    pass [4/8,...,8/8], not [1/5,...,5/5].  ``tracking_margin`` is a nonnegative
    robot-side allowance added separately from pedestrian inflation.

    Returns (accel, fallback_bool) by default.  With ``return_status=True`` the
    second item is ``certified``, ``fallback_no_certificate``, or
    ``fallback_no_feasible_action``.
    """
    if horizon < 1 or horizon > len(preds):
        raise ValueError("horizon must be in [1, len(preds)]")
    if tracking_margin < 0.0:
        raise ValueError("tracking_margin must be nonnegative")
    brake = -v / (np.linalg.norm(v) + 1e-9) * min(A_MAX, np.linalg.norm(v) / DT)
    cands = np.concatenate([_ACC, np.zeros((1, 2)), brake[None]], axis=0)  # [K,2]
    K = len(cands)
    H = horizon
    # rollout with velocity clipping
    pos = np.empty((K, H, 2))
    vel = np.repeat(v[None], K, axis=0)
    cur = np.repeat(p[None], K, axis=0)
    for j in range(H):
        vel = vel + cands * DT
        sp = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(sp > V_MAX, vel * V_MAX / np.maximum(sp, 1e-9), vel)
        cur = cur + vel * DT
        pos[:, j] = cur
    inb = (np.abs(pos[:, :, 0]) <= X_LIM).all(axis=1) & (np.abs(pos[:, :, 1]) <= Y_LIM).all(axis=1)
    if preds.shape[1] > 0:
        d = np.linalg.norm(pos[:, :, None, :] - preds[None, :H], axis=3)  # [K,H,n]
        clear = d.min(axis=(1, 2))
        if np.isfinite(r):
            infl = COLL_DIST + float(tracking_margin) + r * _scales(
                H, preds, region_scales)
            feas = (d - infl[None, :, None]).min(axis=(1, 2)) >= 0.0
        else:
            feas = np.zeros(K, dtype=bool)  # no certificate exists
    else:
        clear = np.full(K, np.inf)
        feas = np.full(K, np.isfinite(r))
    feas &= inb
    gd = np.linalg.norm(pos[:, -1] - goal[None], axis=1)
    if feas.any():
        cost = gd + 0.02 * np.linalg.norm(cands, axis=1) - 0.1 * np.minimum(clear, 1.0)
        return _finish(cands[int(np.argmin(np.where(feas, cost, np.inf)))],
                       "certified", return_status)
    # ---- uncertified fallback: maximise clearance, stay in bounds, prefer goal
    ok = inb if inb.any() else np.ones(K, dtype=bool)
    ev_cost = -clear + 0.02 * gd
    reason = ("fallback_no_certificate" if not np.isfinite(r)
              else "fallback_no_feasible_action")
    return _finish(cands[int(np.argmin(np.where(ok, ev_cost, np.inf)))],
                   reason, return_status)


def robot_step(p, v, a):
    """Apply one control period of the double integrator (with velocity clip)."""
    v = v + a * DT
    sp = np.linalg.norm(v)
    if sp > V_MAX:
        v = v * V_MAX / sp
    return p + v * DT, v


# ---------------------------------------------------------------- nonholonomic
# A differential-drive (unicycle) robot with actuation noise and one step of
# sensing/compute latency.  Reviewers reasonably object that a 2-D double
# integrator is a permissive vehicle model: it can stop and reverse instantly,
# which makes a conservative inflation radius unusually cheap.  The unicycle
# cannot, so the freezing/efficiency trade-off the calibrators induce is
# genuinely harder.  The certified constraint is unchanged --
# clearance >= COLL_DIST + r * j/H at every lookahead j -- so the calibration
# question is identical and only the vehicle is harder.

W_MAX = 2.5          # rad/s yaw rate
A_LON_MAX = 1.5      # m/s^2 longitudinal
V_MAX_UNI = 1.4      # m/s


def _uni_candidates(n_w=9, n_a=5):
    ws = np.linspace(-W_MAX, W_MAX, n_w)
    accs = np.linspace(-A_LON_MAX, A_LON_MAX, n_a)
    return np.stack(np.meshgrid(accs, ws, indexing="ij"), axis=-1).reshape(-1, 2)


_UNI = _uni_candidates()   # [45,2] (a_lon, omega) held constant over the horizon


def configure_unicycle(n_w=9, n_a=5):
    global _UNI
    _UNI = _uni_candidates(n_w, n_a)
    return len(_UNI)


def plan_step_uni(state, goal, preds, r, horizon, act_noise=0.0, rng=None, *,
                  region_scales=None, tracking_margin=0.0,
                  return_status=False):
    """One receding-horizon step for the unicycle state = (x, y, theta, v).

    Constant (a_lon, omega) candidates rolled out over `horizon` steps under the
    exact unicycle update; feasibility, cost and the uncertified
    clearance-maximising fallback are identical to `plan_step`.  Gaussian
    actuation noise is not hidden inside the pedestrian radius: a caller seeking
    a robot-motion certificate must provide a separately derived tracking margin.
    """
    if horizon < 1 or horizon > len(preds):
        raise ValueError("horizon must be in [1, len(preds)]")
    if tracking_margin < 0.0:
        raise ValueError("tracking_margin must be nonnegative")
    x, y, th, v = state
    C = _UNI
    K = len(C)
    H = horizon
    pos = np.empty((K, H, 2))
    xs = np.full(K, x); ys = np.full(K, y); ths = np.full(K, th); vs = np.full(K, v)
    for j in range(H):
        vs = np.clip(vs + C[:, 0] * DT, 0.0, V_MAX_UNI)   # no reverse
        ths = ths + C[:, 1] * DT
        xs = xs + vs * np.cos(ths) * DT
        ys = ys + vs * np.sin(ths) * DT
        pos[:, j, 0] = xs
        pos[:, j, 1] = ys
    inb = (np.abs(pos[:, :, 0]) <= X_LIM).all(axis=1) & (np.abs(pos[:, :, 1]) <= Y_LIM).all(axis=1)
    if preds.shape[1] > 0:
        d = np.linalg.norm(pos[:, :, None, :] - preds[None, :H], axis=3)
        clear = d.min(axis=(1, 2))
        if np.isfinite(r):
            infl = COLL_DIST + float(tracking_margin) + r * _scales(
                H, preds, region_scales)
            feas = (d - infl[None, :, None]).min(axis=(1, 2)) >= 0.0
        else:
            feas = np.zeros(K, dtype=bool)
    else:
        clear = np.full(K, np.inf)
        feas = np.full(K, np.isfinite(r))
    feas &= inb
    gd = np.linalg.norm(pos[:, -1] - goal[None], axis=1)
    if feas.any():
        u = C[int(np.argmin(np.where(feas, gd - 0.1 * np.minimum(clear, 1.0), np.inf)))]
        status = "certified"
    else:
        ok = inb if inb.any() else np.ones(K, dtype=bool)
        u = C[int(np.argmin(np.where(ok, -clear + 0.02 * gd, np.inf)))]
        status = ("fallback_no_certificate" if not np.isfinite(r)
                  else "fallback_no_feasible_action")
    if act_noise > 0.0 and rng is not None:
        u = u + rng.normal(0.0, act_noise, 2) * np.array([A_LON_MAX, W_MAX])
    return _finish(u, status, return_status)


def robot_step_uni(state, u):
    x, y, th, v = state
    v = float(np.clip(v + u[0] * DT, 0.0, V_MAX_UNI))
    th = float(th + u[1] * DT)
    return np.array([x + v * np.cos(th) * DT, y + v * np.sin(th) * DT, th, v])
