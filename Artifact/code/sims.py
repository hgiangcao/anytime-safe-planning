"""Compact self-implemented 2-D crowd simulators + replay regime.

- SocialForce: standard social-force pedestrian model (Helbing-Molnar style,
  exponential repulsion), agents flow across a corridor and are recycled to keep
  density stationary.
- OrcaLite: compact sampling-based reciprocal/velocity-obstacle avoidance
  ("ORCA-style"): each agent picks, from a candidate set, the velocity minimizing
  deviation from its preferred velocity plus a time-to-collision penalty against
  other agents (assumed to keep their current velocity) and the robot.
  This is a faithful-in-spirit, compact reimplementation, not exact ORCA.
- Replay: recorded agent-only rollouts replayed open loop (robot invisible to
  agents) -> exchangeable scores; used for the clean-guarantee regime.

Units: meters, seconds.  Control period DT = 0.25 s, simulator substep 0.05 s.
Workspace: |x| <= 7.5, |y| <= 5.5.  Agents spawn on left/right edges and cross.
"""
from __future__ import annotations

import numpy as np

DT = 0.25          # control period (s)
SUBSTEPS = 5       # simulator substeps per control period
DT_SUB = DT / SUBSTEPS
AGENT_R = 0.3
ROBOT_R = 0.3
COLL_DIST = AGENT_R + ROBOT_R  # 0.6 m


def spawn_crossing(n_agents, rng):
    """Agents crossing left<->right through the corridor the robot traverses N->S."""
    side = rng.integers(0, 2, n_agents) * 2 - 1  # -1: starts left, +1: starts right
    x = -side * (5.5 + rng.uniform(0, 2.0, n_agents))
    y = rng.uniform(-3.8, 3.8, n_agents)
    pos = np.stack([x, y], axis=1)
    goal = np.stack([side * 7.5, y + rng.uniform(-1.0, 1.0, n_agents)], axis=1)
    v0 = np.clip(rng.normal(1.2, 0.2, n_agents), 0.6, 1.8)
    vel = np.zeros_like(pos)
    return pos, vel, goal, v0


def _recycle(pos, vel, goal, v0, rng):
    """Respawn agents that crossed; keeps flow density stationary. Returns ids of
    respawned agents (their prediction history is invalidated by the caller)."""
    done = np.abs(pos[:, 0]) > 7.2
    ids = np.where(done)[0]
    for i in ids:
        side = 1 if pos[i, 0] > 0 else -1  # arrived on this side; respawn there, go back
        pos[i] = [side * (5.5 + rng.uniform(0, 2.0)), rng.uniform(-3.8, 3.8)]
        goal[i] = [-side * 7.5, pos[i, 1] + rng.uniform(-1.0, 1.0)]
        vel[i] = 0.0
        v0[i] = np.clip(rng.normal(1.2, 0.2), 0.6, 1.8)
    return ids


class SocialForce:
    A = 2.0      # repulsion strength (m/s^2)
    B = 0.35     # repulsion range (m)
    TAU = 0.5    # relaxation time (s)
    VMAX = 2.2

    def __init__(self, n_agents, rng):
        self.rng = rng
        self.pos, self.vel, self.goal, self.v0 = spawn_crossing(n_agents, rng)

    def step_control(self, robot_pos=None, robot_visible=False, robot_vel=None):
        """Advance one control period; returns (substep_positions [SUBSTEPS,n,2],
        respawned agent ids)."""
        subs = []
        respawned = []
        for _ in range(SUBSTEPS):
            d = self.goal - self.pos
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            f = (self.v0[:, None] * d / dist - self.vel) / self.TAU
            # pairwise repulsion
            diff = self.pos[:, None, :] - self.pos[None, :, :]
            dd = np.linalg.norm(diff, axis=2) + 1e-9
            np.fill_diagonal(dd, np.inf)
            mag = self.A * np.exp((2 * AGENT_R - dd) / self.B)
            f += (mag[:, :, None] * diff / dd[:, :, None]).sum(axis=1)
            if robot_visible and robot_pos is not None:
                rdiff = self.pos - robot_pos[None, :]
                rd = np.linalg.norm(rdiff, axis=1, keepdims=True) + 1e-9
                f += self.A * np.exp((AGENT_R + ROBOT_R - rd) / self.B) * rdiff / rd
            self.vel = self.vel + f * DT_SUB
            sp = np.linalg.norm(self.vel, axis=1, keepdims=True)
            self.vel = np.where(sp > self.VMAX, self.vel * self.VMAX / sp, self.vel)
            self.pos = self.pos + self.vel * DT_SUB
            subs.append(self.pos.copy())
        respawned = _recycle(self.pos, self.vel, self.goal, self.v0, self.rng)
        return np.array(subs), respawned


class OrcaLite:
    TTC_H = 2.5   # time-to-collision horizon (s)
    WC = 1.5      # collision penalty weight
    N_DIR = 16

    def __init__(self, n_agents, rng):
        self.rng = rng
        self.pos, self.vel, self.goal, self.v0 = spawn_crossing(n_agents, rng)
        ang = np.linspace(0, 2 * np.pi, self.N_DIR, endpoint=False)
        self._dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    @staticmethod
    def _ttc(p, v, radius):
        """Time to collision of relative state (p, v) for combined radius."""
        # solve ||p + v t|| = radius
        a = (v * v).sum(-1)
        b = 2 * (p * v).sum(-1)
        c = (p * p).sum(-1) - radius**2
        ttc = np.full(a.shape, np.inf)
        already = c <= 0
        disc = b * b - 4 * a * c
        ok = (disc >= 0) & (a > 1e-12)
        t1 = np.where(ok, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12)), np.inf)
        ttc = np.where(ok & (t1 > 0), t1, np.inf)
        ttc = np.where(already, 0.0, ttc)
        return ttc

    def step_control(self, robot_pos=None, robot_visible=False, robot_vel=None):
        n = len(self.pos)
        d = self.goal - self.pos
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        vpref = self.v0[:, None] * d / dist
        # candidate velocities per agent: dirs x speeds + pref + zero
        speeds = np.array([0.5, 1.0, 1.4])
        cands = (self._dirs[None, :, None, :] * (speeds[None, None, :, None] * self.v0[:, None, None, None]))
        cands = cands.reshape(n, -1, 2)
        cands = np.concatenate([cands, vpref[:, None, :], np.zeros((n, 1, 2))], axis=1)  # n,K,2
        K = cands.shape[1]
        newv = np.empty_like(self.vel)
        for i in range(n):
            relp = self.pos[i] - np.delete(self.pos, i, axis=0)      # n-1,2
            othv = np.delete(self.vel, i, axis=0)
            relv = cands[i][:, None, :] - othv[None, :, :]           # K,n-1,2
            ttc = self._ttc(relp[None, :, :], relv, 2 * AGENT_R)     # K,n-1
            if robot_visible and robot_pos is not None:
                rp = (self.pos[i] - robot_pos)[None, None, :]
                rv = cands[i][:, None, :] - (robot_vel if robot_vel is not None else np.zeros(2))[None, None, :]
                rttc = self._ttc(rp, rv, AGENT_R + ROBOT_R)
                ttc = np.concatenate([ttc, rttc], axis=1)
            pen = np.where(ttc < self.TTC_H, self.WC / (ttc + 0.2), 0.0).sum(axis=1)
            cost = ((cands[i] - vpref[i]) ** 2).sum(axis=1) + pen
            newv[i] = cands[i][np.argmin(cost)]
        self.vel = newv
        subs = []
        for _ in range(SUBSTEPS):
            self.pos = self.pos + self.vel * DT_SUB
            subs.append(self.pos.copy())
        respawned = _recycle(self.pos, self.vel, self.goal, self.v0, self.rng)
        return np.array(subs), respawned


class Replay:
    """Open-loop replay of a recorded rollout: positions [T, n, 2] at control steps
    (with linear substep interpolation), plus recorded respawn events."""

    def __init__(self, positions, respawn_events):
        self.P = positions  # [T+1, n, 2] control-step positions
        self.respawns = respawn_events  # list of arrays per step
        self.t = 0
        self.pos = self.P[0].copy()
        self.vel = np.zeros_like(self.pos)

    def step_control(self, robot_pos=None, robot_visible=False, robot_vel=None):
        t = self.t
        p0, p1 = self.P[t], self.P[t + 1]
        fr = (np.arange(1, SUBSTEPS + 1) / SUBSTEPS)[:, None, None]
        subs = p0[None] * (1 - fr) + p1[None] * fr
        # respawned agents teleport; don't interpolate them
        ids = self.respawns[t]
        if len(ids):
            subs[:, ids, :] = p1[None, ids, :]
        self.vel = (p1 - p0) / DT
        if len(ids):
            self.vel[ids] = 0.0
        self.pos = p1.copy()
        self.t += 1
        return subs, ids


def record_rollout(sim, T):
    """Run an agent-only sim for T control steps; return (positions [T+1,n,2],
    respawn event list of length T)."""
    P = [sim.pos.copy()]
    ev = []
    for _ in range(T):
        _, ids = sim.step_control(robot_visible=False)
        P.append(sim.pos.copy())
        ev.append(np.asarray(ids, dtype=int))
    return np.array(P), ev
