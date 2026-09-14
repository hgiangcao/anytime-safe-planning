"""Lifelong MAPF simulation loop: PIBT + task assignment + metrics.

Task models:
  uniform   : new goal ~ uniform over free cells (excluding current cell)
  warehouse : agents alternate pickup-cell -> station -> pickup ... legs
Throughput = completed goals (legs) per timestep.
"""
import time
import numpy as np

from . import maps as mapmod
from . import flow as flowmod
from .pibt import PIBT, dist_table


class TaskModel:
    def __init__(self, m, seed=0, hotspot=0.0):
        self.m = m
        self.rng = np.random.default_rng(seed + 991)
        self.ci = m["cell_index"]
        self.kind = "warehouse" if m["name"].startswith("warehouse") else "uniform"
        if self.kind == "warehouse":
            self.picks = self.ci[m["meta"]["pickups"]].astype(int)
            self.stats = self.ci[m["meta"]["stations"]].astype(int)
        self.phase = {}  # agent -> next leg type (warehouse)
        # Task-distribution shift: with probability `hotspot`, a uniform-map
        # goal is drawn from the top-left quadrant instead of the whole map.
        # Guidance derived under uniform demand is then mis-specified, which is
        # the regime the online variant is meant to handle.
        self.hotspot = float(hotspot)
        if self.hotspot > 0 and self.kind == "uniform":
            H, W = m["H"], m["W"]
            yx = np.array([divmod(int(v), W) for v in m["cells"]])
            self.hot = np.flatnonzero((yx[:, 0] < H // 2) & (yx[:, 1] < W // 2))
        else:
            self.hot = None

    def goal_cells(self):
        """Compact indices a goal can ever take, or None if unrestricted.
        Restricting the goal set keeps distance tables tractable on large maps
        (the warehouse task model only ever targets pickups and stations)."""
        if self.kind == "uniform":
            return None
        return np.unique(np.concatenate([self.picks, self.stats]))

    def new_goal(self, i, cur):
        if self.kind == "uniform":
            n = len(self.m["cells"])
            hot = self.hot is not None and self.rng.random() < self.hotspot
            while True:
                if hot:
                    g = int(self.hot[self.rng.integers(len(self.hot))])
                else:
                    g = int(self.rng.integers(n))
                if g != cur:
                    return g
        # warehouse: alternate pickup / station legs
        ph = self.phase.get(i, 0)
        self.phase[i] = 1 - ph
        pool = self.picks if ph == 0 else self.stats
        g = int(pool[self.rng.integers(len(pool))])
        if g == cur:
            g = int(pool[self.rng.integers(len(pool))])
        return g


def run_lifelong(m, n_agents, steps, seed=0, edge_weights=None, D=None,
                 record_edges=False, online=None, warmup=0, swap=True,
                 tau=20, eps_tie=1.0, planner="pibt", planner_kw=None,
                 hotspot=0.0, record_full=False):
    """Run lifelong PIBT. Returns metrics dict.

    edge_weights : guidance weights aligned with maps.directed_edges(m) or None
    online       : None or dict(period=int, window=int, K=int, fw_iters=int,
                   net=FlowNet) -> re-derive tolls from observed demand
    warmup       : steps excluded from throughput statistics
    """
    t0 = time.time()
    tasks = TaskModel(m, seed=seed, hotspot=hotspot)
    goal_cells = tasks.goal_cells()
    if D is None:
        D = dist_table(m, edge_weights, goal_cells=goal_cells)
    if planner == "pibt":
        solver = PIBT(m, D, n_agents, seed=seed, swap=swap, tau=tau,
                      eps_tie=eps_tie)
    elif planner == "wpp":
        from .wpp import WindowedPP
        solver = WindowedPP(m, D, n_agents, seed=seed, **(planner_kw or {}))
        solver.set_weights(edge_weights)
    else:
        raise ValueError(planner)
    cells = m["cells"]
    for i in range(n_agents):
        solver.goals[i] = tasks.new_goal(i, solver.pos[i])

    # Optional review-grade trace. Compact-cell indices make the position and
    # goal state exact while keeping even the warehouse arrays tractable.
    pos_trace = goal_trace = None
    task_start_t = task_moves = task_origin = None
    goal_events = []
    if record_full:
        pos_trace = np.empty((steps + 1, n_agents), dtype=np.int32)
        goal_trace = np.empty((steps + 1, n_agents), dtype=np.int32)
        pos_trace[0] = solver.pos
        goal_trace[0] = solver.goals
        task_start_t = np.zeros(n_agents, dtype=np.int32)
        task_moves = np.zeros(n_agents, dtype=np.int32)
        task_origin = np.asarray(solver.pos, dtype=np.int32).copy()

    completed = 0
    completed_warm = 0
    series = []
    edge_counts = None
    eid = None
    if record_edges:
        edges, _ = mapmod.directed_edges(m)
        ci = m["cell_index"]
        n = len(cells)
        eid = -np.ones((n, n), dtype=np.int32)
        eid[ci[edges[:, 0]], ci[edges[:, 1]]] = np.arange(len(edges), dtype=np.int32)
        edge_counts = np.zeros(len(edges), dtype=np.int64)
    od_log = []          # (origin_grid, dest_grid) of assigned legs (for online)
    resolve_time = 0.0
    n_resolves = 0
    if online is not None:
        for i in range(n_agents):
            od_log.append((int(cells[solver.pos[i]]), int(cells[solver.goals[i]])))

    online_mode = "synchronous" if online is None else online.get("mode", "synchronous")
    if online_mode not in ("synchronous", "stale"):
        raise ValueError("online mode must be 'synchronous' or 'stale'")
    pending_guidance = None
    update_events = []

    prev = list(solver.pos)
    window_completed = 0
    for t in range(steps):
        if pending_guidance is not None and t >= pending_guidance[0]:
            _, solver.D, solver.grow, event_index = pending_guidance
            update_events[event_index]["applied_at_step"] = int(t)
            pending_guidance = None
        solver.step()
        if record_edges:
            for i in range(n_agents):
                if solver.pos[i] != prev[i]:
                    edge_counts[eid[prev[i], solver.pos[i]]] += 1
        # goal completions -> new tasks, priority reset
        for i in range(n_agents):
            if record_full and solver.pos[i] != prev[i]:
                task_moves[i] += 1
            if solver.pos[i] == solver.goals[i]:
                completed += 1
                window_completed += 1
                if t >= warmup:
                    completed_warm += 1
                if record_full:
                    goal_events.append((int(t + 1), int(i), int(task_start_t[i]),
                                        int(t + 1 - task_start_t[i]),
                                        int(task_moves[i]), int(task_origin[i]),
                                        int(solver.goals[i])))
                solver.elapsed[i] = 0
                o = solver.pos[i]
                solver.goals[i] = tasks.new_goal(i, o)
                if record_full:
                    task_start_t[i] = t + 1
                    task_moves[i] = 0
                    task_origin[i] = o
                if online is not None:
                    od_log.append((int(cells[o]), int(cells[solver.goals[i]])))
            else:
                solver.elapsed[i] += 1
        prev = list(solver.pos)
        if record_full:
            pos_trace[t + 1] = solver.pos
            goal_trace[t + 1] = solver.goals
        if (t + 1) % 100 == 0:
            series.append(dict(t=t + 1, completed=completed))
        # ---- online re-solve --------------------------------------------
        if online is not None and (t + 1) % online["period"] == 0 and t + 1 < steps:
            if online_mode == "stale" and pending_guidance is not None:
                update_events.append({"requested_at_step": int(t + 1),
                                      "status": "skipped_single_worker_busy"})
                continue
            tr = time.time()
            net = online["net"]
            win = od_log[-online["window"]:]
            lam_obs = window_completed / online["period"]
            window_completed = 0
            dests, demand = flowmod.empirical_demand(net, win, max(lam_obs, 0.1),
                                                     K=online["K"])
            if dests is not None and len(dests) > 0:
                # no warm start: a flow feasible for the previous window's
                # demand is NOT feasible for the new commodities, and FW
                # requires a feasible start.
                sol = net.solve(dests, demand, max_iter=online["fw_iters"],
                                tol=1e-3, f0=None,
                                init_bias=online.get("init_bias"))
                tolls = net.tolls(sol["f"])
                new_D, new_grow = dist_table(m, tolls, goal_cells=goal_cells)
                elapsed = time.time() - tr
                event = {"requested_at_step": int(t + 1),
                         "compute_and_rebuild_s": float(elapsed),
                         "mode": online_mode,
                         "status": "computed"}
                update_events.append(event)
                if online_mode == "synchronous":
                    solver.D, solver.grow = new_D, new_grow
                    event["applied_at_step"] = int(t + 1)
                else:
                    dt = float(online.get("control_period_s", 0.1))
                    delay = max(1, int(np.ceil(elapsed / dt)))
                    event["control_period_s"] = dt
                    event["measured_delay_steps"] = delay
                    pending_guidance = (t + 1 + delay, new_D, new_grow,
                                        len(update_events) - 1)
                n_resolves += 1
            resolve_time += time.time() - tr

    eff_steps = steps - warmup
    out = dict(
        throughput=completed_warm / eff_steps,
        completed=completed, steps=steps, warmup=warmup,
        n_agents=n_agents, seed=seed, map=m["name"],
        wall_time=time.time() - t0, resolve_time=resolve_time,
        n_resolves=n_resolves, series=series,
        edge_counts=edge_counts,
        update_events=update_events,
    )
    if record_full:
        out.update(position_trace=pos_trace, goal_trace=goal_trace,
                   goal_events=np.asarray(goal_events, dtype=np.int64).reshape(-1, 7),
                   trace_schema={"position_trace": "[step,agent] compact free-cell index",
                                 "goal_trace": "[step,agent] compact free-cell index after reassignment",
                                 "goal_events_columns": ["completion_step", "agent", "assigned_step",
                                                         "task_delay_steps", "moves", "origin_compact",
                                                         "goal_compact"]})
    return out


def check_valid_run(m, n_agents, steps, seed=0):
    """Run and assert no vertex/edge collisions each step (test helper)."""
    D = dist_table(m, None)
    solver = PIBT(m, D, n_agents, seed=seed)
    tasks = TaskModel(m, seed=seed)
    for i in range(n_agents):
        solver.goals[i] = tasks.new_goal(i, solver.pos[i])
    for t in range(steps):
        old = list(solver.pos)
        solver.step()
        new = solver.pos
        assert len(set(new)) == n_agents, f"vertex collision at t={t}"
        moved = {}
        for i in range(n_agents):
            moved[(old[i], new[i])] = i
        for i in range(n_agents):
            if old[i] != new[i]:
                j = moved.get((new[i], old[i]))
                assert j is None, f"edge swap {i},{j} at t={t}"
            # moves are to adjacent cells or stay
            if old[i] != new[i]:
                assert new[i] in solver.nbrs[old[i]], f"teleport at t={t}"
        for i in range(n_agents):
            if solver.pos[i] == solver.goals[i]:
                solver.elapsed[i] = 0
                solver.goals[i] = tasks.new_goal(i, solver.pos[i])
            else:
                solver.elapsed[i] += 1
    return True
