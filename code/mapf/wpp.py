"""Windowed prioritized planning (WPP) - a second, structurally different
lifelong planner, used to test whether derived guidance weights are
planner-agnostic or specific to PIBT.

PIBT is a *reactive* rule: each agent greedily descends a distance-to-goal
table one step at a time. WPP is the low-level scheme of RHCR
(Li et al., AAAI-21): every `period` timesteps all agents are replanned, in
decreasing priority order, by space-time A* over a `window`-step horizon
against a reservation table of the already-planned agents. It therefore does a
genuine bounded-horizon search over space-time rather than a one-step greedy
choice, while consuming the guidance graph in exactly the same place -- edge
traversal costs and the distance-to-goal heuristic are both taken under the
guidance weights.

Collision handling: reservations forbid two agents occupying a cell at the same
timestep (vertex conflict) and forbid traversing an edge whose reverse is
traversed at the same timestep (swap conflict). An agent with no feasible plan
holds its cell for the window, which is always reservable because it reserves
its own cell first.
"""
import heapq

import numpy as np

from . import maps as mapmod


class WindowedPP:
    def __init__(self, m, D, n_agents, seed=0, grow=None, period=5, window=16):
        if isinstance(D, tuple):
            D, grow = D
        self.m = m
        self.D = D
        self.grow = np.arange(D.shape[0], dtype=np.int32) if grow is None else grow
        self.ci = m["cell_index"]
        self.cells = m["cells"]
        self.n = len(self.cells)
        self.N = n_agents
        self.period = int(period)
        self.window = max(int(window), int(period) + 1)
        self.rng = np.random.default_rng(seed)
        nbr = mapmod.neighbors_lists(m)
        self.nbrs = [[int(self.ci[u]) for u in lst] for lst in nbr]
        self.pos = [int(p) for p in
                    self.rng.choice(self.n, size=n_agents, replace=False)]
        self.elapsed = [0] * n_agents
        self.goals = [None] * n_agents
        self.eps = list(self.rng.random(n_agents) * 0.5)
        self._plans = [[] for _ in range(n_agents)]   # future cells, index 0 = next
        self._t = 0
        # edge costs under the guidance weights, keyed (u, v) in compact indices
        self._w = self._edge_cost_lookup()

    def _edge_cost_lookup(self):
        edges, _ = mapmod.directed_edges(self.m)
        ci = self.ci
        w = {}
        for (u, v) in edges:
            w[(int(ci[u]), int(ci[v]))] = 1.0
        return w

    def set_weights(self, edge_weights):
        edges, _ = mapmod.directed_edges(self.m)
        ci = self.ci
        ew = np.ones(len(edges)) if edge_weights is None else np.asarray(edge_weights, float)
        self._w = {(int(ci[u]), int(ci[v])): float(ew[k])
                   for k, (u, v) in enumerate(edges)}

    def _row(self, g):
        return self.D[self.grow[g]]

    # ---- space-time A* ---------------------------------------------------
    def _plan_one(self, i, vres, eres):
        """Guidance-weighted space-time A* for agent i over the full window.

        The search always runs to the horizon `window` so that every cell the
        agent occupies -- including idling after an early goal arrival -- is
        checked against the reservation table. Moving costs the guidance weight
        of the traversed edge; waiting costs the free-flow time 1, except at
        the goal where it is free, so parking on the goal is the cheapest way
        to spend leftover horizon and the first expanded horizon state is
        optimal under the admissible heuristic D_w(.,g). Returns `window`
        cells, or None if the agent is boxed in (handled by _replan)."""
        W = self.window
        start, g = self.pos[i], self.goals[i]
        Drow = self._row(g)
        wl = self._w
        openh = [(float(Drow[start]), 0.0, start, 0)]
        best = {(start, 0): 0.0}
        came = {}
        end = None
        while openh:
            _, gc, v, t = heapq.heappop(openh)
            if best.get((v, t), 1e18) < gc - 1e-12:
                continue
            if t >= W:
                end = (v, t)
                break
            for u in self.nbrs[v] + [v]:
                if (u, t + 1) in vres:
                    continue                        # vertex conflict
                if u != v and (u, v, t + 1) in eres:
                    continue                        # swap conflict
                if u == v:
                    step = 0.0 if v == g else 1.0
                else:
                    step = wl[(v, u)]
                ng = gc + step
                if ng < best.get((u, t + 1), 1e18) - 1e-12:
                    best[(u, t + 1)] = ng
                    came[(u, t + 1)] = (v, t)
                    heapq.heappush(openh, (ng + float(Drow[u]), ng, u, t + 1))
        if end is None:
            return None                             # fully blocked: see _replan
        path = []
        s = end
        while s in came:
            path.append(s[0])
            s = came[s]
        path.reverse()
        return path

    # ---- one timestep ----------------------------------------------------
    def _replan(self):
        """Plan every agent in decreasing priority against a shared reservation
        table. An agent that finds no feasible window (fully boxed in by
        higher-priority reservations) is promoted to the front of the order and
        the round restarts: with an empty table, staying put is always
        feasible, so the promotion terminates with a conflict-free plan set."""
        order = sorted(range(self.N), key=lambda i: -(self.elapsed[i] + self.eps[i]))
        for _ in range(8):
            vres, eres = set(), set()
            failed = None
            for i in order:
                path = self._plan_one(i, vres, eres)
                if path is None:
                    failed = i
                    break
                self._plans[i] = path
                prev = self.pos[i]
                for t, u in enumerate(path, start=1):
                    vres.add((u, t))
                    if u != prev:
                        eres.add((prev, u, t))      # blocks the reverse traversal
                    prev = u
            if failed is None:
                return
            order.remove(failed)
            order.insert(0, failed)
        # last resort (should not be reached): everyone holds position
        for i in range(self.N):
            self._plans[i] = [self.pos[i]] * self.window

    def step(self):
        if self._t % self.period == 0:
            self._replan()
        k = self._t % self.period
        for i in range(self.N):
            self.pos[i] = self._plans[i][k]
        self._t += 1
