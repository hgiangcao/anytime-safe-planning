"""PIBT (Priority Inheritance with Backtracking, Okumura et al. IJCAI-19)
with the *swap* operation of Okumura (IJCAI-23, Alg. 4), reimplemented for
lifelong MAPF on 4-connected grids.

- Dynamic priorities: eta_i = timesteps since current goal was assigned,
  + fixed unique tie-break epsilon_i; reset on goal completion.
- Each timestep, undecided agents are processed in decreasing priority;
  PIBT(a) tries candidate vertices (4 neighbors + stay) in increasing
  heuristic distance-to-goal order (weighted Dijkstra distance table),
  with priority inheritance (recursing into lower-priority occupants)
  and backtracking on failure. Vertex and swap (edge) conflicts prevented.
- SWAP OPERATION (`swap=True`, default). In this repository's no-swap
  implementation, corridor/dead-end tests can reach terminal zero-completion
  windows within the measured horizon. Okumura (IJCAI-23) describes a swap
  extension that reverses the
  candidate order for an agent that must exchange places with a neighbour:

    C <- neigh(v_i) u {v_i}, sorted by dist(., g_i)
    j <- swap_required_and_possible(i, C[0])
    if j != None: reverse C
    ... standard PIBT body ...
    if the taken vertex is C[0] (of the reversed list) and j is undecided:
        pull j into v_i

  The detector runs two emulations (Okumura IJCAI-23, "Pattern Detector
  Implementation"), gated on deg(C[0]) <= 2:
    required: advance i into j's cell and j onward, ignoring other agents;
      NOT required once j's cell has degree > 2; required once j's cell has
      degree 1, or once i reaches g_i while j's best neighbour toward g_j
      is g_i.
    possible: reverse the emulation (j into i's cell, i onward); possible
      once i's cell has degree > 2, impossible once i's cell has degree 1.

Distance tables come from `dist_table()`: dist_to[g][v] = shortest-path
distance from v to g in the DIRECTED guidance-weighted graph (weights on
directed edges = guidance / toll weights; unweighted = all ones). Pass
`goal_cells` to build rows only for a restricted goal set (large maps).
"""
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import maps as mapmod

sys.setrecursionlimit(1_000_000)

_EMU_CAP = 256          # step cap for the swap emulations (guards cyclic corridors)


def dist_table(m, edge_weights=None, goal_cells=None):
    """Distance rows toward goal cells.

    Returns (D, grow) where D[grow[g], v] = weighted distance from free node v
    to free node g (compact indices). With goal_cells=None every free cell is a
    goal and grow is the identity; otherwise only the listed compact indices
    get a row (grow[g] = -1 elsewhere), which keeps large maps tractable.
    edge_weights aligned with maps.directed_edges(m); None = unweighted."""
    edges, _ = mapmod.directed_edges(m)
    ci = m["cell_index"]
    n = len(m["cells"])
    w = np.ones(len(edges)) if edge_weights is None else np.asarray(edge_weights, float)
    src = ci[edges[:, 0]].astype(np.int64)
    dst = ci[edges[:, 1]].astype(np.int64)
    gT = csr_matrix((w, (dst, src)), shape=(n, n))
    if goal_cells is None:
        D = dijkstra(gT, directed=True)
        grow = np.arange(n, dtype=np.int32)
    else:
        gc = np.unique(np.asarray(goal_cells, dtype=np.int64))
        D = dijkstra(gT, directed=True, indices=gc)
        grow = -np.ones(n, dtype=np.int32)
        grow[gc] = np.arange(len(gc), dtype=np.int32)
    return D.astype(np.float32), grow


class PIBT:
    """One lifelong PIBT instance on a map with a fixed distance table."""

    def __init__(self, m, D, n_agents, seed=0, grow=None, swap=True, tau=20,
                 D_free=None, eps_tie=1.0):
        self.m = m
        if isinstance(D, tuple):            # accept dist_table()'s (D, grow)
            D, grow = D
        self.D = D                      # (n_goal_rows, n) compact-index distances
        self.grow = np.arange(D.shape[0], dtype=np.int32) if grow is None else grow
        self.swap = bool(swap)
        # Stall escape: an agent blocked for more than `tau` consecutive steps
        # demotes "wait" to its last resort (and, when a guidance-free table
        # D_free is supplied, orders its moves by the UNWEIGHTED heuristic).
        # Guidance changes only the preference order in this implementation.
        # The finite-horizon experiments treat this as an ablation parameter;
        # no general liveness statement follows from the local tests.
        self.tau = int(tau) if tau else 0
        self.D_free = D_free
        # Randomised tie-breaking tolerance. With w == 1 the distance table is
        # integer-valued, so several candidate moves tie and PIBT's random
        # tie-break makes the local rule stochastic. Noninteger guidance removes
        # many such ties and can degrade finite-horizon performance. Comparing
        # candidates in `eps_tie` bins is the evaluated remedy; this is a
        # normalized protocol parameter, not a general liveness proof.
        self.eps_tie = float(eps_tie)
        self.ci = m["cell_index"]
        self.cells = m["cells"]
        self.n = len(self.cells)
        if n_agents > self.n:
            raise ValueError("more agents than free cells")
        self.N = n_agents
        self.rng = np.random.default_rng(seed)
        # neighbor lists in COMPACT indices, including 'stay' (self) appended last
        nbr = mapmod.neighbors_lists(m)
        self.nbrs = [[int(self.ci[u]) for u in lst] for lst in nbr]
        self.deg = np.array([len(x) for x in self.nbrs], dtype=np.int8)
        # start positions: random distinct free cells (compact idx)
        self.pos = list(self.rng.choice(self.n, size=n_agents, replace=False))
        self.pos = [int(p) for p in self.pos]
        self.eps = list(self.rng.random(n_agents) * 0.5)  # unique-ish tiebreaks
        self.elapsed = [0] * n_agents
        self.stall = [0] * n_agents
        self.occ = [-1] * self.n
        for i, p in enumerate(self.pos):
            self.occ[p] = i
        self.next_pos = [-1] * n_agents
        self.next_occ = [-1] * self.n
        self.goals = [None] * n_agents

    def _row(self, g):
        return self.D[self.grow[g]]

    def _row_free(self, g):
        return self.D_free[self.grow[g]] if self.D_free is not None else None

    # ---- swap detector (Okumura IJCAI-23) --------------------------------
    def _other_nbr(self, v, exclude):
        """The unique neighbour of v other than `exclude` (deg(v) == 2)."""
        for u in self.nbrs[v]:
            if u != exclude:
                return u
        return -1

    def _swap_required(self, vi, vj, gi, gj):
        """Emulate: i advances into j's cell, j moves on. Ignores other agents."""
        for _ in range(_EMU_CAP):
            dj = self.deg[vj]
            if dj > 2:
                return False                       # branching: pushing works
            if dj == 1:
                return True                        # dead end: must swap
            if vi == gi:
                # j's best neighbour toward its own goal is i's goal
                rj = self._row(gj)
                best = min(self.nbrs[vj], key=lambda u: rj[u])
                if best == gi:
                    return True
            nxt = self._other_nbr(vj, vi)
            if nxt < 0:
                return True
            vi, vj = vj, nxt
        return False

    def _swap_possible(self, vi, vj):
        """Reverse emulation: j advances into i's cell, i moves on."""
        for _ in range(_EMU_CAP):
            di = self.deg[vi]
            if di > 2:
                return True                        # room to step aside
            if di == 1:
                return False
            nxt = self._other_nbr(vi, vj)
            if nxt < 0:
                return False
            vj, vi = vi, nxt
        return False

    def _swap_partner(self, i, c0):
        """Return the agent to swap with (or -1) when i's best candidate is c0."""
        if self.deg[c0] > 2:
            return -1
        k = self.occ[c0]
        if k < 0 or k == i:
            return -1
        gi, gk = self.goals[i], self.goals[k]
        if gk is None:
            return -1
        if not self._swap_required(self.pos[i], c0, gi, gk):
            return -1
        if not self._swap_possible(self.pos[i], c0):
            return -1
        return k

    # ---- one timestep ----------------------------------------------------
    def step(self):
        N = self.N
        order = sorted(range(N), key=lambda i: -(self.elapsed[i] + self.eps[i]))
        self.next_pos = [-1] * N
        nocc = self.next_occ
        for v in range(self.n):
            nocc[v] = -1
        for i in order:
            if self.next_pos[i] < 0:
                self._pibt(i, -1)
        # commit
        occ, pos, stall = self.occ, self.pos, self.stall
        for i in range(N):
            occ[pos[i]] = -1
        for i in range(N):
            p = self.next_pos[i]
            stall[i] = 0 if p != pos[i] else stall[i] + 1
            pos[i] = p
            occ[p] = i

    def _pibt(self, i, parent):
        pos, npos, nocc, occ = self.pos, self.next_pos, self.next_occ, self.occ
        v = pos[i]
        g = self.goals[i]
        Drow = self._row(g)
        rnd = self.rng.random
        q = self.eps_tie
        if q > 0:
            def key(u):
                return (round(float(Drow[u]) / q), rnd())
        else:
            def key(u):
                return (Drow[u], rnd())
        if self.tau and self.stall[i] > self.tau:
            cands = sorted(self.nbrs[v], key=key) + [v]
        else:
            cands = sorted(self.nbrs[v] + [v], key=key)
        # --- swap operation: reverse the preference order (Alg. 4, lines 3-4)
        jswap = -1
        if self.swap and cands[0] != v:
            jswap = self._swap_partner(i, cands[0])
            if jswap >= 0:
                cands = cands[::-1]
        pv = pos[parent] if parent >= 0 else -1
        for u in cands:
            if nocc[u] >= 0:
                continue                      # vertex conflict with decided agent
            if u == pv:
                continue                      # would swap with the pusher
            k = occ[u]
            if k >= 0 and k != i:
                if npos[k] >= 0:
                    if npos[k] == v:
                        continue              # swap with decided agent
                else:
                    # priority inheritance: ask k to vacate
                    nocc[u] = i
                    npos[i] = u
                    if self._pibt(k, i):
                        self._pull(i, u, cands, jswap, v)
                        return True
                    # backtrack: k could not vacate (k now stays at u and owns
                    # the nocc[u] claim); clear only our own stale claims
                    if nocc[u] == i:
                        nocc[u] = -1
                    npos[i] = -1
                    continue
            nocc[u] = i
            npos[i] = u
            self._pull(i, u, cands, jswap, v)
            return True
        npos[i] = v
        nocc[v] = i
        return False

    def _pull(self, i, u, cands, jswap, v):
        """Alg. 4 line 7: having backed off to the reversed-list head, agent i
        pulls its swap partner into the vertex it just vacated."""
        if jswap < 0 or u != cands[0]:
            return
        if self.next_pos[jswap] >= 0 or self.next_occ[v] >= 0:
            return
        self.next_pos[jswap] = v
        self.next_occ[v] = jswap

    # ---- lifelong task layer --------------------------------------------
    def assign_goal(self, i):
        """Override in task-model wrappers; uniform-random by default."""
        while True:
            g = int(self.rng.integers(self.n))
            if g != self.pos[i]:
                return g
