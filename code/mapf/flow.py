"""Multi-commodity congestion flow model + Frank-Wolfe solver + marginal-cost tolls.

Model (per NOTES.md spec):
  Directed grid graph over free cells; per-directed-edge flow f_e >= 0
  (agents/timestep). BPR latency  l_e(f) = t_e (1 + alpha (f/c_e)^beta),
  t_e = c_e = 1 on grids. Head-on (bidirectional) conflict cost couples an
  edge with its reverse. Model social-cost program:

      min_f  J(f) = sum_e f_e l_e(f_e) + gamma * sum_e f_e f_{rev(e)}
      s.t.   per-commodity flow conservation, f = sum_k f^k, f^k >= 0.

  The BPR part of J is convex; the cross term is a bilinear penalty which is
  convex only when dominated by the BPR curvature (gamma=0 recovers a fully
  convex program -- verified against CVXPY in tests). We solve with
  Frank-Wolfe (one Dijkstra per commodity per iteration, exact line search);
  the objective decreases monotonically and we report the final FW gap.  For
  gamma > 0 this is a nonconvex first-order stationarity calculation, not a
  certificate of global system optimality.

  Guidance weights = Pigouvian marginal social cost at the optimum:
      w_e = dJ/df_e = t_e (1 + alpha (beta+1) (f_e/c_e)^beta) + 2 gamma f_rev(e)

Commodities are aggregated by destination. The vectorized all-or-nothing
assignment makes the FULL destination set (~1000 commodities on 32x32 maps)
affordable, so no subsampling approximation is used offline (K=None default);
the online variant aggregates the observed OD log by destination.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize_scalar

from . import maps as mapmod


class FlowNet:
    """Compact-index directed graph over free cells of a map."""

    def __init__(self, m, alpha=1.0, beta=2, gamma=2.0, mu=0.0, beta_v=None,
                 form="bpr", rho_max=0.97):
        self.m = m
        self.alpha, self.beta, self.gamma = float(alpha), int(beta), float(gamma)
        # Vertex-capacity (cell-exclusion) term. Road networks price only
        # edges; a grid cell holds at most one agent, so the binding constraint
        # on obstacle-dense maps is vertex throughput, not edge throughput.
        # x_v = total inflow to v; delay mu (x_v/c_v)^beta_v is added to every
        # edge entering v. Convex in f (x^(beta_v+1) composed with a linear
        # map), so it also enlarges the region where J is convex despite the
        # indefinite head-on term.
        self.mu = float(mu)
        self.beta_v = int(beta if beta_v is None else beta_v)
        # Latency family. "bpr" is the road-traffic standard
        #   l(f) = t (1 + alpha (f/c)^beta),
        # which stays finite as f -> c. Grid MAPF has a hard one-agent-per-cell
        # exclusion, so throughput does not merely degrade at capacity, it
        # collapses; "queue" uses the Kleinrock/M-M-1 delay
        #   l(f) = t / (1 - f/c),
        # which diverges at capacity and therefore forces the calibrated demand
        # to stay inside what the map can actually carry. Convex on f < c.
        self.form = form
        self.rho_max = float(rho_max)
        edges, rev = mapmod.directed_edges(m)
        ci = m["cell_index"]
        self.n = len(m["cells"])
        self.src = ci[edges[:, 0]].astype(np.int64)
        self.dst = ci[edges[:, 1]].astype(np.int64)
        self.rev = rev.astype(np.int64)
        self.E = len(self.src)
        self.t = np.ones(self.E)
        self.c = np.ones(self.E)
        self.cv = np.ones(self.n)
        # dense edge-id lookup (n~1000 so ~4MB)
        self.eid = -np.ones((self.n, self.n), dtype=np.int32)
        self.eid[self.src, self.dst] = np.arange(self.E, dtype=np.int32)

    # ---- costs -----------------------------------------------------------
    def _rho(self, f, c):
        return np.minimum(np.maximum(f, 0.0) / c, self.rho_max)

    def latency(self, f):
        if self.form == "queue":
            return self.t / (1.0 - self._rho(f, self.c))
        return self.t * (1.0 + self.alpha * (f / self.c) ** self.beta)

    def _edge_marginal(self, f):
        """d/df [ f * l(f) ] for the chosen latency family."""
        if self.form == "queue":
            r = self._rho(f, self.c)
            return self.t / (1.0 - r) ** 2
        return self.t * (1.0 + self.alpha * (self.beta + 1) * (f / self.c) ** self.beta)

    def _vertex_delay(self, x):
        if self.form == "queue":
            return self.mu / (1.0 - self._rho(x, self.cv))
        return self.mu * (x / self.cv) ** self.beta_v

    def _vertex_marginal(self, x):
        if self.form == "queue":
            r = self._rho(x, self.cv)
            return self.mu / (1.0 - r) ** 2
        return self.mu * (self.beta_v + 1) * (x / self.cv) ** self.beta_v

    def vertex_flow(self, f):
        """x_v = total flow entering vertex v."""
        return np.bincount(self.dst, weights=f, minlength=self.n)

    def objective(self, f):
        J = float(np.sum(f * self.latency(f)))
        if self.gamma:
            J += self.gamma * float(np.sum(f * f[self.rev]))
        if self.mu:
            x = self.vertex_flow(f)
            J += float(np.sum(x * self._vertex_delay(x)))
        return J

    def grad(self, f):
        g = self._edge_marginal(f)
        if self.gamma:
            g = g + 2.0 * self.gamma * f[self.rev]
        if self.mu:
            gv = self._vertex_marginal(self.vertex_flow(f))
            g = g + gv[self.dst]
        return g

    def tolls(self, f):
        """Marginal social cost = guidance edge weights (aligned with directed_edges)."""
        return self.grad(f)

    def conservation_residual(self, h):
        """Return node-balance residuals for one or more edge-flow directions.

        ``h`` may have shape ``(E,)`` or ``(K,E)``.  A commodity-wise affine
        feasible direction must have zero residual in every row (in addition
        to respecting active nonnegativity constraints).  This helper makes
        explicit that changing only an arc and its reverse antisymmetrically
        is generally *not* a feasible network-flow direction.
        """
        hh = np.asarray(h, dtype=float)
        one = hh.ndim == 1
        if one:
            hh = hh[None, :]
        if hh.ndim != 2 or hh.shape[1] != self.E:
            raise ValueError(f"h must have shape (E,) or (K,E), E={self.E}")
        out = np.zeros((hh.shape[0], self.n), dtype=float)
        for k, row in enumerate(hh):
            np.add.at(out[k], self.src, row)
            np.add.at(out[k], self.dst, -row)
        return out[0] if one else out

    def curvature_quadratic(self, f, h):
        """Compute ``h.T @ Hessian(J(f)) @ h`` for the BPR objective.

        For an affine commodity-feasible direction this is the second
        derivative of the model objective restricted to the feasible tangent
        space.  The identity includes the endpoint-load curvature omitted by
        the original manuscript's two-arc calculation.  It deliberately does
        not infer feasibility or global optimality from ambient curvature.
        """
        if self.form != "bpr":
            raise ValueError("curvature_quadratic is defined here for BPR only")
        f = np.asarray(f, dtype=float)
        h = np.asarray(h, dtype=float)
        if f.shape != (self.E,) or h.shape != (self.E,):
            raise ValueError(f"f and h must both have shape ({self.E},)")
        a = (self.t * self.alpha * self.beta * (self.beta + 1)
             * np.power(f, self.beta - 1) / np.power(self.c, self.beta))
        value = float(np.dot(a, h * h))
        if self.gamma:
            # The directed sum counts each unordered corridor twice.
            value += 2.0 * self.gamma * float(np.dot(h, h[self.rev]))
        if self.mu:
            x = self.vertex_flow(f)
            b = (self.mu * self.beta_v * (self.beta_v + 1)
                 * np.power(x, self.beta_v - 1)
                 / np.power(self.cv, self.beta_v))
            dx = np.bincount(self.dst, weights=h, minlength=self.n)
            value += float(np.dot(b, dx * dx))
        return value

    def experienced_latency(self, f):
        """Per-edge latency an agent experiences (BPR + opposing-flow interference)."""
        lat = self.latency(f)
        if self.gamma:
            lat = lat + self.gamma * f[self.rev]
        if self.mu:
            lat = lat + self._vertex_delay(self.vertex_flow(f))[self.dst]
        return lat

    def total_time(self, f):
        return float(np.sum(f * self.experienced_latency(f)))

    # ---- shortest paths --------------------------------------------------
    def _csrT(self, w):
        """CSR of the transposed graph with edge weights w (for dist-to-dest)."""
        return csr_matrix((w, (self.dst, self.src)), shape=(self.n, self.n))

    def dists_to(self, w, dest_idx):
        """dist[k, v] = shortest w-dist from node v to dest_idx[k]; also successors."""
        d, pred = dijkstra(self._csrT(w), directed=True, indices=dest_idx,
                           return_predecessors=True)
        return d, pred  # pred[k, v] = next hop of v toward dest k (-9999 none)

    # ---- all-or-nothing assignment --------------------------------------
    def aon_full(self, w, dests, demand):
        """Exact all-or-nothing assignment: route each commodity's demand along
        the shortest-path tree (subtree accumulation in decreasing-dist order),
        vectorized across commodities.

        Correctness: with strictly positive weights, dist[v] > dist[succ[v]],
        so processing nodes of each commodity in decreasing-distance order
        guarantees every node is processed before its successor. Column j of
        the per-commodity descending-order matrix is processed in one
        vectorized step over all commodities (rows write disjoint acc slots;
        np.add.at handles duplicate edge ids).

        Returns (edge_flow, total_shortest_cost)."""
        dist, succ = self.dists_to(w, dests)
        K, n = dist.shape
        eid = self.eid
        fin = np.isfinite(dist)
        sp_cost = float(np.sum(np.where(fin, demand * np.where(fin, dist, 0.0), 0.0)))
        # per-commodity node order by decreasing distance (inf sorts first ->
        # skipped via succ<0 mask; maps are connected so this rarely triggers)
        order = np.argsort(-dist, axis=1, kind="stable")      # (K, n)
        acc = demand.astype(float).copy()                     # (K, n)
        y = np.zeros(self.E)
        rows = np.arange(K)
        succ_ok = succ >= 0
        for j in range(n):
            v = order[:, j]                                   # (K,)
            ok = succ_ok[rows, v]
            if not ok.any():
                continue
            rk = rows[ok]
            vk = v[ok]
            a = acc[rk, vk]
            nz = a > 0.0
            if not nz.any():
                continue
            rk, vk, a = rk[nz], vk[nz], a[nz]
            s = succ[rk, vk]
            np.add.at(y, eid[vk, s], a)
            np.add.at(acc, (rk, s), a)
        return y, sp_cost

    # ---- Frank-Wolfe -----------------------------------------------------
    def solve(self, dests, demand, max_iter=100, tol=1e-3, f0=None,
              init_bias=None, verbose=False):
        """Frank-Wolfe on J. Returns dict(f, obj, gap, iters).

        init_bias: optional per-edge multiplier applied to the free-flow
        gradient for the INITIAL all-or-nothing assignment only. With the
        head-on coupling term (gamma>0) J is nonconvex and the direction-
        symmetric flow is a stationary point FW cannot leave from a symmetric
        start; a direction-biased initial assignment (e.g. mild alternating
        lanes) selects a symmetry-broken stationary point with one-way
        structure. Subsequent FW iterations use the true gradient."""
        dests = np.asarray(dests, dtype=np.int64)
        demand = np.asarray(demand, dtype=float)
        if f0 is None:
            w0 = self.grad(np.zeros(self.E))
            if init_bias is not None:
                w0 = w0 * init_bias
            f, _ = self.aon_full(w0, dests, demand)
        else:
            f = f0.copy()
        gap = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            g = self.grad(f)
            y, _ = self.aon_full(g, dests, demand)
            d = y - f
            gTf = float(g @ f)
            gap = float(g @ (f - y)) / max(gTf, 1e-12)
            if gap < tol:
                break
            res = minimize_scalar(lambda th: self.objective(f + th * d),
                                  bounds=(0.0, 1.0), method="bounded",
                                  options={"xatol": 1e-5})
            th = float(res.x)
            if self.objective(f + th * d) > self.objective(f):
                th = 2.0 / (it + 2.0)  # fallback step
            f = f + th * d
            if verbose and it % 10 == 0:
                print(f"  FW it={it} J={self.objective(f):.4f} gap={gap:.2e}")
        # Recompute at the iterate actually returned.  When the loop reaches
        # max_iter, the old implementation reported the pre-update gap beside
        # the post-update flow.  Besides fixing that mismatch, retain the
        # absolute and normalized aggregate linearized-routing residual used by
        # Eq. (approx-align) in the manuscript.
        residual = self.linearization_residual(f, dests, demand)
        return dict(f=f, obj=self.objective(f), gap=residual["relative_gap"],
                    linear_cost=residual["returned_linear_cost"],
                    linear_optimum=residual["minimum_linear_cost"],
                    linear_residual=residual["absolute_residual"],
                    linear_factor_bound=residual["multiplicative_bound"],
                    iters=it)

    def linearization_residual(self, f, dests, demand):
        """Aggregate fixed-gradient routing residual at ``f``.

        With g=grad J(f) and y the all-or-nothing minimum of g^T z over the
        feasible flow polytope, r=g^T(f-y)/g^T f implies
        g^T f <= g^T y/(1-r) for r<1.  This is an aggregate statement; it is
        neither a nonconvex objective bound nor a claim that every used path is
        exactly shortest when r is nonzero.
        """
        f = np.asarray(f, dtype=float)
        g = self.grad(f)
        y, _ = self.aon_full(g, np.asarray(dests, dtype=np.int64),
                            np.asarray(demand, dtype=float))
        returned = float(g @ f)
        optimum = float(g @ y)
        absolute = max(0.0, returned - optimum)
        relative = absolute / max(returned, 1e-12)
        factor = float(1.0 / (1.0 - relative)) if relative < 1.0 else float("inf")
        return {"returned_linear_cost": returned,
                "minimum_linear_cost": optimum,
                "absolute_residual": absolute,
                "relative_gap": relative,
                "multiplicative_bound": factor}


# ---- demand construction -------------------------------------------------
def uniform_demand(net, lam, K=None, seed=0):
    """Uniform OD over free-cell ordered pairs, aggregated by destination.

    Returns (dests, demand (K,n)) with total demand = lam (after K-subsampling
    rescale)."""
    n = net.n
    rng = np.random.default_rng(seed)
    if K is None or K >= n:
        dests = np.arange(n)
        scale = 1.0
    else:
        dests = np.sort(rng.choice(n, size=K, replace=False))
        scale = n / K
    per_pair = lam / (n * (n - 1))
    demand = np.full((len(dests), n), per_pair * scale)
    demand[np.arange(len(dests)), dests] = 0.0
    return dests, demand


def warehouse_demand(net, lam, seed=0, K_pick=64):
    """Alternating pickup->station and station->pickup legs, equal rates."""
    m = net.m
    ci = m["cell_index"]
    picks = ci[m["meta"]["pickups"]].astype(np.int64)
    stats = ci[m["meta"]["stations"]].astype(np.int64)
    rng = np.random.default_rng(seed)
    if len(picks) > K_pick:
        pick_d = np.sort(rng.choice(picks, size=K_pick, replace=False))
        pscale = len(picks) / K_pick
    else:
        pick_d, pscale = picks, 1.0
    dests = np.concatenate([stats, pick_d])
    demand = np.zeros((len(dests), net.n))
    # legs into stations: from every pickup, rate lam/2 total
    r1 = (lam / 2.0) / (len(picks) * len(stats))
    for i in range(len(stats)):
        demand[i, picks] = r1
    # legs into pickups: from every station
    r2 = (lam / 2.0) / (len(stats) * len(picks)) * pscale
    for j in range(len(pick_d)):
        demand[len(stats) + j, stats] = r2
    return dests, demand


def make_demand(net, lam, K=None, seed=0):
    if net.m["name"].startswith("warehouse"):
        # ``K`` is the pickup-destination resolution for warehouse demand;
        # every station destination is always retained.  None preserves the
        # historical 64-pickup default.
        return warehouse_demand(net, lam, seed=seed,
                                K_pick=64 if K is None else int(K))
    return uniform_demand(net, lam, K=K, seed=seed)


# ---- Little's-law calibration + full pipeline ---------------------------
def calibrate_and_solve(net, N, K=None, seed=0, iters_cal=40, iters_final=100,
                        n_rounds=5, tol=1e-3, init_bias=None, verbose=False):
    """Find demand scale lam s.t. agents-in-transit (Little's law) == N,
    then solve to final tolerance. Returns dict with f, tolls, lam
    (= predicted throughput, tasks/timestep), obj, gap.
    init_bias: see FlowNet.solve (symmetry-breaking initial assignment)."""
    # initial guess: lam0 = N / mean unweighted OD distance
    dests, demand = make_demand(net, 1.0, K=K, seed=seed)
    dist, _ = net.dists_to(net.t, np.asarray(dests, dtype=np.int64))
    wsum = float(np.sum(demand * np.where(np.isfinite(dist), dist, 0.0)))
    Lbar = wsum / max(float(demand.sum()), 1e-12)
    lam = N / max(Lbar, 1.0)
    f0 = None
    for r in range(n_rounds):
        dests, demand = make_demand(net, lam, K=K, seed=seed)
        sol = net.solve(dests, demand, max_iter=iters_cal, tol=tol, f0=f0,
                        init_bias=init_bias)
        f0 = sol["f"]
        T = net.total_time(f0)  # = agents in transit by Little's law
        ratio = N / max(T, 1e-9)
        if verbose:
            print(f" cal round {r}: lam={lam:.3f} in-transit={T:.1f} -> x{ratio:.3f}")
        if abs(ratio - 1.0) < 0.02:
            break
        damped = ratio ** 0.7
        f0 = f0 * damped
        lam *= damped
    dests, demand = make_demand(net, lam, K=K, seed=seed)
    sol = net.solve(dests, demand, max_iter=iters_final, tol=tol, f0=f0)
    f = sol["f"]
    return dict(f=f, tolls=net.tolls(f), lam=lam, obj=sol["obj"], gap=sol["gap"],
                linear_cost=sol["linear_cost"],
                linear_optimum=sol["linear_optimum"],
                linear_residual=sol["linear_residual"],
                linear_factor_bound=sol["linear_factor_bound"],
                iters=sol["iters"], in_transit=net.total_time(f),
                mean_latency=net.total_time(f) / max(lam, 1e-12))


def empirical_demand(net, od_pairs, lam, K=128, seed=0):
    """Demand from observed (origin,dest) grid-id pairs (online variant).

    Aggregates by destination; keeps the K most frequent destinations and
    rescales so total demand = lam."""
    ci = net.m["cell_index"]
    od = np.asarray(od_pairs, dtype=np.int64)
    o = ci[od[:, 0]].astype(np.int64)
    d = ci[od[:, 1]].astype(np.int64)
    dcnt = np.bincount(d, minlength=net.n)
    dests = np.argsort(-dcnt, kind="stable")[:K]
    dests = np.sort(dests[dcnt[dests] > 0])
    dset = {int(x): i for i, x in enumerate(dests)}
    demand = np.zeros((len(dests), net.n))
    kept = 0
    for oo, dd in zip(o, d):
        i = dset.get(int(dd))
        if i is not None:
            demand[i, oo] += 1.0
            kept += 1
    if kept == 0:
        return None, None
    # rescale so total demand = lam (kept commodities stand in for the full
    # task flow; dropped-destination load is redistributed proportionally).
    # NOTE(verification fix): was lam/kept*(len(od)/kept), which over-scales
    # by len(od)/kept when destinations are dropped; inactive in all reported
    # experiments (ONLINE_CFG K=100000 keeps every destination, factor = 1).
    demand *= lam / kept
    return dests, demand
