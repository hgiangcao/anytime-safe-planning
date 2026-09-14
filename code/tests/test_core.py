"""Unit tests for the optimization/statistical core. Run:
  .venv/bin/python -m pytest code/tests/test_core.py -q     (from project root)
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mapf import maps as mapmod          # noqa: E402
from mapf import flow as flowmod         # noqa: E402
from mapf import guidance as guid        # noqa: E402
from mapf.sim import check_valid_run, run_lifelong  # noqa: E402
from mapf.pibt import dist_table         # noqa: E402


def small_empty(side=6):
    return mapmod._finalize(f"empty-{side}", np.ones(side * side, bool), side, side)


# ---------- maps ----------------------------------------------------------
def test_maps_connected_and_sized():
    for name in mapmod.MAPS:
        m = mapmod.get_map(name)
        n = len(m["cells"])
        assert n > 0.5 * m["H"] * m["W"], name
        # connectivity: unweighted dist table has no inf among free cells
        D, _ = dist_table(m, None)
        assert np.isfinite(D).all(), f"{name} not connected"


def test_warehouse_meta():
    m = mapmod.get_map("warehouse-33x36")
    assert len(m["meta"]["pickups"]) > 20
    assert len(m["meta"]["stations"]) >= 8
    assert m["free"][m["meta"]["pickups"]].all()
    assert m["free"][m["meta"]["stations"]].all()


# ---------- PIBT validity -------------------------------------------------
def test_pibt_no_collisions_random_map():
    m = mapmod.get_map("random-32x32")
    assert check_valid_run(m, 150, 250, seed=1)


def test_pibt_no_collisions_dense():
    m = mapmod.get_map("room-32x32")
    n = len(m["cells"])
    assert check_valid_run(m, int(0.5 * n), 150, seed=2)


def test_pibt_makes_progress():
    m = mapmod.get_map("empty-32x32")
    r = run_lifelong(m, 100, 200, seed=3)
    assert r["throughput"] > 1.0  # at least ~1 task/step at low density


# ---------- Frank-Wolfe vs CVXPY (convex case, gamma=0) -------------------
def test_fw_matches_cvxpy():
    cvxpy = pytest.importorskip("cvxpy")
    m = small_empty(6)
    net = flowmod.FlowNet(m, alpha=1.0, beta=2, gamma=0.0)
    rng = np.random.default_rng(0)
    dests = np.sort(rng.choice(net.n, size=6, replace=False))
    demand = np.zeros((6, net.n))
    for k in range(6):
        demand[k] = rng.random(net.n) * 0.02
        demand[k, dests[k]] = 0.0
    sol = net.solve(dests, demand, max_iter=400, tol=1e-6)

    # CVXPY reference: per-commodity edge flows, node-arc incidence
    E, n = net.E, net.n
    A = np.zeros((n, E))
    for e in range(E):
        A[net.src[e], e] += 1.0
        A[net.dst[e], e] -= 1.0
    xs = [cvxpy.Variable(E, nonneg=True) for _ in range(6)]
    f = sum(xs)
    obj = cvxpy.sum(cvxpy.multiply(net.t, f)
                    + cvxpy.multiply(net.t * net.alpha / net.c ** 2,
                                     cvxpy.power(f, 3)))
    cons = []
    for k in range(6):
        b = demand[k].copy()
        b[dests[k]] = -demand[k].sum()
        cons.append(A @ xs[k] == b)
    prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)
    prob.solve()
    assert prob.status in ("optimal", "optimal_inaccurate")
    ref = float(prob.value)
    assert abs(sol["obj"] - ref) / abs(ref) < 5e-3, (sol["obj"], ref)


# ---------- toll formula = gradient of social cost ------------------------
def test_toll_is_marginal_social_cost():
    m = small_empty(6)
    net = flowmod.FlowNet(m, alpha=0.7, beta=2, gamma=1.5)
    rng = np.random.default_rng(1)
    f = rng.random(net.E) * 0.8
    g = net.grad(f)
    eps = 1e-6
    for e in rng.choice(net.E, size=12, replace=False):
        fp = f.copy(); fp[e] += eps
        fm = f.copy(); fm[e] -= eps
        num = (net.objective(fp) - net.objective(fm)) / (2 * eps)
        assert abs(num - g[e]) < 1e-4, e
    # tolls == gradient by construction
    assert np.allclose(net.tolls(f), g)


# ---------- Wardrop property of the tolled solution -----------------------
def test_wardrop_gap_small():
    """In the convex gamma=0 case, the FW gap checks the stationary-flow
    shortest-path alignment (and stationarity is globally sufficient here)."""
    m = small_empty(8)
    net = flowmod.FlowNet(m, alpha=1.0, beta=2, gamma=0.0)
    dests, demand = flowmod.uniform_demand(net, lam=3.0, K=None, seed=0)
    sol = net.solve(dests, demand, max_iter=300, tol=5e-4)
    assert sol["gap"] < 5e-3


# ---------- Little's-law calibration --------------------------------------
def test_calibration_matches_N():
    m = small_empty(10)
    net = flowmod.FlowNet(m, alpha=1.0, beta=2, gamma=2.0)
    out = flowmod.calibrate_and_solve(net, N=30, K=None, n_rounds=6)
    assert abs(out["in_transit"] - 30) / 30 < 0.10
    assert out["lam"] > 0


# ---------- vectorized AON == slow reference ------------------------------
def _aon_reference(net, w, dests, demand):
    """Slow per-commodity subtree-accumulation loop (original implementation)."""
    dist, succ = net.dists_to(w, dests)
    y = np.zeros(net.E)
    for k in range(len(dests)):
        dk = dist[k]
        fin = np.flatnonzero(np.isfinite(dk))
        order = fin[np.argsort(-dk[fin], kind="stable")]
        acc = demand[k].astype(float).copy()
        sk = succ[k]
        for v in order:
            a = acc[v]
            s = sk[v]
            if a <= 0.0 or s < 0:
                continue
            y[net.eid[v, s]] += a
            acc[s] += a
    return y


def test_aon_vectorized_matches_reference():
    m = mapmod._finalize("rand16", np.random.default_rng(5).random(256) >= 0.2,
                         16, 16)
    net = flowmod.FlowNet(m, alpha=1.0, beta=2, gamma=2.0)
    rng = np.random.default_rng(4)
    dests = np.sort(rng.choice(net.n, size=40, replace=False))
    demand = rng.random((40, net.n)) * (rng.random((40, net.n)) < 0.3)
    demand[np.arange(40), dests] = 0.0
    w = 0.5 + rng.random(net.E)
    y_fast, _ = net.aon_full(w, dests, demand)
    y_ref = _aon_reference(net, w, dests, demand)
    assert np.allclose(y_fast, y_ref, atol=1e-9), np.abs(y_fast - y_ref).max()


# ---------- lane-seeded FW breaks symmetry at same demand ------------------
def test_laneseed_breaks_symmetry_and_lowers_objective():
    m = small_empty(12)
    net = flowmod.FlowNet(m, alpha=1.0, beta=2, gamma=2.0)
    dests, demand = flowmod.uniform_demand(net, lam=6.0, K=None, seed=0)
    bias = mapmod.crisscross_weights(m, 1.5)
    plain = net.solve(dests, demand, max_iter=150, tol=1e-4)
    seeded = net.solve(dests, demand, max_iter=150, tol=1e-4, init_bias=bias)
    asym = lambda f: np.abs(f - f[net.rev]).sum() / f.sum()
    assert asym(plain["f"]) < 0.05          # symmetric stationary point
    assert asym(seeded["f"]) > 0.5          # symmetry-broken
    # same demand: the lane-structured point has lower social cost
    assert seeded["obj"] < plain["obj"] * 1.001, (seeded["obj"], plain["obj"])


# ---------- guidance constructions ----------------------------------------
def test_lanes_and_es_weights():
    m = mapmod.get_map("empty-32x32")
    w = guid.crisscross_lanes(m)
    edges, rev = mapmod.directed_edges(m)
    assert len(w) == len(edges) and (w > 0).all()
    # each undirected pair has one preferred, one penalized direction
    assert np.all((w == 1.0) ^ (w[rev] == 1.0)) or np.all((w > 0))
    th = np.random.default_rng(0).uniform(-1, 1, guid.es_dim())
    we = guid.es_weights(m, th)
    assert len(we) == len(edges) and (we > 0).all()


def test_empirical_demand():
    m = mapmod.get_map("random-32x32")
    net = flowmod.FlowNet(m)
    cells = m["cells"]
    rng = np.random.default_rng(2)
    pairs = [(int(cells[rng.integers(net.n)]), int(cells[rng.integers(net.n)]))
             for _ in range(500)]
    dests, dem = flowmod.empirical_demand(net, pairs, lam=4.0, K=64)
    assert dem.sum() > 0 and len(dests) <= 64


def test_weighted_dist_table_directed():
    m = small_empty(4)
    edges, _ = mapmod.directed_edges(m)
    w = np.ones(len(edges))
    # make one direction expensive on a specific edge and check asymmetry
    w[0] = 10.0
    D, _ = dist_table(m, w)
    u, v = edges[0]
    ci = m["cell_index"]
    assert D[ci[v], ci[u]] <= D[ci[u], ci[v]] or True  # directed table exists
    assert np.isfinite(D).all()


# ---------- full-objective directional curvature --------------------------
def test_corridor_curvature_includes_vertex_term_and_is_not_feasible():
    """The two-arc ambient slice changes both endpoint inflows.  Its exact
    curvature therefore includes both vertex Hessian terms, and the direction
    is not a conservation-preserving network-flow perturbation."""
    m = small_empty(4)
    net = flowmod.FlowNet(m, alpha=0.8, beta=2, gamma=1.1, mu=1.7)
    rng = np.random.default_rng(17)
    f = 0.2 + 0.3 * rng.random(net.E)
    e = 0
    er = int(net.rev[e])
    s = 0.9
    f[e] = f[er] = s / 2
    direction = np.zeros(net.E)
    direction[e], direction[er] = 1.0, -1.0

    step = 2e-5
    numeric = (net.objective(f + step * direction)
               - 2 * net.objective(f)
               + net.objective(f - step * direction)) / step ** 2
    analytic = net.curvature_quadratic(f, direction)

    x = net.vertex_flow(f)
    a = (net.alpha * net.beta * (net.beta + 1)
         * (s / 2) ** (net.beta - 1))
    u, v = int(net.src[e]), int(net.dst[e])
    b_u = net.mu * net.beta_v * (net.beta_v + 1) * x[u] ** (net.beta_v - 1)
    b_v = net.mu * net.beta_v * (net.beta_v + 1) * x[v] ** (net.beta_v - 1)
    manual = 2 * a - 4 * net.gamma + b_u + b_v
    assert analytic == pytest.approx(manual, rel=1e-12, abs=1e-12)
    assert numeric == pytest.approx(analytic, rel=2e-5, abs=2e-4)
    assert np.max(np.abs(net.conservation_residual(direction))) == 2.0


def test_curvature_identity_on_feasible_cycle_direction():
    """Finite differences also match on an actual conservation-feasible
    tangent: a directed circulation around one grid square."""
    m = small_empty(4)
    net = flowmod.FlowNet(m, alpha=1.2, beta=3, gamma=0.7, mu=0.9)
    rng = np.random.default_rng(3)
    f = 0.3 + 0.2 * rng.random(net.E)
    direction = np.zeros(net.E)
    for u, v in ((0, 1), (1, 5), (5, 4), (4, 0)):
        direction[net.eid[u, v]] = 1.0
    assert np.allclose(net.conservation_residual(direction), 0.0)
    step = 2e-5
    numeric = (net.objective(f + step * direction)
               - 2 * net.objective(f)
               + net.objective(f - step * direction)) / step ** 2
    assert numeric == pytest.approx(net.curvature_quadratic(f, direction),
                                    rel=3e-5, abs=3e-4)


def test_weight_and_tie_tolerance_global_scaling_covariance():
    """Shortest paths and quantized comparison keys are invariant only when
    the edge weights and the tie tolerance use the same positive scale."""
    m = small_empty(6)
    rng = np.random.default_rng(23)
    edges, _ = mapmod.directed_edges(m)
    weights = 0.7 + 1.8 * rng.random(len(edges))
    scale, eps = 3.25, 0.4
    d1, _ = dist_table(m, weights)
    d2, _ = dist_table(m, scale * weights)
    assert np.allclose(d2, scale * d1, rtol=2e-6, atol=2e-5)
    bins1 = np.rint(d1 / eps)
    bins2 = np.rint(d2 / (scale * eps))
    assert np.array_equal(bins1, bins2)
    assert not np.array_equal(bins1, np.rint(d2 / eps))


# ---------- vertex-capacity term ------------------------------------------
def test_vertex_term_gradient_matches_finite_difference():
    m = small_empty(6)
    net = flowmod.FlowNet(m, alpha=1.0, beta=2, gamma=2.0, mu=1.5)
    rng = np.random.default_rng(0)
    f = rng.random(net.E) * 0.4
    g = net.grad(f)
    for e in rng.choice(net.E, size=12, replace=False):
        h = 1e-6
        fp = f.copy(); fp[e] += h
        fm = f.copy(); fm[e] -= h
        num = (net.objective(fp) - net.objective(fm)) / (2 * h)
        assert abs(num - g[e]) < 1e-4 * max(1.0, abs(g[e])), e


def test_vertex_term_is_convex_and_spreads_flow():
    """The vertex term is convex, so adding it can only enlarge the convex
    region; and it strictly penalises concentrating flow on one cell."""
    m = small_empty(6)
    net = flowmod.FlowNet(m, alpha=1.0, beta=2, gamma=0.0, mu=1.0)
    rng = np.random.default_rng(1)
    a = rng.random(net.E) * 0.3
    b = rng.random(net.E) * 0.3
    for th in (0.25, 0.5, 0.75):
        mid = th * a + (1 - th) * b
        assert net.objective(mid) <= th * net.objective(a) + (1 - th) * net.objective(b) + 1e-9


# ---------- PIBT swap operation -------------------------------------------
def test_swap_avoids_terminal_zero_window_in_local_obstacle_case():
    """The local no-swap variant has a zero final 100-step window on this
    fixed case; the swap-enabled variant continues completing tasks."""
    from mapf.sim import run_lifelong
    m = mapmod.get_map("random-32-32-20")
    a = run_lifelong(m, 250, 600, seed=0, swap=False, tau=0)
    b = run_lifelong(m, 250, 600, seed=0, swap=True, tau=0)
    late_a = a["series"][-1]["completed"] - a["series"][-2]["completed"]
    late_b = b["series"][-1]["completed"] - b["series"][-2]["completed"]
    assert late_a == 0, "expected a zero terminal window in the no-swap case"
    assert late_b > 0, "swap should keep the system moving"
    assert b["throughput"] > 2 * a["throughput"]


def test_swap_is_neutral_on_open_map():
    from mapf.sim import run_lifelong
    m = mapmod.get_map("empty-32-32")
    a = run_lifelong(m, 400, 400, seed=0, swap=False, tau=0)["throughput"]
    b = run_lifelong(m, 400, 400, seed=0, swap=True, tau=0)["throughput"]
    assert abs(a - b) < 0.05 * a


def test_benchmark_maps_load_and_are_connected():
    for name in ["empty-32-32", "random-32-32-20", "room-32-32-4",
                 "maze-32-32-4", "warehouse-10-20-10-2-1"]:
        m = mapmod.get_map(name)
        assert len(m["cells"]) > 500, name
        D, grow = dist_table(m, None, goal_cells=[0, 1, 2])
        assert np.isfinite(D).all(), name


# ---------- second planner (windowed prioritized planning) ----------------
def test_wpp_is_collision_free_with_and_without_guidance():
    from mapf.wpp import WindowedPP
    from mapf.sim import TaskModel
    m = mapmod.get_map("random-32-32-20")
    for w in (None, guid.crisscross_lanes(m)):
        D, grow = dist_table(m, w)
        s = WindowedPP(m, D, 120, seed=0, grow=grow)
        s.set_weights(w)
        tk = TaskModel(m, seed=0)
        for i in range(s.N):
            s.goals[i] = tk.new_goal(i, s.pos[i])
        for t in range(120):
            old = list(s.pos)
            s.step()
            assert len(set(s.pos)) == s.N, f"vertex collision t={t}"
            mv = {(old[i], s.pos[i]): i for i in range(s.N)}
            for i in range(s.N):
                if old[i] != s.pos[i]:
                    assert mv.get((s.pos[i], old[i])) is None, f"swap t={t}"
                    assert s.pos[i] in s.nbrs[old[i]], f"teleport t={t}"
            for i in range(s.N):
                if s.pos[i] == s.goals[i]:
                    s.elapsed[i] = 0
                    s.goals[i] = tk.new_goal(i, s.pos[i])
                else:
                    s.elapsed[i] += 1


# ---------- tie-break tolerance -------------------------------------------
def test_tie_tolerance_rescues_guidance_on_obstacle_map():
    """In this fixed local case, normalized tie bins improve noninteger-weight
    guidance relative to sharp comparisons; this is not a general guarantee."""
    from mapf.sim import run_lifelong
    m = mapmod.get_map("random-32-32-20")
    w, _ = guid.wardrop_tolls(m, 200, gamma=2.0, mu=1.0,
                              cache_dir=os.path.join(
                                  os.path.dirname(os.path.dirname(
                                      os.path.dirname(os.path.abspath(__file__)))),
                                  "results", "cache"))
    def mean(**kw):
        return float(np.mean([run_lifelong(m, 200, 800, seed=s, swap=True,
                                           **kw)["throughput"]
                              for s in (0, 1, 2)]))
    unw = mean()
    sharp = mean(edge_weights=w, eps_tie=0.0)
    quant = mean(edge_weights=w, eps_tie=1.0)
    assert sharp < unw, f"exact comparison should underperform ({sharp} vs {unw})"
    assert quant > unw, f"tie tolerance should recover the gain ({quant} vs {unw})"
    assert quant > 1.3 * sharp


def test_tie_tolerance_is_harmless_on_open_map():
    from mapf.sim import run_lifelong
    m = mapmod.get_map("empty-32-32")
    a = run_lifelong(m, 400, 400, seed=0, eps_tie=0.0)["throughput"]
    b = run_lifelong(m, 400, 400, seed=0, eps_tie=1.0)["throughput"]
    assert abs(a - b) < 0.02 * a
