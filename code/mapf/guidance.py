"""Guidance-weight constructors: handcrafted lanes, Wardrop tolls, ES params."""
import json
import os
import time
import numpy as np

from . import maps as mapmod
from . import flow as flowmod


def unweighted(m):
    return None


def crisscross_lanes(m, penalty=5.0):
    """Handcrafted alternating one-way lanes (the strong human baseline in
    GGO): see maps.crisscross_weights."""
    return mapmod.crisscross_weights(m, penalty=penalty)


SEED_BIAS_PENALTY = 1.5   # mild lane bias for the symmetry-breaking init


def _toll_key(m, N, alpha, beta, gamma, K, seed, seed_lanes, tag, mu=0.0,
              form="bpr"):
    sl = "_L" if seed_lanes else ""
    mtag = "" if not mu else f"_m{mu}"
    ftag = "" if form == "bpr" else f"_{form}"
    return (f"tolls_{m['name']}_N{N}_a{alpha}_b{beta}_g{gamma}_K{K}_s{seed}"
            f"{sl}{mtag}{ftag}{tag}")


def wardrop_tolls(m, N, alpha=1.0, beta=2, gamma=2.0, mu=0.0, K=None, seed=0,
                  seed_lanes=True, cache_dir=None, tag="", form="bpr"):
    """Find a stationary-flow candidate calibrated to N agents; return
    ``(weights, info)``. Caches weights and first-order solve metadata to disk.

    seed_lanes: initialize Frank-Wolfe from an all-or-nothing assignment under
    mildly lane-biased free-flow costs.  This is a structured initialization
    for the nonconvex head-on-coupled objective, not a global-optimality
    certificate; ``seed_lanes=False`` is the frozen attribution ablation."""
    key = _toll_key(m, N, alpha, beta, gamma, K, seed, seed_lanes, tag, mu, form)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fz = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(fz):
            z = np.load(fz, allow_pickle=True)
            return z["w"], json.loads(str(z["info"]))
    t0 = time.time()
    net = flowmod.FlowNet(m, alpha=alpha, beta=beta, gamma=gamma, mu=mu, form=form)
    bias = mapmod.crisscross_weights(m, SEED_BIAS_PENALTY) if seed_lanes else None
    sol = flowmod.calibrate_and_solve(net, N, K=K, seed=seed, init_bias=bias)
    w = sol["tolls"]
    f = sol["f"]
    # Both sums below are over directed arcs.  The denominator must therefore
    # include f_rev as well; using f.sum() alone gives a legacy [0,2] quantity.
    asym = float(np.abs(f - f[net.rev]).sum()
                 / max((f + f[net.rev]).sum(), 1e-12))
    info = dict(solve_time=time.time() - t0, lam=sol["lam"], obj=sol["obj"],
                gap=sol["gap"], linear_cost=sol["linear_cost"],
                linear_optimum=sol["linear_optimum"],
                linear_residual=sol["linear_residual"],
                linear_factor_bound=sol["linear_factor_bound"],
                iters=sol["iters"], in_transit=sol["in_transit"],
                mean_latency=sol["mean_latency"], flow_asymmetry=asym,
                alpha=alpha, beta=beta, gamma=gamma, mu=mu, form=form, K=K,
                seed_lanes=seed_lanes)
    if cache_dir:
        np.savez_compressed(fz, w=w, f=f, info=json.dumps(info))
    return w, info


def load_toll_flow(m, N, alpha=1.0, beta=2, gamma=2.0, mu=0.0, K=None, seed=0,
                   seed_lanes=True, cache_dir=None, tag="", form="bpr"):
    key = _toll_key(m, N, alpha, beta, gamma, K, seed, seed_lanes, tag, mu, form)
    z = np.load(os.path.join(cache_dir, key + ".npz"), allow_pickle=True)
    return z["f"], json.loads(str(z["info"]))


# ---- compact ES parameterization ----------------------------------------
def es_dim(tile=4):
    return tile * tile * 4


def es_weights(m, theta, tile=4):
    """theta (tile*tile*4,) log-weights in [-2,2]; edge weight =
    exp(theta[tile(u), dir(u->v)]). Compact tiled parameterization (honest
    laptop-scale stand-in for GGO's full per-edge search space)."""
    edges, _ = mapmod.directed_edges(m)
    W = m["W"]
    th = np.asarray(theta, float).reshape(tile, tile, 4)
    w = np.empty(len(edges))
    for e, (u, v) in enumerate(edges):
        uy, ux = divmod(int(u), W)
        vy, vx = divmod(int(v), W)
        if vy < uy:
            d = 0      # north
        elif vy > uy:
            d = 1      # south
        elif vx > ux:
            d = 2      # east
        else:
            d = 3      # west
        w[e] = np.exp(np.clip(th[uy % tile, ux % tile, d], -2.0, 2.0))
    return w


# ---- expressive (non-periodic) ES parameterization -----------------------
def es_dim_bilinear(K=8):
    return K * K * 4


def es_weights_bilinear(m, theta, K=8):
    """Bilinearly interpolated control-point parameterization.

    theta has shape (K, K, 4): a log-weight per direction at each of K x K
    control points laid over the map, bilinearly interpolated to every cell.
    Unlike the `es_weights` tiling (which indexes `uy % tile`, and is therefore
    spatially PERIODIC and cannot represent boundary-aware or map-specific
    structure), this is position-dependent and can express any smooth field --
    the solution class the toll construction produces. Dimension 4*K^2
    (256 at K=8) vs 64 for the tiled version."""
    edges, _ = mapmod.directed_edges(m)
    H, W = m["H"], m["W"]
    th = np.asarray(theta, float).reshape(K, K, 4)
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    uy, ux = np.divmod(u, W)
    vy, vx = np.divmod(v, W)
    d = np.where(vy < uy, 0, np.where(vy > uy, 1, np.where(vx > ux, 2, 3)))
    # control-point coordinates in [0, K-1]
    gy = uy * (K - 1) / max(H - 1, 1)
    gx = ux * (K - 1) / max(W - 1, 1)
    y0 = np.clip(np.floor(gy).astype(int), 0, K - 2)
    x0 = np.clip(np.floor(gx).astype(int), 0, K - 2)
    ty, tx = gy - y0, gx - x0
    f = th[:, :, :]
    val = ((1 - ty) * (1 - tx) * f[y0, x0, d]
           + (1 - ty) * tx * f[y0, x0 + 1, d]
           + ty * (1 - tx) * f[y0 + 1, x0, d]
           + ty * tx * f[y0 + 1, x0 + 1, d])
    return np.exp(np.clip(val, -2.0, 2.0))
