"""Procedural 4-connected grid maps.

A map is a dict with:
  H, W        : grid dims
  free        : bool array (H*W,) free cells (True) / obstacle (False)
  name        : str
  cells       : array of free cell ids (row-major index y*W+x)
  cell_index  : array (H*W,) mapping cell id -> compact free-cell index (-1 if obstacle)
  meta        : optional extras (e.g. warehouse pickup/station cells, compact indices)
All maps are guaranteed connected (largest component kept, rest turned to obstacles).
"""
import os as _os
import numpy as np


def _largest_component(free, H, W):
    """Keep only the largest 4-connected free component."""
    lab = -np.ones(H * W, dtype=np.int32)
    comp = 0
    best, best_size = -1, 0
    for s in range(H * W):
        if not free[s] or lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = comp
        size = 0
        while stack:
            v = stack.pop()
            size += 1
            y, x = divmod(v, W)
            for u in ((v - W) if y > 0 else -1, (v + W) if y < H - 1 else -1,
                      (v - 1) if x > 0 else -1, (v + 1) if x < W - 1 else -1):
                if u >= 0 and free[u] and lab[u] < 0:
                    lab[u] = comp
                    stack.append(u)
        if size > best_size:
            best, best_size = comp, size
        comp += 1
    return free & (lab == best)


def _finalize(name, free, H, W, meta=None):
    free = _largest_component(free.copy(), H, W)
    cells = np.flatnonzero(free).astype(np.int32)
    cell_index = -np.ones(H * W, dtype=np.int32)
    cell_index[cells] = np.arange(len(cells), dtype=np.int32)
    return dict(name=name, H=H, W=W, free=free, cells=cells,
                cell_index=cell_index, meta=meta or {})


def empty_32():
    H = W = 32
    return _finalize("empty-32x32", np.ones(H * W, bool), H, W)


def random_32(obstacle_frac=0.20, seed=7):
    H = W = 32
    rng = np.random.default_rng(seed)
    free = rng.random(H * W) >= obstacle_frac
    return _finalize("random-32x32", free, H, W)


def room_32(seed=3):
    """Room-like: 8x8 rooms separated by walls with 2-wide doors."""
    H = W = 32
    rng = np.random.default_rng(seed)
    free = np.ones((H, W), bool)
    for wall in (8, 16, 24):
        free[wall, :] = False
        free[:, wall] = False
    # doors: for each wall segment between crossings, open a 2-wide gap
    bounds = [(0, 8), (9, 16), (17, 24), (25, 32)]
    for wall in (8, 16, 24):
        for (a, b) in bounds:
            d = rng.integers(a, b - 1)
            free[wall, d:d + 2] = True
            d = rng.integers(a, b - 1)
            free[d:d + 2, wall] = True
    return _finalize("room-32x32", free.ravel(), H, W)


def warehouse_33x36():
    """Kiva-style warehouse: shelf blocks in the middle, stations on both sides.

    H=33, W=36. Shelf blocks are 8-long, 1-tall segments arranged in rows with
    2-wide aisles; a 3-wide open corridor rings the shelf area; workstations
    (delivery cells) sit on the left and right edges.
    """
    H, W = 33, 36
    free = np.ones((H, W), bool)
    shelf = np.zeros((H, W), bool)
    for r in range(4, 29, 3):          # shelf rows every 3 rows
        for c0 in (4, 14, 24):         # three 8-long blocks per row
            shelf[r, c0:c0 + 8] = True
    free[shelf] = False
    stations = [y * W + 0 for y in range(2, 31, 4)] + \
               [y * W + (W - 1) for y in range(2, 31, 4)]
    m = _finalize("warehouse-33x36", free.ravel(), H, W)
    # pickup cells: free cells 4-adjacent to a shelf
    sh = shelf.ravel()
    pick = []
    for v in m["cells"]:
        y, x = divmod(int(v), W)
        for u in ((v - W) if y > 0 else -1, (v + W) if y < H - 1 else -1,
                  (v - 1) if x > 0 else -1, (v + 1) if x < W - 1 else -1):
            if u >= 0 and sh[u]:
                pick.append(int(v))
                break
    stations = [s for s in stations if m["free"][s]]
    m["meta"] = dict(pickups=np.array(sorted(set(pick)), dtype=np.int32),
                     stations=np.array(stations, dtype=np.int32))
    return m


MAPS = {
    "empty-32x32": empty_32,
    "random-32x32": random_32,
    "room-32x32": room_32,
    "warehouse-33x36": warehouse_33x36,
}


def get_map(name):
    return MAPS[name]()


# ---- standard MAPF benchmark maps (movingai.com / Stern et al. 2019) ------
_BENCH_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))), "data", "maps")

# Free-space characters in the octile .map format; every other character
# (@ obstacle, T tree, S swamp, W water) is impassable for MAPF.
_PASSABLE = ".G"


def load_benchmark(fname, name=None):
    """Load a movingai.com octile .map file from data/maps/."""
    path = _os.path.join(_BENCH_DIR, fname if fname.endswith(".map")
                         else fname + ".map")
    with open(path) as fh:
        lines = fh.read().split("\n")
    H = int(lines[1].split()[1])
    W = int(lines[2].split()[1])
    assert lines[3].strip() == "map"
    rows = lines[4:4 + H]
    free = np.array([c in _PASSABLE for r in rows for c in r[:W]], dtype=bool)
    assert free.size == H * W, f"{fname}: expected {H*W} cells, got {free.size}"
    return _finalize(name or _os.path.splitext(fname)[0], free, H, W)


def _bench_warehouse(fname, name):
    """Benchmark warehouse map + Kiva-style endpoint sets.

    Pickup cells: free cells 4-adjacent to a shelf block (the 'T' regions).
    Station cells: free cells in the two outermost open columns, which is
    where workstations sit in the Kiva/RHCR warehouse layouts."""
    m = load_benchmark(fname, name)
    H, W, free = m["H"], m["W"], m["free"]
    blocked = ~free
    pick = []
    for v in m["cells"]:
        v = int(v)
        y, x = divmod(v, W)
        for u in ((v - W) if y > 0 else -1, (v + W) if y < H - 1 else -1,
                  (v - 1) if x > 0 else -1, (v + 1) if x < W - 1 else -1):
            if u >= 0 and blocked[u]:
                pick.append(v)
                break
    # station columns: leftmost / rightmost columns holding free cells
    cols = sorted({int(v) % W for v in m["cells"]})
    lo, hi = cols[0], cols[-1]
    stations = [int(v) for v in m["cells"] if int(v) % W in (lo, hi)]
    pick = sorted(set(pick) - set(stations))
    m["meta"] = dict(pickups=np.array(pick, dtype=np.int32),
                     stations=np.array(sorted(stations), dtype=np.int32))
    return m


BENCH_MAPS = {
    "empty-32-32":            lambda: load_benchmark("empty-32-32"),
    "random-32-32-20":        lambda: load_benchmark("random-32-32-20"),
    "room-32-32-4":           lambda: load_benchmark("room-32-32-4"),
    "maze-32-32-2":           lambda: load_benchmark("maze-32-32-2"),
    "maze-32-32-4":           lambda: load_benchmark("maze-32-32-4"),
    "warehouse-10-20-10-2-1": lambda: _bench_warehouse(
        "warehouse-10-20-10-2-1", "warehouse-10-20-10-2-1"),
}
MAPS.update(BENCH_MAPS)



def crisscross_weights(m, penalty=5.0):
    """Alternating one-way-lane weights: even rows prefer east, odd rows west;
    even cols south, odd cols north. Against-preference edges get `penalty`,
    preferred get 1. Used both as the handcrafted-lanes baseline (penalty=5)
    and, with a mild penalty, as the symmetry-breaking initial assignment bias
    for Frank-Wolfe (see flow.calibrate_and_solve)."""
    edges, _ = directed_edges(m)
    W = m["W"]
    w = np.ones(len(edges))
    for e, (u, v) in enumerate(edges):
        uy, ux = divmod(int(u), W)
        vy, vx = divmod(int(v), W)
        if vy == uy:
            if ((vx > ux) != (uy % 2 == 0)):
                w[e] = penalty
        else:
            if ((vy > uy) != (ux % 2 == 0)):
                w[e] = penalty
    return w


def neighbors_lists(m):
    """List (len n_free) of python lists of free neighbor cell ids (grid ids)."""
    H, W, free = m["H"], m["W"], m["free"]
    out = []
    for v in m["cells"]:
        v = int(v)
        y, x = divmod(v, W)
        nb = []
        for u in ((v - W) if y > 0 else -1, (v + W) if y < H - 1 else -1,
                  (v - 1) if x > 0 else -1, (v + 1) if x < W - 1 else -1):
            if u >= 0 and free[u]:
                nb.append(u)
        out.append(nb)
    return out


def directed_edges(m):
    """Directed edge list over free cells.

    Returns (E,2) int array of (u,v) grid ids, plus rev index array where
    rev[e] is the index of edge (v,u)."""
    nbr = neighbors_lists(m)
    edges = []
    for i, v in enumerate(m["cells"]):
        for u in nbr[i]:
            edges.append((int(v), u))
    edges = np.array(edges, dtype=np.int32)
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(edges)}
    rev = np.array([key[(int(b), int(a))] for (a, b) in edges], dtype=np.int32)
    return edges, rev
