"""Export this project's maps, guidance fields and endpoint pools for the NATIVE planner check.

The native check runs the SAME guidance fields through a compiled, third-party PIBT
(Kei18/lacam3 `PIBT::set_new_config`, commit 55347c8, vendored under
`../mapf-throughput-envelope/_probes/ext/lacam3`) inside a C++ lifelong driver that generates its
own tasks.  Nothing of this repository's planner, simulator or RNG takes part in that arm; the
only things crossing the boundary are the map geometry and the guidance weights, which is exactly
what is under test.

Written to `results/native_check/`:
  maps/<name>.map        octile map, '.' free '@' blocked, in this repository's own grid indexing
  weights/<name>__<method>__N<N>.txt
                         "E" then E lines "u v w": directed arc u->v by GRID index (y*W+x) and its
                         guidance cost.  Unweighted is not written (the driver defaults to 1).
  pools/<name>.txt       "np ns" then the pickup and station grid indices (warehouse only)
  export_manifest.json   sizes, checksums and the free-cell count each file must reproduce
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                          # noqa: E402
from mapf import maps as mapmod                                        # noqa: E402

OUT = os.path.join(common.RESULTS, "native_check")
METHODS = ["unweighted", "lanes", "tolls"]
# Two map families.  `v1` is the procedural family of the superseded exp1 corpus, kept because its
# native check was preregistered and scored before the mix-up was noticed.  `bench` is the
# BENCH_GRID family the manuscript actually headlines (exp1b).
FAMILIES = {
    "v1": {"empty-32x32": [100, 250, 400], "random-32x32": [100, 250, 400],
           "room-32x32": [100, 250, 400], "warehouse-33x36": [100, 250, 400]},
    "bench": {k: v[1] for k, v in common.BENCH_GRID.items()},
}


def write_map(m, path):
    H, W, free = m["H"], m["W"], m["free"]
    rows = ["".join("." if free[y * W + x] else "@" for x in range(W)) for y in range(H)]
    with open(path, "w") as fh:
        fh.write(f"type octile\nheight {H}\nwidth {W}\nmap\n")
        fh.write("\n".join(rows) + "\n")


def write_weights(m, w, path):
    edges, _ = mapmod.directed_edges(m)
    assert len(w) == len(edges), f"{len(w)} weights vs {len(edges)} arcs"
    with open(path, "w") as fh:
        fh.write(f"{len(edges)}\n")
        for (u, v), wt in zip(edges, np.asarray(w, dtype=float)):
            fh.write(f"{int(u)} {int(v)} {wt!r}\n")


def write_pools(m, path):
    pick = [int(v) for v in m["meta"]["pickups"]]
    stat = [int(v) for v in m["meta"]["stations"]]
    with open(path, "w") as fh:
        fh.write(f"{len(pick)} {len(stat)}\n")
        fh.write(" ".join(str(v) for v in pick) + "\n")
        fh.write(" ".join(str(v) for v in stat) + "\n")
    return len(pick), len(stat)


def main():
    fams = sys.argv[1:] or list(FAMILIES)
    for sub in ("maps", "weights", "pools"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    mpath = os.path.join(OUT, "export_manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for k in ("maps", "weights", "pools"):
        manifest.setdefault(k, {})
    todo = {}
    for f in fams:
        for name, ns in FAMILIES[f].items():
            todo.setdefault(name, []).extend(ns)
    for name, NS in todo.items():
        m = mapmod.get_map(name)
        mp_ = os.path.join(OUT, "maps", f"{name}.map")
        write_map(m, mp_)
        manifest["maps"][name] = dict(
            H=int(m["H"]), W=int(m["W"]), n_free=int(len(m["cells"])),
            n_arcs=int(len(mapmod.directed_edges(m)[0])),
            sha256=hashlib.sha256(open(mp_, "rb").read()).hexdigest())
        if "meta" in m and m.get("meta") is not None and "pickups" in (m.get("meta") or {}):
            pp = os.path.join(OUT, "pools", f"{name}.txt")
            npick, nstat = write_pools(m, pp)
            manifest["pools"][name] = dict(
                n_pickups=npick, n_stations=nstat,
                sha256=hashlib.sha256(open(pp, "rb").read()).hexdigest())
        for method in METHODS:
            if method == "unweighted":
                continue
            for N in NS:
                w, _ = common.build_weights(m, dict(map=name, method=method, N=N))
                if w is None:
                    continue
                key = f"{name}__{method}__N{N}"
                wp = os.path.join(OUT, "weights", key + ".txt")
                write_weights(m, w, wp)
                manifest["weights"][key] = dict(
                    n_arcs=int(len(w)), w_min=float(np.min(w)), w_max=float(np.max(w)),
                    w_mean=float(np.mean(w)),
                    sha256=hashlib.sha256(open(wp, "rb").read()).hexdigest())
                if method == "lanes":
                    break                       # lanes do not depend on N
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print(f"maps {len(manifest['maps'])}  weight fields {len(manifest['weights'])}  "
          f"pools {len(manifest['pools'])} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
