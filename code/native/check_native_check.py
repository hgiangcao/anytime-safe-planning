"""The three provenance checks the native-check protocol registered, plus the run inventory.

1. export_round_trip   -- the exported arc weights re-read bit-identically to build_weights().
2. vertex_set          -- the native driver's |V| and |largest component| equal this repository's
                          free-cell count for every map it ran on.
3. python_arm_reproduction -- one stored Python condition re-runs to its frozen throughput.

Imports nothing from the native runner; it re-derives everything from the exported files and the
frozen raw records.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PROJ = os.path.dirname(CODE)
sys.path.insert(0, CODE)
import common                                                          # noqa: E402
from mapf import maps as mapmod                                        # noqa: E402

NC = os.path.join(PROJ, "results", "native_check")


def read_weight_file(path):
    with open(path) as fh:
        E = int(fh.readline())
        u = np.empty(E, dtype=np.int64)
        v = np.empty(E, dtype=np.int64)
        w = np.empty(E, dtype=float)
        for i in range(E):
            a, b, c = fh.readline().split()
            u[i], v[i], w[i] = int(a), int(b), float(c)
    return u, v, w


def main():
    fails = []
    man = json.load(open(os.path.join(NC, "export_manifest.json")))

    # 1. round trip
    n_rt = 0
    for key in sorted(man["weights"]):
        name, method, ntag = key.split("__")
        N = int(ntag[1:])
        m = mapmod.get_map(name)
        want, _ = common.build_weights(m, dict(map=name, method=method, N=N))
        u, v, w = read_weight_file(os.path.join(NC, "weights", key + ".txt"))
        edges, _ = mapmod.directed_edges(m)
        if not (np.array_equal(u, edges[:, 0].astype(np.int64))
                and np.array_equal(v, edges[:, 1].astype(np.int64))):
            fails.append(f"{key}: arc order does not match directed_edges")
            continue
        if not np.array_equal(w, np.asarray(want, dtype=float)):
            fails.append(f"{key}: {int((w != np.asarray(want)).sum())} of {len(w)} weights differ")
            continue
        n_rt += 1
    print(f"1. export round trip: {n_rt}/{len(man['weights'])} weight fields bit-identical")

    # 2. vertex set
    recs = []
    for sub in ("runs", "runs_bench"):
        recs += [json.load(open(f)) for f in sorted(glob.glob(os.path.join(NC, sub, "*.json")))]
    seen = {}
    for r in recs:
        seen.setdefault(r["map_name"], set()).add((r["V"], r["V_component"]))
    for name, vs in sorted(seen.items()):
        want = man["maps"][name]["n_free"]
        for V, Vc in vs:
            if V != want or Vc != want:
                fails.append(f"{name}: driver |V|={V} |component|={Vc}, repository free cells {want}")
    print(f"2. vertex set: {len(seen)} maps, "
          f"{'all agree' if not fails else 'DISAGREEMENT'} with the repository free-cell counts")

    # 3. python arm reproduction
    cond = dict(map="room-32-32-4", N=170, method="tolls", seed=0,
                steps=common.STEPS, mu=common.MU, gamma=common.GAMMA)
    key = common.cond_key(cond)
    frozen_path = os.path.join(common.RAW, "exp1b", key + ".json")
    if not os.path.exists(frozen_path):
        fails.append(f"no frozen Python record at {frozen_path}")
        print("3. python arm reproduction: SKIPPED (no frozen record)")
    else:
        frozen = json.load(open(frozen_path))
        now = common.run_condition(cond)
        ok = abs(now["throughput"] - frozen["throughput"]) < 1e-12
        if not ok:
            fails.append(f"python reproduction {cond}: frozen {frozen['throughput']} "
                         f"now {now['throughput']}")
        print(f"3. python arm reproduction ({key}): frozen {frozen['throughput']} "
              f"now {now['throughput']} -> {'identical' if ok else 'DIFFERS'}")

    # inventory
    per = {}
    for r in recs:
        per[(r["map_name"], r["n_agents"], r["method"], r["eps_tie_req"])] = \
            per.get((r["map_name"], r["n_agents"], r["method"], r["eps_tie_req"]), 0) + 1
    bad_n = [k for k, v in per.items() if v != 10]
    print(f"4. inventory: {len(recs)} native runs in {len(per)} groups, "
          f"{'all 10 seeds' if not bad_n else f'{len(bad_n)} groups off 10 seeds'}")
    if bad_n:
        fails.append(f"groups without 10 seeds: {bad_n[:5]}")

    print("\n" + ("ALL PROVENANCE CHECKS PASS" if not fails
                  else f"{len(fails)} FAILURES:\n  " + "\n  ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
