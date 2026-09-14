"""Run the preregistered NATIVE PLANNER CHECK matrix.

Protocol: `code/native/native_check_protocol.json`, SHA-256 in the sibling `.sha256`, locked
before any matrix cell was generated.  This script refuses to run if the file no longer hashes
to the registered digest.

Each cell is one lifelong run of upstream LaCAM3's compiled PIBT inside `ext_guided`, which draws
its own starts and goals.  Every attempted cell is written to `results/native_check/runs/`.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PROJ = os.path.dirname(CODE)
NC = os.path.join(PROJ, "results", "native_check")
BIN = os.path.join(HERE, "ext_guided")


def protocol(proto):
    raw = open(proto, "rb").read()
    want = open(os.path.splitext(proto)[0] + ".sha256").read().split()[0]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise SystemExit(f"protocol hash mismatch: registered {want}, on disk {got}")
    return json.loads(raw), want


def weight_file(name, method, N):
    """Locate the exported field.  Lane weights do not depend on N, so exactly one file per map
    is written and its N tag is whatever N it was first exported under."""
    if method == "unweighted":
        return None
    if method == "lanes":
        hits = sorted(glob.glob(os.path.join(NC, "weights", f"{name}__lanes__N*.txt")))
        if len(hits) != 1:
            raise SystemExit(f"expected one lane field for {name}, found {len(hits)}")
        return hits[0]
    p = os.path.join(NC, "weights", f"{name}__{method}__N{N}.txt")
    if not os.path.exists(p):
        raise SystemExit(f"missing exported weights {p}; run code/export_native_check.py")
    return p


def run_cell(arg):
    name, N, method, seed, eps, P, sha, out = arg
    key = f"{name}__{method}__N{N}__s{seed}__eps{eps:g}"
    path = os.path.join(out, key + ".json")
    if os.path.exists(path):
        return "skip", key
    M = P["matrix"]
    cmd = [BIN, "-m", os.path.join(NC, "maps", f"{name}.map"), "-N", str(N),
           "--steps", str(M["steps"]), "--warmup", str(M["warmup"]),
           "--bin", str(M["bin"]), "--seed", str(seed), "--eps-tie", repr(eps)]
    wf = weight_file(name, method, N)
    if wf:
        cmd += ["--weights", wf]
    if name.startswith("warehouse"):
        cmd += ["--pools", os.path.join(NC, "pools", f"{name}.txt")]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return "err", f"{key}: rc={r.returncode} {r.stderr[:200]}"
        rec = json.loads(r.stdout)
        rec.update(map_name=name, method=method, n_agents=N, seed=seed, eps_tie_req=eps,
                   protocol_sha256=sha, lacam3_commit="55347c8")
        os.makedirs(out, exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, path)
        return "ok", key
    except Exception as e:                                             # noqa: BLE001
        return "err", f"{key}: {e}"


def fleet_sizes(M, name):
    """Registered fleet sizes: a shared list, or one list per map."""
    if isinstance(M["N"], dict):
        return M["N"][name]
    return M["N"]


def jobs_from(P, sha, out):
    M = P["matrix"]
    jobs = []
    for name in M["maps"]:
        for N in fleet_sizes(M, name):
            for method in M["methods"]:
                for seed in M["seeds"]:
                    jobs.append((name, N, method, seed, M["eps_tie_main"], P, sha, out))
    ab = M["eps_tie_ablation"]
    for name in M["maps"]:
        for N in fleet_sizes(M, name):
            for method in ab["methods"]:
                for seed in M["seeds"]:
                    jobs.append((name, N, method, seed, ab["eps_tie"], P, sha, out))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default=os.path.join(HERE, "native_check_protocol.json"))
    args = ap.parse_args()
    P, sha = protocol(args.proto)
    if not (os.path.exists(BIN) and os.access(BIN, os.X_OK)):
        raise SystemExit(f"native binary missing: {BIN}")
    out = os.path.join(NC, P["matrix"].get("runs_dir", "runs"))
    os.makedirs(out, exist_ok=True)
    jobs = jobs_from(P, sha, out)
    assert len(jobs) == P["matrix"]["n_cells"], \
        f"{len(jobs)} jobs vs registered {P['matrix']['n_cells']}"
    n = dict(ok=0, skip=0, err=0)
    with mp.Pool(int(os.environ.get("NC_PROCS", "4"))) as pool:
        for i, (st, msg) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            n[st] = n.get(st, 0) + 1
            if st == "err":
                print(f"  ERR {msg}", flush=True)
            if i % 40 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {n}", flush=True)
    return 0 if n["err"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
