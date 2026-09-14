"""Score the preregistered native planner check into results/native_check/summary.json.

The five criteria are read from `native_check_protocol.json`, whose SHA-256 was locked before any
matrix cell existed; the script re-verifies the hash and refuses to score if it moved.  The Python
comparator is the frozen `results/table_headline.csv`.  Every attempted native cell is used; none
is dropped and no criterion is retuned.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PROJ = os.path.dirname(CODE)
NC = os.path.join(PROJ, "results", "native_check")
METHODS = ["unweighted", "lanes", "tolls"]


def python_arm_v2():
    """{(map, N): {method: mean}} from the frozen exp1b headline in results/summary_v2.json."""
    s = json.load(open(os.path.join(PROJ, "results", "summary_v2.json")))
    out = {}
    for row in s["headline"]:
        out[(row["map"], int(row["N"]))] = {m: row[m]["mean"] for m in METHODS if m in row}
    return out


def python_arm():
    """{(map, N): {method: mean_throughput}} from the frozen headline table."""
    path = os.path.join(PROJ, "results", "table_headline.csv")
    with open(path) as fh:
        rd = list(csv.reader(fh))
    head = rd[0]
    cols = {}
    for j, name in enumerate(head):
        if name in ("map", "N"):
            continue
        cols.setdefault(name, []).append(j)
    out = {}
    for row in rd[1:]:
        key = (row[0], int(row[1]))
        out[key] = {}
        for name, js in cols.items():
            v = row[js[0]]
            if v != "":
                out[key][name] = float(v)
    return out


def fleet_sizes(M, name):
    return M["N"][name] if isinstance(M["N"], dict) else M["N"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default=os.path.join(HERE, "native_check_protocol.json"))
    args = ap.parse_args()
    raw = open(args.proto, "rb").read()
    want = open(os.path.splitext(args.proto)[0] + ".sha256").read().split()[0]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise SystemExit(f"protocol hash mismatch: registered {want}, on disk {got}")
    P = json.loads(raw)
    M, R = P["matrix"], P["comparison_rule_declared_in_advance"]
    runs_dir = M.get("runs_dir", "runs")
    out_path = os.path.join(NC, M.get("summary", "summary.json"))

    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(NC, runs_dir, "*.json")))]
    if len(recs) != M["n_cells"]:
        raise SystemExit(f"{len(recs)} native cells on disk, {M['n_cells']} registered")
    py = python_arm_v2() if runs_dir != "runs" else python_arm()

    def nat(mp, N, method, eps):
        v = [r["throughput"] for r in recs
             if r["map_name"] == mp and r["n_agents"] == N and r["method"] == method
             and abs(r["eps_tie_req"] - eps) < 1e-12]
        return np.array(v, dtype=float)

    eps0 = M["eps_tie_main"]
    cells, level_rel = [], []
    for mp in M["maps"]:
        for N in fleet_sizes(M, mp):
            row = dict(map=mp, N=N)
            for method in METHODS:
                a = nat(mp, N, method, eps0)
                row[f"native_{method}"] = float(a.mean())
                row[f"native_{method}_sd"] = float(a.std(ddof=1))
                row[f"native_{method}_n"] = int(a.size)
                row[f"python_{method}"] = py[(mp, N)].get(method)
                if row[f"python_{method}"]:
                    level_rel.append(abs(a.mean() - row[f"python_{method}"])
                                     / row[f"python_{method}"])
            row["d_TU_native"] = row["native_tolls"] - row["native_unweighted"]
            row["d_TL_native"] = row["native_tolls"] - row["native_lanes"]
            row["d_TU_python"] = row["python_tolls"] - row["python_unweighted"]
            row["d_TL_python"] = row["python_tolls"] - row["python_lanes"]
            row["sign_agree_TU"] = bool(np.sign(row["d_TU_native"]) == np.sign(row["d_TU_python"]))
            row["sign_agree_TL"] = bool(np.sign(row["d_TL_native"]) == np.sign(row["d_TL_python"]))
            fine = nat(mp, N, "tolls", M["eps_tie_ablation"]["eps_tie"])
            row["native_tolls_eps_fine"] = float(fine.mean())
            row["native_tolls_eps_fine_sd"] = float(fine.std(ddof=1))
            row["tie_bins_help_native"] = bool(fine.mean() < row["native_tolls"])
            row["tie_bin_gain_native"] = row["native_tolls"] - float(fine.mean())
            row["mean_task_delay_native_tolls"] = float(np.mean(
                [r["mean_task_delay"] for r in recs if r["map_name"] == mp
                 and r["n_agents"] == N and r["method"] == "tolls"
                 and abs(r["eps_tie_req"] - eps0) < 1e-12]))
            cells.append(row)

    n_cells = len(cells)
    thresh = 9 if n_cells == 12 else 11
    n_tu = sum(c["sign_agree_TU"] for c in cells)
    n_tl = sum(c["sign_agree_TL"] for c in cells)
    n_tie = sum(c["tie_bins_help_native"] for c in cells)
    med_rel = float(np.median(level_rel))
    room_map = "room-32-32-4" if runs_dir != "runs" else "room-32x32"
    room = {c["N"]: c for c in cells if c["map"] == room_map}
    room_Ns = sorted(room)[1:]                    # the two denser fleets, as registered
    c4 = all(room[N]["d_TU_native"] > 0 and room[N]["d_TL_native"] > 0 for N in room_Ns)

    d = dict(
        protocol_sha256=want, lacam3_commit="55347c8", native_cells=len(recs),
        quantizer="round(dist_w / eps_tie), matching mapf.pibt.PIBT._pibt; the superseded floor()"
                  " build's runs are retained under results/native_check/runs*_floor/",
        maps=M["maps"], N=M["N"], methods=METHODS, seeds=len(M["seeds"]),
        steps=M["steps"], eps_tie_main=eps0, eps_tie_ablation=M["eps_tie_ablation"]["eps_tie"],
        cells=cells,
        criteria=dict(
            C1_sign_tolls_vs_unweighted=dict(rule=R["C1_sign_tolls_vs_unweighted"],
                                             value=f"{n_tu}/{n_cells}",
                                             passed=bool(n_tu >= thresh)),
            C2_sign_tolls_vs_lanes=dict(rule=R["C2_sign_tolls_vs_lanes"],
                                        value=f"{n_tl}/{n_cells}",
                                        passed=bool(n_tl >= thresh)),
            C3_level=dict(rule=R["C3_level"], value=med_rel, passed=bool(med_rel <= 0.10)),
            C4_headline_open_map=dict(rule=R["C4_headline_open_map"], passed=bool(c4),
                                      room_map=room_map,
                                      room_cells={str(N): dict(
                                          d_TU=room[N]["d_TU_native"],
                                          d_TL=room[N]["d_TL_native"]) for N in room_Ns}),
            C5_tie_rule=dict(rule=R["C5_tie_rule"], value=f"{n_tie}/{n_cells}",
                             passed=bool(n_tie >= thresh))),
    )
    d["all_criteria_passed"] = all(c["passed"] for c in d["criteria"].values())
    d["n_criteria_passed"] = sum(c["passed"] for c in d["criteria"].values())
    with open(out_path, "w") as fh:
        json.dump(d, fh, indent=1)
    print(json.dumps({k: v for k, v in d.items() if k != "cells"}, indent=1))
    print("\nper cell (native | python):")
    for c in cells:
        print(f"  {c['map']:16s} N={c['N']:<4d} "
              f"u {c['native_unweighted']:6.3f}|{c['python_unweighted']:6.3f}  "
              f"l {c['native_lanes']:6.3f}|{c['python_lanes']:6.3f}  "
              f"t {c['native_tolls']:6.3f}|{c['python_tolls']:6.3f}  "
              f"dTU {c['d_TU_native']:+6.3f}|{c['d_TU_python']:+6.3f} "
              f"{'ok' if c['sign_agree_TU'] else 'XX'}  "
              f"dTL {c['d_TL_native']:+6.3f}|{c['d_TL_python']:+6.3f} "
              f"{'ok' if c['sign_agree_TL'] else 'XX'}  "
              f"eps.01 {c['native_tolls_eps_fine']:6.3f} "
              f"{'ok' if c['tie_bins_help_native'] else 'XX'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
