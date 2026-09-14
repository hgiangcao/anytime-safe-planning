"""Aggregate the benchmark-map experiments into tables, statistics and
results/summary_v2.json.

Every number the paper quotes is produced here from results/raw/*, so the
paper can be regenerated from raw data alone.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import BENCH_GRID, RESULTS


def label_of(cond):
    return cond.get("label", cond["method"])


def load(exp):
    """(map, N, label) -> {seed: throughput}"""
    out = defaultdict(dict)
    meta = defaultdict(dict)
    try:
        recs = common.load_results(exp)
    except FileNotFoundError:
        return out, meta
    for r in recs:
        c = r["cond"]
        key = (c["map"], c["N"], label_of(c))
        out[key][c["seed"]] = r["throughput"]
        if "weight_build" in r:
            meta[key] = r["weight_build"]
        if r.get("n_resolves"):
            meta[key] = dict(meta.get(key, {}),
                             resolve_time=r.get("resolve_time"),
                             n_resolves=r.get("n_resolves"))
    return out, meta


def ms(d):
    v = np.array([d[k] for k in sorted(d)], float)
    return float(v.mean()), float(v.std(ddof=0)), len(v)


def paired(a, b):
    """Paired two-sided t-test on matched seeds; also Wilcoxon (n=10 is small
    enough that a distribution-free check is worth reporting)."""
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    x = np.array([a[k] for k in ks], float)
    y = np.array([b[k] for k in ks], float)
    if np.allclose(x, y):
        return dict(n=len(ks), diff=0.0, t=0.0, df=len(ks) - 1, p=1.0,
                    p_wilcoxon=1.0, dz=0.0)
    t, p = sps.ttest_rel(x, y)
    try:
        _, pw = sps.wilcoxon(x, y)
    except Exception:
        pw = float("nan")
    d = x - y
    return dict(n=len(ks), diff=float(d.mean()), t=float(t), df=len(ks) - 1,
                p=float(p), p_wilcoxon=float(pw),
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)))


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p-values in the input order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def headline(summary):
    data, meta = load("exp1b")
    methods = ["unweighted", "lanes", "tolls", "tolls-online"]
    rows, tests = [], []
    for mp_, (ncells, Ns) in BENCH_GRID.items():
        for N in Ns:
            row = dict(map=mp_, N=N, cells=ncells,
                       density=round(100.0 * N / ncells, 1))
            for meth in methods:
                d = data.get((mp_, N, meth))
                if d:
                    mu_, sd, n = ms(d)
                    row[meth] = dict(mean=round(mu_, 4), std=round(sd, 4), n=n)
            for meth in ("tolls", "tolls-online"):
                for base in ("unweighted", "lanes"):
                    a, b = data.get((mp_, N, meth)), data.get((mp_, N, base))
                    if a and b:
                        st = paired(a, b)
                        if st:
                            tests.append(dict(map=mp_, N=N, cmp=f"{meth}-vs-{base}", **st))
            info = meta.get((mp_, N, "tolls"))
            if info:
                row["solve_s"] = round(info.get("solve_time", float("nan")), 2)
                row["lam_pred"] = round(info.get("lam", float("nan")), 3)
                row["fw_gap"] = info.get("gap")
            rows.append(row)
    if tests:
        adj = holm([t["p"] for t in tests])
        for t, a in zip(tests, adj):
            t["p_holm"] = float(a)
            t["significant"] = bool(a < 0.05)
    summary["headline"] = rows
    summary["headline_tests"] = tests
    return rows, tests


def transfer(summary):
    d, _ = load("exp3b_agentcount")
    out = defaultdict(dict)
    for (mp_, N, lab), v in d.items():
        mu_, sd, n = ms(v)
        out[lab][N] = dict(mean=round(mu_, 4), std=round(sd, 4), n=n)
    summary["transfer_agentcount"] = {k: dict(sorted(v.items())) for k, v in out.items()}

    d2, _ = load("exp3b_demandshift")
    hot = defaultdict(dict)
    try:
        for r in common.load_results("exp3b_demandshift"):
            c = r["cond"]
            hot[label_of(c)].setdefault(c.get("hotspot", 0.0), {})[c["seed"]] = r["throughput"]
    except FileNotFoundError:
        pass
    summary["transfer_demandshift"] = {
        lab: {str(h): dict(zip(("mean", "std", "n"),
                              (round(x, 4) if isinstance(x, float) else x
                               for x in ms(v))))
              for h, v in sorted(hs.items())}
        for lab, hs in hot.items()}
    return out, hot


def planner(summary):
    d, _ = load("exp7")
    out = defaultdict(dict)
    for (mp_, N, lab), v in d.items():
        mu_, sd, n = ms(v)
        out[f"{mp_}@{N}"][lab] = dict(mean=round(mu_, 4), std=round(sd, 4), n=n)
    # paired tests vs unweighted, on the second planner
    tests = []
    for (mp_, N, lab), v in d.items():
        if lab in ("tolls", "tolls-online"):
            b = d.get((mp_, N, "unweighted"))
            if b:
                st = paired(v, b)
                if st:
                    tests.append(dict(map=mp_, N=N, cmp=f"{lab}-vs-unweighted", **st))
    if tests:
        for t, a in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = float(a)
            t["significant"] = bool(a < 0.05)
    summary["planner_transfer"] = dict(out)
    summary["planner_transfer_tests"] = tests
    return out


def ablations(summary):
    d, _ = load("exp6b")
    out = defaultdict(dict)
    for (mp_, N, lab), v in d.items():
        mu_, sd, n = ms(v)
        out[f"{mp_}@{N}"][lab] = dict(mean=round(mu_, 4), std=round(sd, 4), n=n)
    summary["ablations"] = dict(out)
    return out


def fluid(summary):
    """Model-predicted lambda vs realised tolled throughput."""
    data, meta = load("exp1b")
    pts = []
    for (mp_, N, lab), v in data.items():
        if lab != "tolls":
            continue
        info = meta.get((mp_, N, lab))
        if not info or "lam" not in info:
            continue
        mu_, sd, n = ms(v)
        pts.append(dict(map=mp_, N=N, predicted=round(info["lam"], 4),
                        realised=round(mu_, 4), std=round(sd, 4), n=n))
    if len(pts) >= 3:
        p = np.array([x["predicted"] for x in pts])
        r = np.array([x["realised"] for x in pts])
        summary["fluid"] = dict(
            points=sorted(pts, key=lambda x: (x["map"], x["N"])),
            pearson_r=round(float(np.corrcoef(p, r)[0, 1]), 4),
            spearman_r=round(float(sps.spearmanr(p, r).statistic), 4),
            mean_ratio=round(float(np.mean(r / p)), 4),
            n_points=len(pts))
    return pts


def compute_tradeoff(summary):
    """Guidance-construction compute vs throughput at the ES training cell."""
    data, meta = load("exp1b")
    MAP, N = "empty-32-32", 400
    out = {}
    for lab in ("unweighted", "lanes", "tolls", "tolls-online"):
        d = data.get((MAP, N, lab))
        if not d:
            continue
        mu_, sd, n = ms(d)
        info = meta.get((MAP, N, lab), {})
        cs = info.get("solve_time")
        if lab == "tolls-online":
            cs = info.get("resolve_time")
        out[lab] = dict(tput=round(mu_, 4), std=round(sd, 4), n=n,
                        compute_s=round(cs, 2) if cs else 0.01)
    for tag in ("cold", "warm", "fullcold", "fullwarm"):
        try:
            recs = common.load_results(f"exp2c_eval_{tag}")
        except FileNotFoundError:
            continue
        v = {r["cond"]["seed"]: r["throughput"] for r in recs}
        if not v:
            continue
        mu_, sd, n = ms(v)
        z = np.load(os.path.join(common.CACHE, f"es_strong_best_{tag}.npz"),
                    allow_pickle=True)
        dim = int(z["dim"]) if "dim" in z.files else int(z["K"]) ** 2 * 4
        out[f"es-strong-{tag}"] = dict(
            tput=round(mu_, 4), std=round(sd, 4), n=n,
            compute_s=round(float(z["search_time"]), 2),
            n_evals=int(z["n_evals"]), dim=dim,
            param=str(z["param"]) if "param" in z.files else "grid",
            train_fitness=round(-float(z["fit"]), 4))
    # Wall-clock for the searches is contaminated by CPU contention (two
    # searches shared the machine), so also record the least-contended
    # per-evaluation cost as a hardware-comparable rate, and express search
    # budgets in rollout evaluations, which is what GGO reports too.
    rates = [v["compute_s"] / v["n_evals"] for v in out.values()
             if v.get("n_evals")]
    if rates:
        out["_eval_rate_s"] = round(min(rates), 3)
    summary["compute_vs_tput"] = out
    # paired tests tolls vs each ES variant
    tests = []
    tolls = data.get((MAP, N, "tolls"))
    for tag in ("cold", "warm", "fullcold", "fullwarm"):
        try:
            recs = common.load_results(f"exp2c_eval_{tag}")
        except FileNotFoundError:
            continue
        v = {r["cond"]["seed"]: r["throughput"] for r in recs}
        if tolls and v:
            st = paired(tolls, v)
            if st:
                tests.append(dict(cmp=f"tolls-vs-es-strong-{tag}", **st))
    summary["compute_tests"] = tests
    return out


def online_cost(summary):
    """Measured per-re-solve wall time of the online variant, by map size."""
    small, big = [], []
    try:
        recs = common.load_results("exp1b")
    except FileNotFoundError:
        return
    for r in recs:
        c = r["cond"]
        if c["method"] != "tolls-online" or not r.get("n_resolves"):
            continue
        per = r["resolve_time"] / r["n_resolves"]
        (big if c["map"].startswith("warehouse") else small).append(per)
    def rng(v):
        return f"{min(v):.1f}--{max(v):.1f}" if v else "??"
    summary["online_cost"] = dict(
        small_range=rng(small), warehouse_range=rng(big),
        small_mean=round(float(np.mean(small)), 2) if small else None,
        warehouse_mean=round(float(np.mean(big)), 2) if big else None)


def main():
    summary = dict(
        note="Benchmark-map revision: standard MAPF maps (movingai.com), "
             "PIBT with the swap operation, congestion-flow model with head-on "
             "coupling and vertex capacity.",
        config=dict(maps={k: dict(cells=v[0], N=v[1]) for k, v in BENCH_GRID.items()},
                    gamma=common.GAMMA, mu=common.MU, tau=common.TAU,
                    steps=common.STEPS, seeds=len(common.SEEDS)))
    rows, tests = headline(summary)
    transfer(summary)
    planner(summary)
    ablations(summary)
    fluid(summary)
    compute_tradeoff(summary)
    online_cost(summary)
    path = os.path.join(RESULTS, "summary_v2.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {path}")

    print("\n=== HEADLINE (tasks/step, mean +- std over seeds) ===")
    hdr = f"{'map':24s} {'N':>5s} {'dens':>5s} " + "".join(
        f"{m:>17s}" for m in ("unweighted", "lanes", "tolls", "tolls-online"))
    print(hdr)
    for r in rows:
        line = f"{r['map']:24s} {r['N']:5d} {r['density']:4.0f}% "
        for m in ("unweighted", "lanes", "tolls", "tolls-online"):
            c = r.get(m)
            cell = "-" if not c else "%.2f+-%.2f" % (c["mean"], c["std"])
            line += "%17s" % cell
        print(line)
    sig = [t for t in tests if t.get("significant")]
    print(f"\n{len(sig)}/{len(tests)} comparisons significant after Holm correction")
    return summary


if __name__ == "__main__":
    main()
