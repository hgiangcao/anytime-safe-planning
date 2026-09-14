"""Aggregate raw results into paper-ready tables, figures, and summary.json.

Figures (PDF + PNG, sized for a 3.4in single column, fonts >= 7pt):
  fig_throughput_bars : headline throughput, maps x densities x methods
  fig_compute_tput    : compute-vs-throughput scatter (log-x seconds)
  fig_transfer        : transfer across N (weights from N=400)
  fig_heatmaps        : congestion heatmaps unweighted vs ES vs tolls (+ model flow)
  fig_fluid           : fluid-model predicted vs realized throughput
  fig_ablations       : gamma / beta / online-period ablations
Tables:
  table_headline.csv (+ .tex fragment)
Run anytime; uses whatever raw results exist.
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from mapf import maps as mapmod
from mapf import guidance as guid

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

COL = {"unweighted": "#888888", "lanes": "#2ca02c", "es": "#d62728",
       "tolls": "#1f77b4", "tolls-online": "#9467bd",
       "tolls-fixed400": "#17becf", "tolls-adapt": "#1f77b4",
       "es-fixed400": "#d62728",
       "es1000": "#d62728", "es3000": "#d62728"}
LBL = {"unweighted": "PIBT (unweighted)", "lanes": "Crisscross lanes",
       "es": "CMA-ES (300 evals)", "tolls": "Tolls (ours, offline)",
       "tolls-online": "Tolls (ours, online)",
       "tolls-adapt": "Tolls re-solved per $N$ (ours)",
       "tolls-fixed400": "Tolls frozen @$N{=}400$",
       "es-fixed400": "CMA-ES frozen @$N{=}400$",
       "es1000": "CMA-ES (1,000 evals)", "es3000": "CMA-ES (3,000 evals)"}
MAPS = ["empty-32x32", "random-32x32", "room-32x32", "warehouse-33x36"]
NS = [100, 250, 400]


def label_of(cond):
    return cond.get("label", cond["method"])


def agg(rows, keyfn):
    d = defaultdict(list)
    for r in rows:
        d[keyfn(r["cond"])].append(r["throughput"])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in d.items()}


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(common.FIGS, f"{name}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("fig:", name)


# ---------------------------------------------------------------- headline
def headline():
    rows = [r for r in common.load_results("exp1")]
    es_rows = common.load_results("exp2_eval")
    if not rows:
        return None
    a = agg(rows, lambda c: (c["map"], c["N"], c["method"]))
    ae = agg(es_rows, lambda c: (c["map"], c["N"], c["method"]))
    a.update(ae)
    methods = ["unweighted", "lanes", "es", "tolls", "tolls-online"]
    # ---- CSV table
    import csv
    path = os.path.join(common.RESULTS, "table_headline.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "N"] + [m for m in methods for _ in (0, 1)])
        for mp in MAPS:
            for N in NS:
                row = [mp, N]
                for meth in methods:
                    v = a.get((mp, N, meth))
                    row += [f"{v[0]:.3f}", f"{v[1]:.3f}"] if v else ["", ""]
                w.writerow(row)
    print("table:", path)
    # ---- bar figure (2x2 maps, grouped by N)
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.6), sharex=True)
    for ax, mp in zip(axes.ravel(), MAPS):
        present = [m for m in methods if any((mp, N, m) in a for N in NS)]
        x = np.arange(len(NS))
        wd = 0.8 / len(present)
        for i, meth in enumerate(present):
            mu = [a.get((mp, N, meth), (np.nan,) * 3)[0] for N in NS]
            sd = [a.get((mp, N, meth), (np.nan,) * 3)[1] for N in NS]
            ax.bar(x + (i - len(present) / 2 + 0.5) * wd, mu, wd, yerr=sd,
                   color=COL[meth], label=LBL[meth], error_kw=dict(lw=0.7))
        ax.set_title(mp, fontsize=8)
        ax.set_xticks(x, [str(n) for n in NS])
    axes[1, 0].set_xlabel("agents $N$")
    axes[1, 1].set_xlabel("agents $N$")
    axes[0, 0].set_ylabel("throughput (tasks/step)")
    axes[1, 0].set_ylabel("throughput (tasks/step)")
    axes[0, 0].legend(ncol=1, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig_throughput_bars")
    return a


# ------------------------------------------------------- compute vs tput
def compute_tput(a):
    """Compute = one-off guidance construction cost; y = throughput at
    empty-32x32 N=400 (ES's training condition)."""
    if a is None:
        return None
    mp, N = "empty-32x32", 400
    pts = {}
    # ours: flow solve time from cache info
    m = mapmod.get_map(mp)
    _, info = guid.wardrop_tolls(m, N, cache_dir=common.CACHE)
    if (mp, N, "tolls") in a:
        pts["tolls"] = (info["solve_time"], a[(mp, N, "tolls")])
    # online: mean total resolve time per run
    on = [r for r in common.load_results("exp1")
          if r["cond"]["map"] == mp and r["cond"]["N"] == N
          and r["cond"]["method"] == "tolls-online"]
    if on:
        rt = float(np.mean([r["resolve_time"] for r in on]))
        pts["tolls-online"] = (rt, a[(mp, N, "tolls-online")])
    # ES: total search time
    if os.path.exists(common.es_best_path()):
        z = np.load(common.es_best_path(), allow_pickle=True)
        if (mp, N, "es") in a:
            pts["es"] = (float(z["search_time"]), a[(mp, N, "es")])
    # extended-budget ES points (exp2b): full-protocol evals + snapshot times
    a2b = agg(common.load_results("exp2b_eval"),
              lambda c: (c["map"], c["N"], c["method"]))
    for lb in ("es1000", "es3000"):
        snap = os.path.join(common.CACHE, f"es_best_{lb}.npz")
        if os.path.exists(snap) and (mp, N, lb) in a2b:
            z = np.load(snap, allow_pickle=True)
            pts[lb] = (float(z["search_time"]), a2b[(mp, N, lb)])
    # references at ~0 compute (plot at small epsilon)
    for meth in ("unweighted", "lanes"):
        if (mp, N, meth) in a:
            pts[meth] = (1e-2, a[(mp, N, meth)])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    es_curve = [pts[k] for k in ("es", "es1000", "es3000") if k in pts]
    if len(es_curve) > 1:  # connect the ES budget points into a trend
        ax.plot([t for t, _ in es_curve], [v[0] for _, v in es_curve],
                "-", color=COL["es"], lw=0.9, alpha=0.6, zorder=1)
    for meth, (t, (mu, sd, n)) in pts.items():
        fmt = {"es1000": "^", "es3000": "s"}.get(meth, "o")
        ax.errorbar(t, mu, yerr=sd, fmt=fmt, ms=5, color=COL[meth],
                    label=LBL[meth], capsize=2, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("guidance construction compute (s, log)")
    ax.set_ylabel("throughput (tasks/step)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.13, 0.0), fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig_compute_tput")
    return {k: dict(compute_s=v[0], tput=v[1][0], std=v[1][1]) for k, v in pts.items()}


# ------------------------------------------------------------- transfer
def transfer():
    rows = common.load_results("exp3")
    if not rows:
        return None
    a = agg(rows, lambda c: (label_of(c), c["N"]))
    labels = ["unweighted", "lanes", "tolls-fixed400", "tolls-adapt",
              "es-fixed400"]
    NS8 = sorted({c for (_, c) in a})
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for lb in labels:
        xs = [N for N in NS8 if (lb, N) in a]
        if not xs:
            continue
        mu = np.array([a[(lb, N)][0] for N in xs])
        sd = np.array([a[(lb, N)][1] for N in xs])
        style = "--s" if lb.endswith("fixed400") else "-o"
        ax.plot(xs, mu, style, ms=2.5, lw=1.1, color=COL[lb], label=LBL[lb])
        ax.fill_between(xs, mu - sd, mu + sd, color=COL[lb], alpha=0.15, lw=0)
    ax.set_xlabel("agents $N$ (weights from $N{=}400$)")
    ax.set_ylabel("throughput (tasks/step)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.13, 0.0), fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig_transfer")
    return {f"{lb}@N{N}": a[(lb, N)] for (lb, N) in a}


# ------------------------------------------------------------- heatmaps
def _edge_field(m, edge_counts):
    """Map per-directed-edge counts to a per-cell traversal intensity."""
    edges, _ = mapmod.directed_edges(m)
    H, W = m["H"], m["W"]
    img = np.zeros(H * W)
    for (u, v), c in zip(edges, edge_counts):
        img[u] += c / 2.0
        img[v] += c / 2.0
    img[~m["free"]] = np.nan
    return img.reshape(H, W)


def heatmaps():
    rows = common.load_results("exp4")
    if not rows:
        return None
    have = {(r["cond"]["map"], r["cond"]["method"]): r for r in rows}
    m = mapmod.get_map("empty-32x32")
    order = [k for k in ["unweighted", "lanes", "es", "tolls"]
             if ("empty-32x32", k) in have]
    ncol = len(order) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(1.65 * ncol, 1.95))
    vmax = None
    for ax, meth in zip(axes, order):
        r = have[("empty-32x32", meth)]
        z = np.load(os.path.join(common.RESULTS, r["edge_counts_file"]))
        img = _edge_field(m, z["edge_counts"])
        if vmax is None:
            vmax = np.nanpercentile(img, 99)
        im = ax.imshow(img, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(LBL[meth].replace(" (ours, offline)", " (ours)"),
                     fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
    # model flow (predicted congestion) in last panel
    f, _ = guid.load_toll_flow(m, 400, cache_dir=common.CACHE)
    edges, _ = mapmod.directed_edges(m)
    scale = 1000.0  # counts are over 1000 steps; f is per-step flow
    imgf = _edge_field(m, f * scale)
    axes[-1].imshow(imgf, cmap="magma", vmin=0, vmax=vmax)
    axes[-1].set_title("model flow $f^*$ (ours)", fontsize=6.5)
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="cell traversals / 1000 steps")
    savefig(fig, "fig_heatmaps")

    # warehouse pair
    mw = mapmod.get_map("warehouse-33x36")
    order = [k for k in ["unweighted", "tolls"] if ("warehouse-33x36", k) in have]
    if order:
        fig, axes = plt.subplots(1, len(order), figsize=(1.9 * len(order), 2.0))
        axes = np.atleast_1d(axes)
        for ax, meth in zip(axes, order):
            r = have[("warehouse-33x36", meth)]
            z = np.load(os.path.join(common.RESULTS, r["edge_counts_file"]))
            img = _edge_field(mw, z["edge_counts"])
            im = ax.imshow(img, cmap="magma", vmin=0,
                           vmax=np.nanpercentile(img, 99))
            ax.set_title(LBL[meth], fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
        savefig(fig, "fig_heatmaps_warehouse")
    return True


# ------------------------------------------------------------- fluid model
def fluid():
    path = os.path.join(common.RESULTS, "fluid_model.csv")
    if not os.path.exists(path):
        return None
    import csv
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["realized_mean"]]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    mark = {"empty-32x32": "o", "random-32x32": "s", "room-32x32": "^",
            "warehouse-33x36": "D"}
    for mp in MAPS:
        rs = [r for r in rows if r["map"] == mp]
        if not rs:
            continue
        x = [float(r["predicted_lam"]) for r in rs]
        y = [float(r["realized_mean"]) for r in rs]
        e = [float(r["realized_std"]) for r in rs]
        ax.errorbar(x, y, yerr=e, fmt=mark[mp], ms=3.5, lw=0, elinewidth=0.7,
                    capsize=1.5, label=mp)
    lim = max(float(r["predicted_lam"]) for r in rows) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="$y=x$")
    ax.set_xlabel("fluid-model predicted throughput $\\lambda$")
    ax.set_ylabel("realized throughput (tolls)")
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig_fluid")
    x = np.array([float(r["predicted_lam"]) for r in rows])
    y = np.array([float(r["realized_mean"]) for r in rows])
    return dict(pearson_r=float(np.corrcoef(x, y)[0, 1]),
                mean_ratio=float(np.mean(y / x)), n_points=len(rows))


# ------------------------------------------------------------- ablations
def ablations():
    rows = common.load_results("exp6")
    if not rows:
        return None
    a = agg(rows, lambda c: (c["map"], label_of(c)))
    groups = [("$\\gamma$ (head-on coupling)",
               ["gamma=0.0", "gamma=1.0", "gamma=2.0", "gamma=4.0"],
               ["0", "1", "2*", "4"]),
              ("$\\beta$ (BPR exponent)", ["beta=1", "beta=2", "beta=4"],
               ["1", "2*", "4"]),
              ("online period $T$", ["period=100", "period=200", "period=400"],
               ["100", "200*", "400"]),
              ("FW initialization", ["sym-init", "gamma=2.0"],
               ["symmetric", "lane-seeded*"])]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.8))
    for k, (ax, (title, labels, ticks)) in enumerate(zip(axes, groups)):
        ax2 = ax.twinx()  # separate scale for the saturated random map
        for mi, (mp, axm) in enumerate([("empty-32x32", ax),
                                        ("random-32x32", ax2)]):
            mu = [a.get((mp, lb), (np.nan,) * 3)[0] for lb in labels]
            sd = [a.get((mp, lb), (np.nan,) * 3)[1] for lb in labels]
            x = np.arange(len(labels)) + (mi - 0.5) * 0.3
            axm.bar(x, mu, 0.3, yerr=sd, error_kw=dict(lw=0.7),
                    color=["#1f77b4", "#ff7f0e"][mi], label=mp)
        ax.set_xticks(np.arange(len(labels)), ticks)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylim(0, 17)
        ax2.set_ylim(0, 2.0)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="y", colors="#ff7f0e", labelsize=6)
        ax2.spines["right"].set_color("#ff7f0e")
        if k < 3:
            ax2.set_yticklabels([])
        else:
            ax2.set_ylabel("throughput (random)", color="#ff7f0e", fontsize=7)
    axes[0].set_ylabel("throughput (empty)")
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1 + [plt.Rectangle((0, 0), 1, 1, color="#ff7f0e")],
               l1 + ["random-32x32 (right axis)"], fontsize=6.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.tight_layout()
    savefig(fig, "fig_ablations")
    return {f"{mp}|{lb}": v for (mp, lb), v in a.items()}


# ------------------------------------------------------------- summary
def main():
    summary = dict(headline={}, experiments=[], caveats=[])
    a = headline()
    if a:
        key = lambda mp, N, m: a.get((mp, N, m))
        gains = {}
        for mp in MAPS:
            for N in NS:
                base, ours = key(mp, N, "unweighted"), key(mp, N, "tolls")
                if base and ours and base[0] > 0:
                    gains[f"{mp}@N{N}"] = round(ours[0] / base[0], 4)
        summary["headline"]["tolls_over_unweighted_ratio"] = gains
        summary["experiments"].append(dict(
            name="exp1_headline", config="4 maps x N in {100,250,400} x methods",
            n_seeds=10, steps=1000,
            key_numbers={f"{mp}@N{N}|{m}": list(np.round(v, 4))
                         for (mp, N, m), v in a.items()},
            files=["table_headline.csv", "figures/fig_throughput_bars.pdf"]))
    ct = compute_tput(a)
    if ct:
        summary["headline"]["compute_vs_tput@empty400"] = ct
        summary["experiments"].append(dict(
            name="exp2_compute_vs_throughput", config="empty-32x32 N=400",
            key_numbers=ct, files=["figures/fig_compute_tput.pdf"]))
    tr = transfer()
    if tr:
        def g(lb, N):
            v = tr.get(f"{lb}@N{N}")
            return v[0] if v else None
        if g("tolls-fixed400", 600) and g("unweighted", 600):
            summary["headline"]["transfer@N600"] = dict(
                tolls_fixed400=g("tolls-fixed400", 600),
                tolls_adapt=g("tolls-adapt", 600),
                es_fixed400=g("es-fixed400", 600),
                lanes=g("lanes", 600), unweighted=g("unweighted", 600))
        summary["experiments"].append(dict(
            name="exp3_transfer", config="empty-32x32, N in 100..800, weights@400",
            n_seeds=10, steps=1000,
            key_numbers={k: list(np.round(v, 4)) for k, v in tr.items()},
            files=["figures/fig_transfer.pdf"]))
    if heatmaps():
        summary["experiments"].append(dict(
            name="exp4_heatmaps", config="1000-step edge counts, seed 0",
            key_numbers={}, files=["figures/fig_heatmaps.pdf",
                                   "figures/fig_heatmaps_warehouse.pdf"]))
    fl = fluid()
    if fl:
        summary["headline"]["fluid_model_fit"] = fl
        summary["experiments"].append(dict(
            name="exp5_fluid", config="predicted lam vs realized (tolls)",
            key_numbers=fl, files=["fluid_model.csv", "figures/fig_fluid.pdf"]))
    ab = ablations()
    if ab:
        hl = {}
        for k in ("gamma=0.0", "gamma=2.0", "sym-init"):
            v = ab.get(f"empty-32x32|{k}")
            if v:
                hl[k] = v[0]
        summary["headline"]["ablation_empty400"] = hl
        summary["experiments"].append(dict(
            name="exp6_ablations", config="empty+random @N=400, 5 seeds",
            key_numbers={k: list(np.round(v, 4)) for k, v in ab.items()},
            files=["figures/fig_ablations.pdf"]))
    if os.path.exists(common.es_best_path()):
        z = np.load(common.es_best_path(), allow_pickle=True)
        summary["headline"]["es_search"] = dict(
            n_evals=int(z["n_evals"]), search_time_s=float(z["search_time"]),
            train_fitness=-float(z["fit"]))
    summary["caveats"] = CAVEATS
    with open(os.path.join(common.RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote results/summary.json")


CAVEATS = [
    "All components (PIBT, maps, CMA-ES, flow solver) are compact self-reimplementations, not the public GGO codebase; absolute throughputs are not directly comparable to published GGO numbers.",
    "CMA-ES baseline uses a laptop-scale budget (300 rollout evaluations, 64-dim tiled weight parameterization) on ONE map/density (empty-32x32, N=400), far below GGO's cluster budgets; its numbers are a budget-matched reference, not a reproduction.",
    "Maps are procedurally generated to match the spec's classes (empty/random/room/warehouse); they are not byte-identical to MAPF benchmark files.",
    "The bidirectional cross term gamma*f_e*f_rev makes the objective nonconvex; Frank-Wolfe decreases it monotonically to a stationary point and final FW gaps are reported (gamma=0 convex case verified against CVXPY). The direction-symmetric flow is a stationary point, so FW is initialized from a mildly lane-biased all-or-nothing assignment (crisscross penalty 1.5) to select a symmetry-broken, one-way-structured stationary point; at equal demand this point has LOWER social cost than the symmetric one (unit-tested), and the symmetric-init variant is reported as an ablation.",
    "Fluid-model calibration matches agents-in-transit to N via Little's law; predicted throughput ignores discreteness, vertex-exclusion, and PIBT suboptimality.",
    "RHCR planner and maze/sortation maps from the original spec were not implemented (PIBT-only, 4 maps) to fit the compute budget.",
    "random-32x32 (20% obstacles) saturates at ~1.2 tasks/step for ALL methods at N>=100 (near-ideal throughput is reached by N~25; 1-wide passages cap capacity and cause PIBT corridor contention with large seed-to-seed variance ~0.2-0.4); method differences on this map at the tested N are within noise, and the tolls/unweighted ratios 0.89-1.03 there are not significant.",
    "The ES baseline was trained only on empty-32x32 at N=400 (its budget-limited training condition); ES numbers on other maps/N are transfer evaluations of those frozen weights (exp3), not per-condition searches.",
    "In table_headline.csv, methods without data for a cell have empty fields (e.g. es outside empty-32x32@400); mean and std columns alternate.",
]


if __name__ == "__main__":
    main()
