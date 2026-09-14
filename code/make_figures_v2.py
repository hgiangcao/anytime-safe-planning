"""Figures for the benchmark-map revision. Reads results/summary_v2.json and
results/raw/*, writes paper/figures/*.pdf (and .png next to them)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (Rectangle, Circle, Polygon, Wedge,
                                FancyBboxPatch)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import RESULTS, CACHE, MU, GAMMA
from mapf import maps as mapmod
from mapf import guidance as guid

FIGDIR = os.path.join(common.PROJ, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     # Submission PDFs must use embedded outline fonts; the
                     # Matplotlib default would emit Type 3 resources.
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.dpi": 150})

# Okabe-Ito colour-blind-safe palette (Okabe & Ito 2008); every categorical
# distinction below is carried by a marker or a line style as well as a colour.
C = {"unweighted": "#000000", "lanes": "#009E73", "tolls": "#0072B2",
     "tolls-online": "#D55E00", "es-strong-cold": "#CC79A7",
     "es-strong-warm": "#E69F00", "es-strong-fullcold": "#56B4E9",
     "es-strong-fullwarm": "#F0E442"}
M = {"unweighted": "o", "lanes": "s", "tolls": "D", "tolls-online": "*",
     "es-strong-cold": "^", "es-strong-warm": "^", "es-strong-fullcold": "v",
     "es-strong-fullwarm": "v"}
# A cold search and the same search seeded from the toll field share a marker
# and differ by fill, so the legend needs one entry per method rather than two
# and every label can stay above 7 pt in a single column.
COLD = {"es-strong-cold", "es-strong-fullcold"}
# Legend labels sized for a single-column figure whose text must stay >= 7 pt.
TERSE = {"unweighted": "unweighted", "lanes": "lanes", "tolls": "tolls (ours)",
         "tolls-online": "tolls, adapt.", "es-strong-cold": "CMA-ES 256-D",
         "es-strong-warm": "CMA-ES 256-D, seeded",
         "es-strong-fullcold": "CMA-ES per-edge",
         "es-strong-fullwarm": "CMA-ES per-edge, seeded"}
NAME = {"unweighted": "PIBT (unweighted)", "lanes": "Crisscross lanes",
        "tolls": "Tolls, static (ours)", "tolls-online": "Tolls, sync. adapt.",
        "es-strong-cold": "CMA-ES (256-dim)",
        "es-strong-warm": "CMA-ES (256-dim, from tolls)",
        "es-strong-fullcold": "CMA-ES (per-edge)",
        "es-strong-fullwarm": "CMA-ES (per-edge, from tolls)"}
SHORT = {"empty-32-32": "empty", "random-32-32-20": "random-20",
         "room-32-32-4": "room-4", "maze-32-32-4": "maze-4",
         "warehouse-10-20-10-2-1": "warehouse"}


def S():
    with open(os.path.join(RESULTS, "summary_v2.json")) as f:
        return json.load(f)


# Column and text widths of ieeeconf at letterpaper/10pt, in points.
COL_PT, TEXT_PT = 245.72, 505.89
FS, FS_TITLE = 7.2, 7.6          # >= 7 pt at final rendered size


def _in(pt):
    return pt / 72.0


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def save_at(fig, name):
    r"""Write the figure at exactly the width LaTeX will render it at.

    bbox_inches="tight" crops to the drawn extent, which is generally *wider*
    than the requested figsize; \includegraphics then scales the file down and
    every glyph lands smaller than authored -- the reason four figures here
    used to render between 3.3 and 6.6 pt.  Building the canvas at the final
    width and saving it untrimmed keeps the scale at 1.000, so an authored
    7.2 pt tick label is a 7.2 pt tick label on the page.  Callers must lay the
    figure out with explicit subplots_adjust so nothing needs trimming.
    """
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print("  wrote", name)



# ------------------------------------------------------------- framework ----
def _icon_grid(ax, cx, cy, s=1.0):
    """Pictogram of a gridmap with obstacles, an agent and a goal."""
    n = 4
    cell = 0.028 * s
    x0, y0 = cx - n * cell / 2, cy - n * cell / 2
    for i in range(n):
        for j in range(n):
            blocked = (i, j) in {(1, 2), (2, 0)}
            ax.add_patch(Rectangle((x0 + i * cell, y0 + j * cell), cell, cell,
                                   facecolor="#3a3a3a" if blocked else "white",
                                   edgecolor="#9a9a9a", lw=0.35, zorder=3))
    ax.add_patch(Circle((x0 + 0.5 * cell, y0 + 0.5 * cell), cell * 0.3,
                        facecolor=C["tolls"], edgecolor="none", zorder=4))
    ax.plot([x0 + 3.5 * cell], [y0 + 3.5 * cell], marker="*", ms=3.6,
            color="#d4a017", zorder=4)


def _icon_gauge(ax, cx, cy, s=1.0):
    """Pictogram of the Little's-law calibration dial."""
    r = 0.050 * s
    ax.add_patch(Wedge((cx, cy - r * 0.30), r, 15, 165, width=r * 0.52,
                       facecolor="#dfe7f2", edgecolor="#7f8c9b", lw=0.5,
                       zorder=3))
    ax.annotate("", xy=(cx + r * 0.55, cy + r * 0.42),
                xytext=(cx, cy - r * 0.25),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#d15b47",
                                mutation_scale=5), zorder=4)


def _icon_flow(ax, cx, cy, s=1.0):
    """Pictogram of a symmetry-broken (one-way) flow field."""
    for k, dy in enumerate((-0.042, 0.0, 0.042)):
        d = 1 if k % 2 == 0 else -1
        ax.annotate("", xy=(cx + d * 0.055 * s, cy + dy * s),
                    xytext=(cx - d * 0.055 * s, cy + dy * s),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1,
                                    color=C["tolls"], mutation_scale=6),
                    zorder=4)


def _icon_price(ax, cx, cy, s=1.0):
    """Pictogram of a price tag: the toll read off the optimality conditions."""
    w, h = 0.095 * s, 0.070 * s
    ax.add_patch(Polygon([[cx - w / 2, cy - h / 2], [cx + w / 4, cy - h / 2],
                          [cx + w / 2, cy], [cx + w / 4, cy + h / 2],
                          [cx - w / 2, cy + h / 2]],
                         closed=True, facecolor="#fdf1d6",
                         edgecolor="#d4a017", lw=0.7, zorder=3))
    ax.add_patch(Circle((cx + w / 4.6, cy), 0.011 * s, facecolor="#d4a017",
                        edgecolor="none", zorder=4))
    ax.text(cx - w / 8, cy, r"$w_e$", ha="center", va="center", fontsize=4.8,
            zorder=5)


def _icon_robot(ax, cx, cy, s=1.0):
    """Pictogram of a planner: an agent descending a weighted distance field."""
    ax.add_patch(FancyBboxPatch((cx - 0.030 * s, cy - 0.048 * s),
                                0.060 * s, 0.044 * s,
                                boxstyle="round,pad=0.004,rounding_size=0.006",
                                facecolor=C["tolls"], edgecolor="none",
                                zorder=4))
    ax.add_patch(FancyBboxPatch((cx - 0.024 * s, cy + 0.004 * s),
                                0.048 * s, 0.034 * s,
                                boxstyle="round,pad=0.003,rounding_size=0.006",
                                facecolor=C["tolls"], edgecolor="none",
                                zorder=4))
    for dx in (-0.010, 0.010):
        ax.add_patch(Circle((cx + dx * s, cy + 0.021 * s), 0.005 * s,
                            facecolor="white", edgecolor="none", zorder=5))
    ax.plot([cx, cx], [cy + 0.038 * s, cy + 0.056 * s], lw=0.7,
            color="#555", zorder=4)
    ax.add_patch(Circle((cx, cy + 0.060 * s), 0.005 * s, facecolor="#d4a017",
                        edgecolor="none", zorder=5))


def _icon_bars(ax, cx, cy, s=1.0):
    """Pictogram of the measured outcome: throughput."""
    for k, h in enumerate((0.035, 0.058, 0.082)):
        ax.add_patch(Rectangle((cx - 0.055 * s + k * 0.038 * s,
                                cy - 0.045 * s), 0.026 * s, h * s,
                               facecolor=[C["unweighted"], C["lanes"],
                                          C["tolls"]][k],
                               edgecolor="none", zorder=4))


def fig_framework(_s=None):
    """Study-design schematic: computed and simulated stages."""
    fig, ax = plt.subplots(figsize=(7.1, 1.52))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        ("Benchmark map\n& task model", "Sec. IV-A", _icon_grid),
        ("Demand\ncalibration", "Alg. 1", _icon_gauge),
        ("Stationary-flow\ncandidate", "Eq. (3)", _icon_flow),
        ("Marginal-cost\ntolls " + r"$w{=}\nabla J$", "Eq. (4)", _icon_price),
        ("PIBT planner\n" + r"$D_w$, tol. $\varepsilon$", "Sec. III-E",
         _icon_robot),
        ("Throughput\n" + r"$\lambda$", "Sec. IV", _icon_bars),
    ]
    n = len(labels)
    left, right = 0.012, 0.988
    slot = (right - left) / n
    boxw, gap = slot * 0.79, slot * 0.21
    boxh, ytop = 0.50, 0.735
    cxs = [left + slot * (k + 0.5) for k in range(n)]
    for k, (title, ref, icon) in enumerate(labels):
        cx = cxs[k]
        derived = k <= 3
        ax.add_patch(FancyBboxPatch(
            (cx - boxw / 2, ytop - boxh), boxw, boxh,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            facecolor="#eef3fa" if derived else "#f5f1e8",
            edgecolor="#8fa6c4" if derived else "#c3b79c", lw=0.8, zorder=2))
        icon(ax, cx, ytop - 0.135, s=0.72)
        ax.text(cx, ytop - boxh + 0.180, title, ha="center", va="center",
                fontsize=5.7, zorder=5, linespacing=1.35)
        ax.text(cx, ytop - boxh + 0.052, ref, ha="center", va="center",
                fontsize=4.9, color="#6b6b6b", zorder=5)
        if k:
            ax.annotate("", xy=(cx - boxw / 2 - 0.004, ytop - boxh / 2),
                        xytext=(cxs[k - 1] + boxw / 2 + 0.004, ytop - boxh / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=1.1,
                                        color="#333333", mutation_scale=8,
                                        shrinkA=0, shrinkB=0), zorder=6)
    def span(x0, x1, y, colour, text):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3",
                                    lw=0.7, color=colour))
        ax.text((x0 + x1) / 2, y + 0.075, text, ha="center", va="center",
                fontsize=6.2, color=colour)
    span(cxs[0] - boxw / 2, cxs[3] + boxw / 2, 0.845, C["tolls"],
         "derived offline: zero simulator rollouts, one solve")
    span(cxs[4] - boxw / 2, cxs[5] + boxw / 2, 0.845, "#8a7b52", "simulated")
    ybot = ytop - boxh
    ax.plot([cxs[4], cxs[4]], [ybot, 0.105], lw=0.9, ls=(0, (3, 2)),
            color=C["tolls-online"], zorder=1)
    ax.plot([cxs[1], cxs[4]], [0.105, 0.105], lw=0.9, ls=(0, (3, 2)),
            color=C["tolls-online"], zorder=1)
    ax.annotate("", xy=(cxs[1], ybot - 0.004), xytext=(cxs[1], 0.105),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, ls=(0, (3, 2)),
                                color=C["tolls-online"], mutation_scale=8,
                                shrinkA=0, shrinkB=0), zorder=1)
    ax.text((cxs[1] + cxs[4]) / 2, 0.042,
            r"synchronous adaptation: pause and re-solve every "
            r"$T_{\mathrm{re}}{=}200$ steps",
            ha="center", va="center", fontsize=6.2, color=C["tolls-online"])
    save(fig, "fig_framework")


# -------------------------------------------------------- validity boundary --
# Okabe-Ito colour-blind-safe palette (Okabe & Ito 2008).
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7A7A7A"}


def _boundary_rows():
    """The 15 benchmark cells, open map first, with every quantity the central
    figure shows.  Read from two frozen result files; nothing is typed here.

      results/checks/tie_resolution.json  -- per-cell paired eps_tie contrast
          (code/analyze_tie_resolution.py, frozen native runs)
      results/native_check/summary_bench.json -- the scored native check
          (code/native/analyze_native_check.py, 600 locked runs)
    """
    tie_p = os.path.join(RESULTS, "checks", "tie_resolution.json")
    nat_p = os.path.join(RESULTS, "native_check", "summary_bench.json")
    with open(tie_p) as fh:
        tie = json.load(fh)
    with open(nat_p) as fh:
        nat = json.load(fh)
    ncell = {(c["map"], c["N"]): c for c in nat["cells"]}
    rows = []
    for r in tie["cells"]:
        key = (r["map"], r["N"])
        if key not in ncell:
            raise RuntimeError(f"native check has no cell {key}")
        rows.append(dict(
            map=r["map"], N=r["N"], open_map=bool(r["open_map"]),
            coarse=r["d_TU_coarse"]["mean"], coarse_ci=r["d_TU_coarse"]["ci95"],
            fine=r["d_TU_fine"]["mean"], fine_ci=r["d_TU_fine"]["ci95"],
            effect=r["d_fine_minus_coarse"]["mean"],
            effect_ci=r["d_fine_minus_coarse"]["ci95"],
            ours=ncell[key]["d_TU_python"], native=ncell[key]["d_TU_native"],
            sign_agree=bool(ncell[key]["sign_agree_TU"])))
    if len(rows) != 15:
        raise RuntimeError(f"expected 15 benchmark cells, got {len(rows)}")
    # open map on top, then the obstacle maps in the manuscript's order
    order = {m: i for i, m in enumerate(
        ["empty-32-32", "random-32-32-20", "room-32-32-4", "maze-32-32-4",
         "warehouse-10-20-10-2-1"])}
    rows.sort(key=lambda r: (order[r["map"]], r["N"]))
    return rows, tie


def fig_boundary(_s=None):
    """CENTRAL FIGURE: the validity boundary of rollout-free fluid guidance.

    (a) what a native planner does as it is made to follow the field more
        faithfully -- every open-map cell improves, every obstacle-map cell
        degrades, and the faithful endpoint itself is above zero on 3/3 open
        cells and below zero on 12/12 obstacle cells;
    (b) the guidance ordering our simulator reports against the ordering the
        compiled third-party planner reports on the same fields.
    """
    rows, tie = _boundary_rows()
    n = len(rows)
    n_open = sum(r["open_map"] for r in rows)
    # y positions, top to bottom, with a gap between the open and obstacle block
    ypos = []
    for i in range(n):
        ypos.append(-(i + (0.85 if i >= n_open else 0.0)))
    ypos = np.array(ypos, float)

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(7.03, 2.36), sharey=True, sharex=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.07,
                         left=0.145, right=0.995, top=0.87, bottom=0.265))

    for ax in (a1, a2):
        ax.tick_params(labelsize=7.35)
        ax.axvline(0.0, color=OI["black"], lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axhline((ypos[n_open - 1] + ypos[n_open]) / 2, color="#bbbbbb",
                   lw=0.6, zorder=1)
        ax.set_ylim(ypos[-1] - 0.75, ypos[0] + 0.75)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)

    # ---- (a) fidelity: eps_tie = 1 -> 0.01, native planner ------------------
    for y, r in zip(ypos, rows):
        col = OI["blue"] if r["effect"] > 0 else OI["vermillion"]
        a1.annotate("", xy=(r["fine"], y), xytext=(r["coarse"], y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.15, color=col,
                                    mutation_scale=6, shrinkA=0.6, shrinkB=0.0),
                    zorder=3)
        a1.plot([r["fine_ci"][0], r["fine_ci"][1]], [y, y], lw=2.4,
                color=col, alpha=0.30, solid_capstyle="butt", zorder=2)
        a1.plot([r["coarse"]], [y], marker="o", ms=3.1, mfc="white",
                mec=OI["black"], mew=0.8, ls="none", zorder=4)
        a1.plot([r["fine"]], [y], marker="o", ms=3.0, color=col, ls="none",
                zorder=4)
    a1.set_xlabel("native PIBT: tolls $-$ unweighted (tasks/step)")
    a1.set_title("(a) coarse bins ($\\varepsilon_{\\rm tie}{=}1$) "
                 "$\\rightarrow$ faithful ($0.01$)", fontsize=7.95, pad=3.0)

    # ---- (b) our simulator vs compiled third-party PIBT ---------------------
    for y, r in zip(ypos, rows):
        col = OI["purple"] if not r["sign_agree"] else OI["grey"]
        a2.annotate("", xy=(r["native"], y), xytext=(r["ours"], y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.15, color=col,
                                    mutation_scale=6, shrinkA=0.6, shrinkB=0.0),
                    zorder=3)
        a2.plot([r["ours"]], [y], marker="s", ms=3.0, mfc="white",
                mec=OI["black"], mew=0.8, ls="none", zorder=4)
        a2.plot([r["native"]], [y], marker="s", ms=2.9, color=col, ls="none",
                zorder=4)
    a2.set_xlabel("same axis, coarse bins ($\\varepsilon_{\\rm tie}{=}1$) in both arms")
    a2.set_title("(b) our simulator $\\rightarrow$ compiled third-party PIBT",
                 fontsize=7.95, pad=3.0)

    a1.set_yticks(ypos)
    a1.set_yticklabels([f"{SHORT[r['map']]}  $N{{=}}{r['N']}$" for r in rows],
                       fontsize=7.35)
    for lab, r in zip(a1.get_yticklabels(), rows):
        lab.set_color(OI["black"])

    ob = tie["obstacle_maps_only"]
    op = tie["open_map_only"]
    n_flip = sum(1 for r in rows if not r["sign_agree"])
    h1 = [Line2D([], [], color=OI["blue"], lw=1.3, marker="o", ms=3.2,
                 label=f"faithful guidance is better: "
                       f"{op['n_cells'] - op['n_fine_worse']}/{op['n_cells']} open-map cells"),
          Line2D([], [], color=OI["vermillion"], lw=1.3, marker="o", ms=3.2,
                 label=f"faithful guidance is worse: {ob['n_fine_worse']}/{ob['n_cells']} "
                       f"obstacle-map cells")]
    h2 = [Line2D([], [], color=OI["grey"], lw=1.3, marker="s", ms=3.0,
                 label=f"guidance ordering agrees ({n - n_flip}/{n} cells)"),
          Line2D([], [], color=OI["purple"], lw=1.3, marker="s", ms=3.0,
                 label=f"guidance ordering reverses ({n_flip}/{n} cells)")]
    fig.legend(h1, [h.get_label() for h in h1], loc="upper left",
               bbox_to_anchor=(0.145, 0.115), frameon=False, fontsize=7.35,
               ncol=1, handlelength=1.9, handletextpad=0.5,
               labelspacing=0.28, borderpad=0.0)
    fig.legend(h2, [h.get_label() for h in h2], loc="upper left",
               bbox_to_anchor=(0.655, 0.115), frameon=False, fontsize=7.35,
               ncol=1, handlelength=1.9, handletextpad=0.5,
               labelspacing=0.28, borderpad=0.0)
    save(fig, "fig_boundary")


# --------------------------------------------------------------- headline ---
def fig_headline(s):
    rows = s["headline"]
    maps = list(dict.fromkeys(r["map"] for r in rows))
    fig, axes = plt.subplots(1, len(maps), figsize=(7.1, 1.9), sharey=False)
    meths = ["unweighted", "lanes", "tolls", "tolls-online"]
    for ax, mp_ in zip(np.atleast_1d(axes), maps):
        rr = [r for r in rows if r["map"] == mp_]
        x = np.arange(len(rr))
        w = 0.2
        for k, meth in enumerate(meths):
            mu_ = [r.get(meth, {}).get("mean", np.nan) for r in rr]
            sd = [r.get(meth, {}).get("std", 0) for r in rr]
            ax.bar(x + (k - 1.5) * w, mu_, w, yerr=sd, capsize=1.5,
                   color=C[meth], label=NAME[meth], error_kw=dict(lw=0.6))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['N']}\n{r['density']:.0f}%" for r in rr])
        ax.set_title(SHORT.get(mp_, mp_), fontsize=8)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("throughput (tasks/step)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[m]) for m in meths]
    fig.legend(handles, [NAME[m] for m in meths], loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.text(0.5, -0.06, "fleet size $N$ (and density)", ha="center", fontsize=8)
    save(fig, "fig_headline")


# ------------------------------------------------- compute + fluid (merged) ---
def fig_compute(s):
    """Left: guidance-construction cost vs throughput. Right: the model rate
    vs what the simulator delivers. Merged into one
    single-column float to stay inside the page limit."""
    ct = s.get("compute_vs_tput", {})
    rate = ct.get("_eval_rate_s")
    d = {k: v for k, v in ct.items() if isinstance(v, dict)}
    fl = s.get("fluid")
    if not d:
        return
    W, H = COL_PT, 161.2                 # \\columnwidth, aspect unchanged
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(_in(W), _in(H)))
    fig.subplots_adjust(left=0.125, right=0.995, top=0.905, bottom=0.517,
                        wspace=0.46)
    for lab, v in d.items():
        # For searches, wall-clock was contaminated by CPU contention; plot the
        # contention-free equivalent (evaluations x measured per-eval rate) so
        # the x-axis is comparable across methods.
        x = v["compute_s"]
        if rate and v.get("n_evals"):
            x = v["n_evals"] * rate
        col = C.get(lab, "k")
        a1.errorbar(max(x, 0.01), v["tput"], yerr=v["std"],
                    fmt=M.get(lab, "o"), ms=3.8, capsize=1.2, color=col,
                    mfc="white" if lab in COLD else col,
                    mec="black",
                    mew=0.6, label=TERSE.get(lab))
    a1.set_xscale("log")
    a1.set_xticks([0.01, 1.0, 100.0])
    a1.set_xticklabels(["0.01", "1", "100"])
    a1.minorticks_off()
    a1.set_xlabel("construction compute (s, log)", fontsize=FS)
    a1.set_ylabel("throughput (tasks/step)", fontsize=FS)
    a1.tick_params(labelsize=FS, pad=1.5, length=2)
    a1.set_title("(a) cost vs throughput", fontsize=FS_TITLE, pad=2.5)
    if fl:
        bymap, mk = {}, ["o", "s", "D", "^", "v"]
        for p_ in fl["points"]:
            bymap.setdefault(p_["map"], []).append(p_)
        for k, (mp_, pts) in enumerate(sorted(bymap.items())):
            a2.errorbar([p_["predicted"] for p_ in pts],
                        [p_["realised"] for p_ in pts],
                        yerr=[p_["std"] for p_ in pts], fmt=mk[k % len(mk)],
                        ms=2.9, capsize=1.0, mec="black", mew=0.4, lw=0,
                        elinewidth=0.7, label=SHORT.get(mp_, mp_))
        lim = max(max(p_["predicted"] for p_ in fl["points"]),
                  max(p_["realised"] for p_ in fl["points"])) * 1.08
        a2.plot([0, lim], [0, lim], "k--", lw=0.7)
        a2.set_xlim(0, lim); a2.set_ylim(0, lim)
        a2.set_xlabel("fluid model rate $\\lambda$", fontsize=FS)
        a2.set_ylabel("measured (tolls)", fontsize=FS)
        a2.tick_params(labelsize=FS, pad=1.5, length=2)
        a2.locator_params(nbins=5)
        a2.set_title("(b) model agreement", fontsize=FS_TITLE, pad=2.5)
    # Legends live outside, below the panels, so they never occlude data.
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    h1, l1 = zip(*[(a, b) for a, b in zip(h1, l1) if b])
    fig.legend(h1, l1, loc="upper left", bbox_to_anchor=(0.02, 0.378),
               ncol=2, frameon=False, fontsize=FS, handletextpad=0.35,
               columnspacing=1.0, labelspacing=0.24, borderpad=0.0)
    fig.legend(h2, l2, loc="upper left", bbox_to_anchor=(0.02, 0.138),
               ncol=3, frameon=False, fontsize=FS, handletextpad=0.35,
               columnspacing=1.0, labelspacing=0.24, borderpad=0.0)
    save_at(fig, "fig_compute_tput")


# --------------------------------------------------------------- transfer ---
def fig_transfer(s):
    ac = s.get("transfer_agentcount", {})
    ds = s.get("transfer_demandshift", {})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.2))
    style = {"tolls-frozen": ("-", "tolls frozen @N=400"),
             "tolls-resolved": ("--", "tolls re-solved per $N$"),
             "tolls-online": ("-", "tolls online"),
             "lanes": ("-", "crisscross lanes"),
             "unweighted": ("-", "unweighted")}
    ck = {"tolls-frozen": C["tolls"], "tolls-resolved": "#17becf",
          "tolls-online": C["tolls-online"], "lanes": C["lanes"],
          "unweighted": C["unweighted"]}
    for lab, series in ac.items():
        Ns = sorted(int(k) for k in series)
        y = [series[str(n)]["mean"] if str(n) in series else series[n]["mean"]
             for n in Ns]
        e = [series[str(n)]["std"] if str(n) in series else series[n]["std"]
             for n in Ns]
        ls, nm = style.get(lab, ("-", lab))
        a1.errorbar(Ns, y, yerr=e, ls=ls, marker="o", ms=3, lw=1.2,
                    color=ck.get(lab, "k"), label=nm, capsize=1.5)
    a1.set_xlabel("agents $N$ (weights built at $N{=}400$)")
    a1.set_ylabel("throughput (tasks/step)")
    a1.legend(frameon=False, fontsize=6.5)
    a1.set_title("(a) fleet-size transfer", fontsize=8)

    for lab, series in ds.items():
        hs = sorted(float(k) for k in series)
        y = [series[str(h)]["mean"] for h in hs]
        e = [series[str(h)]["std"] for h in hs]
        a2.errorbar(hs, y, yerr=e, marker="o", ms=3, lw=1.2,
                    color=C.get(lab, "k"), label=NAME.get(lab, lab), capsize=1.5)
    a2.set_xlabel("task-distribution shift (hot-spot probability)")
    a2.set_ylabel("throughput (tasks/step)")
    a2.legend(frameon=False, fontsize=6.5)
    a2.set_title("(b) demand-shift transfer", fontsize=8)
    save(fig, "fig_transfer")


# ------------------------------------------------------------- directional ---
def _net_field(m, f):
    """Per-cell net flow vector from directed edge flows."""
    edges, rev = mapmod.directed_edges(m)
    W = m["W"]
    H = m["H"]
    U = np.zeros((H, W)); V = np.zeros((H, W))
    net = np.asarray(f, float) - np.asarray(f, float)[rev]
    for k, (u, v) in enumerate(edges):
        uy, ux = divmod(int(u), W)
        vy, vx = divmod(int(v), W)
        U[uy, ux] += net[k] * (vx - ux)
        V[uy, ux] += net[k] * (vy - uy)
    return U, V


def fig_directional(s):
    """Net-flow quiver: model flow, tolled traffic, symmetric-init tolls."""
    import csv
    mp_, N = "empty-32-32", 400
    m = mapmod.get_map(mp_)
    _, rev = mapmod.directed_edges(m)
    panels = []
    f_model, _ = guid.load_toll_flow(m, N, gamma=GAMMA, mu=MU, cache_dir=CACHE)
    panels.append(("model $\\hat f$", f_model))
    try:
        f_sym, _ = guid.load_toll_flow(m, N, gamma=GAMMA, mu=MU,
                                       seed_lanes=False, cache_dir=CACHE)
        panels.append(("symmetric init", f_sym))
    except Exception:
        pass
    recs = {}
    try:
        for r in common.load_results("exp4b"):
            c = r["cond"]
            if c["map"] == mp_ and c["N"] == N and "edge_counts_file" in r:
                recs[c.get("label", c["method"])] = r
    except FileNotFoundError:
        pass
    for lab in ("unweighted", "tolls"):
        if lab in recs:
            z = np.load(os.path.join(common.RESULTS, recs[lab]["edge_counts_file"]))
            short = {"unweighted": "sim.: unweighted",
                     "tolls": "sim.: tolls"}[lab]
            panels.append((short, z["edge_counts"].astype(float)))
    n = len(panels)
    W, H = COL_PT, 72.7                  # \columnwidth, aspect unchanged
    fig, axes = plt.subplots(1, n, figsize=(_in(W), _in(H)))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.815, bottom=0.175,
                        wspace=0.12)
    for ax, (title, f) in zip(np.atleast_1d(axes), panels):
        U, V = _net_field(m, f)
        d = np.abs(np.asarray(f, float) - np.asarray(f, float)[rev]).sum() / \
            max((np.asarray(f, float) + np.asarray(f, float)[rev]).sum(), 1e-12)
        # NO spatial coarsening: the lane structure alternates cell by cell,
        # so block-averaging would cancel exactly the signal being shown.
        Uc, Vc = U, V
        mag = np.hypot(Uc, Vc)
        ax.quiver(np.arange(Uc.shape[1]), np.arange(Uc.shape[0]), Uc, -Vc, mag,
                  cmap="viridis", scale_units="xy", angles="xy",
                  scale=float(np.percentile(mag[mag > 0], 90) + 1e-9) * 1.1,
                  width=0.006, headwidth=3.0, headlength=3.5, minlength=0.2)
        # One-line title (identity) plus Dir beneath the panel: at 7 pt a
        # two-line title does not fit, and the symmetric-init panel is blank
        # precisely because its Dir is zero, so the number has to stay visible.
        ax.set_title(title, fontsize=FS, pad=2.5)
        ax.set_xlabel(f"Dir$\\,={d:.2f}$", fontsize=FS, labelpad=1.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.invert_yaxis()
    save_at(fig, "fig_directional")


# -------------------------------------------------------------- ablations ---
def fig_ablations(s):
    ab = s.get("ablations", {})
    if not ab:
        return
    # beta and T_re are reported numerically in the text; so is the swap
    # ablation, which Fig. 2 already plots -- the panel that repeated it has
    # been dropped so the remaining text can stay above 7 pt.
    groups = [("gamma", "head-on $\\gamma$"), ("mu", "vertex cap. $\\mu$"),
              ("eps", "tie tol. $\\varepsilon$")]
    cells = sorted(ab)
    W, H = COL_PT, 137.5                 # \\columnwidth, aspect unchanged
    fig, axes = plt.subplots(len(cells), len(groups), squeeze=False,
                             figsize=(_in(W), _in(H)))
    fig.subplots_adjust(left=0.175, right=0.995, top=0.915, bottom=0.205,
                        wspace=0.42, hspace=0.72)
    for i, cell in enumerate(cells):
        base = ab[cell].get("unweighted-swap=True")
        for j, (pre, title) in enumerate(groups):
            ax = axes[i][j]
            items = sorted(((k, v) for k, v in ab[cell].items()
                            if k.startswith(pre + "=")),
                           key=lambda kv: float(kv[0].split("=")[1]))
            if not items:
                ax.axis("off"); continue
            xs = [f"{float(k.split(chr(61))[1]):g}" for k, _ in items]
            ys = [v["mean"] for _, v in items]
            es = [v["std"] for _, v in items]
            ax.bar(xs, ys, yerr=es, capsize=1.2, color=C["tolls"],
                   error_kw=dict(lw=0.6))
            if base:
                ax.axhline(base["mean"], color=C["unweighted"], ls=":", lw=1)
            ax.set_title(title, fontsize=FS, pad=2.5)
            ax.tick_params(labelsize=FS, pad=1.5, length=2)
            ax.locator_params(axis="y", nbins=4)
            if len(xs) > 4:   # five sweep points do not fit side by side at 7 pt
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(xs, rotation=45, ha="right",
                                   rotation_mode="anchor")
            if j == 0:
                ax.set_ylabel(f"{SHORT.get(cell.split(chr(64))[0], cell)}\n"
                              f"tasks/step", fontsize=FS)
    handles = [Line2D([], [], color=C["unweighted"], ls=":", lw=1.2),
               Rectangle((0, 0), 1, 1, color=C["tolls"])]
    fig.legend(handles, ["unweighted PIBT (dotted)", "derived tolls (bars)"],
               loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=2,
               frameon=False, fontsize=FS, handlelength=1.9,
               handletextpad=0.5, columnspacing=1.6, borderpad=0.0)
    save_at(fig, "fig_ablations")


# ------------------------------------------------------------------ fluid ---
def fig_fluid(s):
    fl = s.get("fluid")
    if not fl:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    bymap = {}
    for p in fl["points"]:
        bymap.setdefault(p["map"], []).append(p)
    for mp_, pts in bymap.items():
        ax.errorbar([p["predicted"] for p in pts], [p["realised"] for p in pts],
                    yerr=[p["std"] for p in pts], fmt="o", ms=4, capsize=1.5,
                    label=SHORT.get(mp_, mp_))
    lim = max(max(p["predicted"] for p in fl["points"]),
              max(p["realised"] for p in fl["points"])) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="$y=x$")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("fluid-model predicted $\\lambda$")
    ax.set_ylabel("realised throughput (tolls)")
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig_fluid")


# -------------------------------------------------------------- livelock ----
def fig_livelock():
    """Plot the already-frozen swap/no-swap traces; never run a simulation."""
    W, H = 0.85 * COL_PT, 148.0          # 0.85\columnwidth, aspect unchanged
    fig, ax = plt.subplots(figsize=(_in(W), _in(H)))
    fig.subplots_adjust(left=0.235, right=0.985, top=0.885, bottom=0.335)
    records = common.load_results("exp6b")
    for swap, ls, lab in [(False, "--", "local no-swap"),
                          (True, "-", "local + swap")]:
        matches = [r for r in records
                   if r["cond"]["map"] == "random-32-32-20"
                   and r["cond"]["N"] == 200
                   and r["cond"]["method"] == "unweighted"
                   and r["cond"]["seed"] == 0
                   and r["cond"].get("label") == f"unweighted-swap={swap}"]
        if len(matches) != 1:
            raise RuntimeError(f"expected one frozen swap={swap} trace, got {len(matches)}")
        r = matches[0]
        c = np.array([d["completed"] for d in r["series"]], float)
        t = np.array([d["t"] for d in r["series"]], float)
        rate = np.diff(np.concatenate([[0], c])) / np.diff(np.concatenate([[0], t]))
        ax.plot(t, rate, ls, lw=1.4, label=lab,
                color=C["unweighted"] if not swap else C["tolls"])
    ax.set_xlabel("timestep", fontsize=FS)
    ax.set_ylabel("throughput in window\n(tasks/step)", fontsize=FS)
    ax.set_title("random-32-32-20, $N{=}200$, unweighted", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS)
    ax.set_ylim(bottom=0)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.58, 0.005), ncol=2, frameon=False,
               fontsize=FS, handlelength=2.2, handletextpad=0.5,
               columnspacing=1.4, borderpad=0.0)
    save_at(fig, "fig_livelock")


def main():
    s = S()
    print("figures ->", FIGDIR)
    # fig_headline and fig_transfer are superseded by Tables I and II, and the
    # study-design schematic fig_framework was replaced as Fig. 1 by the
    # validity-boundary figure the paper is now built around; none of the three
    # is emitted, so no unreferenced figure can end up in paper/figures.
    for fn in (fig_boundary, fig_compute, fig_directional, fig_ablations):
        try:
            fn(s)
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    try:
        fig_livelock()
    except Exception as e:
        print(f"  SKIP fig_livelock: {e}")


if __name__ == "__main__":
    main()
