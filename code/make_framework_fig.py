"""Study-design figure: rollout-free guidance construction and where it is valid.

Structural only -- no measured quantity is typed here; every number that makes a
claim lives in a generated macro in the manuscript.  Writes paper/figures/fig_framework.pdf.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, Circle, FancyBboxPatch,
                                FancyArrowPatch, Polygon)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

FIGDIR = os.path.join(common.PROJ, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 6.2, "figure.dpi": 150})

# Okabe-Ito, matching make_figures_v2.py; every stage is also distinguished by
# its icon and its position, never by colour alone.
BLUE, ORANGE, GREEN, PINK, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#5a5a5a"


def _stage(ax, x, y, w, h, colour, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=0.9, edgecolor=colour,
                                facecolor=colour + "14", zorder=2))
    ax.text(x + w / 2, y + h - 0.030, title, ha="center", va="top",
            fontsize=6.6, fontweight="bold", color=colour, zorder=5)


def _arrow(ax, x0, y0, x1, y1, colour=GREY, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=7, linewidth=lw,
                                 color=colour, zorder=6,
                                 shrinkA=0, shrinkB=0))


def _gridmap(ax, cx, cy, s, obstacles=True):
    """Icon: a 4x4 warehouse grid, optionally with two blocked cells."""
    n, c = 4, s / 4.0
    for i in range(n):
        for j in range(n):
            blocked = obstacles and ((i, j) in ((1, 1), (2, 2)))
            ax.add_patch(Rectangle((cx - s / 2 + i * c, cy - s / 2 + j * c), c, c,
                                   facecolor=(GREY if blocked else "white"),
                                   edgecolor=GREY, linewidth=0.35, zorder=4))


def _robot(ax, cx, cy, r, colour):
    ax.add_patch(FancyBboxPatch((cx - r, cy - r * 0.72), 2 * r, 1.44 * r,
                                boxstyle="round,pad=0.004", facecolor=colour,
                                edgecolor="none", zorder=5))
    ax.plot([cx, cx], [cy + r * 0.72, cy + r * 1.25], color=colour, lw=0.7, zorder=5)
    ax.add_patch(Circle((cx, cy + r * 1.4), r * 0.22, facecolor=colour,
                        edgecolor="none", zorder=5))
    for dx in (-r * 0.45, r * 0.45):
        ax.add_patch(Circle((cx + dx, cy + r * 0.1), r * 0.2, facecolor="white",
                            edgecolor="none", zorder=6))


def _flow(ax, cx, cy, s, colour):
    """Icon: three directed streamlines converging -- the fluid relaxation."""
    for k, dy in enumerate((-0.34, 0.0, 0.34)):
        y = cy + dy * s
        _arrow(ax, cx - s * 0.52, y, cx + s * 0.52, cy + dy * s * 0.30,
               colour=colour, lw=0.9 + 0.35 * (k == 1))


def _toll(ax, cx, cy, s, colour):
    """Icon: a directed edge carrying a weight tag."""
    _arrow(ax, cx - s * 0.5, cy - s * 0.22, cx + s * 0.5, cy - s * 0.22,
           colour=colour, lw=1.3)
    ax.add_patch(FancyBboxPatch((cx - s * 0.20, cy + s * 0.06), s * 0.40, s * 0.30,
                                boxstyle="round,pad=0.008", facecolor="white",
                                edgecolor=colour, linewidth=0.8, zorder=6))
    ax.text(cx, cy + s * 0.21, r"$d_w$", ha="center", va="center",
            fontsize=6.0, color=colour, zorder=7)


def _verdict(ax, cx, cy, s, ok, colour):
    ax.add_patch(Circle((cx, cy), s, facecolor=colour + "22", edgecolor=colour,
                        linewidth=0.8, zorder=5))
    if ok:
        ax.plot([cx - s * 0.42, cx - s * 0.08, cx + s * 0.46],
                [cy + s * 0.02, cy - s * 0.34, cy + s * 0.36],
                color=colour, lw=1.2, solid_capstyle="round", zorder=6)
    else:
        for a, b in (((-1, -1), (1, 1)), ((-1, 1), (1, -1))):
            ax.plot([cx + a[0] * s * 0.38, cx + b[0] * s * 0.38],
                    [cy + a[1] * s * 0.38, cy + b[1] * s * 0.38],
                    color=colour, lw=1.2, solid_capstyle="round", zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(3.42, 1.16))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ytop, hbox, w = 0.030, 0.760, 0.212
    xs = [0.001, 0.263, 0.525, 0.787]
    yicon, ytext = ytop + 0.520, ytop + 0.195

    _stage(ax, xs[0], ytop, w, hbox, BLUE, "frozen cells")
    _gridmap(ax, xs[0] + w / 2, yicon, 0.098)
    ax.text(xs[0] + w / 2, ytext, "five maps\n" + r"$\times$ three" + "\ndensities",
            ha="center", va="center", fontsize=5.5, color=GREY, linespacing=1.30)

    _stage(ax, xs[1], ytop, w, hbox, GREEN, "fluid solve")
    _flow(ax, xs[1] + w / 2, yicon, 0.128, GREEN)
    ax.text(xs[1] + w / 2, ytext, "Frank\u2013Wolfe on\ncongestion,\nhead-on, load",
            ha="center", va="center", fontsize=5.5, color=GREY, linespacing=1.30)

    _stage(ax, xs[2], ytop, w, hbox, ORANGE, "weights")
    _toll(ax, xs[2] + w / 2, yicon, 0.130, ORANGE)
    ax.text(xs[2] + w / 2, ytext, "marginal social\ncosts in the\ngradient",
            ha="center", va="center", fontsize=5.5, color=GREY, linespacing=1.30)

    _stage(ax, xs[3], ytop, w, hbox, PINK, "planner")
    _robot(ax, xs[3] + w / 2 - 0.038, yicon, 0.026, PINK)
    _robot(ax, xs[3] + w / 2 + 0.038, yicon, 0.026, PINK)
    ax.text(xs[3] + w / 2, ytext + 0.012,
            r"reads only" "\n" r"round$(d_w/\epsilon_{\mathrm{tie}})$",
            ha="center", va="center", fontsize=5.5, color=GREY, linespacing=1.30)

    for i in range(3):
        _arrow(ax, xs[i] + w + 0.009, ytop + hbox / 2,
               xs[i + 1] - 0.009, ytop + hbox / 2, colour=GREY, lw=1.0)

    ax.text(0.5, 1.000, "guidance built with no simulator rollouts\n"
                        "(prior fitted guidance: tens of thousands per map)",
            ha="center", va="top", fontsize=5.8, style="italic", color=GREY,
            linespacing=1.25)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_framework.{ext}"),
                    bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", os.path.join(FIGDIR, "fig_framework.pdf"))

if __name__ == "__main__":
    main()
