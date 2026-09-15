"""exp13: one combined figure for validity + certifiable horizon.

Replaces the separate `fig_validity` (2 stacked panels) and `fig_radius` (2
panels) with a single full-width 1x3 figure, which carries the same evidence in
roughly half the column area:

  (a) time-uniform validity: P(any undercoverage by t) for the CS at four
      (beta, delta) settings against a plug-in quantile recomputed at every
      score update -- an uncorrected repeated-testing baseline;
  (b) inflation radius vs episode horizon, with the two guarantee families kept
      apart and crosses marking where each certificate expires;
  (c) certifiable windows vs calibration-set size (the Theta(sqrt(delta n)) vs
      Theta(delta n) separation).

All legends are placed outside, below their panel.

Run: .venv/bin/python code/exp13_core_fig.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.0,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.35,
})
MCOL = {"cs": "#d62728", "auc": "#ff7f0e", "aucb": "#8c564b",
        "union": "#1f77b4", "aci": "#2ca02c", "gauss": "#9467bd"}


def below(ax, ncol=2, y=-0.30, fs=5.3):
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     frameon=False, fontsize=fs, handletextpad=0.4,
                     columnspacing=1.0, labelspacing=0.25, borderpad=0.0,
                     handlelength=1.6)


def panel_validity(ax):
    d = json.load(open(os.path.join(RES, "exp1_validity.json")))
    for rec in d["cs_time_uniform"]:
        prof = np.asarray(rec["profile_t_rate"], dtype=float)
        ax.plot(prof[:, 0], prof[:, 1], lw=1.0,
                label=f"CS $\\beta{{=}}{rec['beta']}$, $\\delta{{=}}{rec['delta']}$")
    pl = [r["anytime_undercoverage_rate"] for r in d["plugin_time_uniform"]]
    ax.axhspan(min(pl), max(pl), color="0.55", alpha=0.30, lw=0)
    ax.axhline(float(np.mean(pl)), color="0.35", ls="--", lw=1.0,
               label="plug-in quantile, repeated per update")
    for tgt in (0.05, 0.10):
        ax.axhline(tgt, color="k", ls=":", lw=0.6,
                   label="target $\\delta$" if tgt == 0.05 else None)
    ax.set_xscale("log")
    ax.set_xlabel("score updates $t$")
    ax.set_ylabel("P(any undercoverage by $t$)")
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    below(ax, ncol=2, y=-0.30)


def panel_radius(ax, d=0.05):
    df = pd.read_csv(os.path.join(RES, "exp2_radius.csv"))
    u = df[(df.curve == "union") & (df.delta == d)].sort_values("x")
    fin = u[np.isfinite(u.radius)]
    Tb = int(u[~np.isfinite(u.radius)].x.min())
    ax.plot(fin.x, fin.radius, color=MCOL["union"], label="Union CP ($T$ known)")
    ax.axvline(Tb, color=MCOL["union"], ls=":", lw=0.8)
    ax.annotate(f"uncertifiable\n$T\\geq{Tb}$", (Tb * 1.15, 3.25), fontsize=5.4,
                color=MCOL["union"])
    for cur, lab in [("auc", "AUC (ours)"), ("aucb", "AUC-B (ours)")]:
        a = df[(df.curve == cur) & (df.delta == d)].sort_values("x")
        a = a[np.isfinite(a.radius)]
        ax.plot(a.x, a.radius, color=MCOL[cur], ls="--", marker=".", ms=2.0, label=lab)
        if len(a):
            ax.plot([a.x.iloc[-1]], [a.radius.iloc[-1]], "x", ms=4.5, mew=1.1,
                    color=MCOL[cur])
    g = df[(df.curve == "gauss") & (df.delta == d)].sort_values("x")
    ax.plot(g.x, g.radius, color=MCOL["gauss"], ls="-.", lw=0.8, label="Gauss CC")
    ax.axhline(float(df[(df.curve == "cs") & (df.delta == d)].radius.iloc[-1]),
               color=MCOL["cs"], lw=1.3, label="CS (ours)")
    ax.axhline(float(df[(df.curve == "plugin") & (df.delta == d)].radius.iloc[0]),
               color="k", ls=":", lw=0.8, label="plug-in $\\hat Q_{1-\\epsilon}$")
    ax.set_xscale("log")
    ax.set_xlabel("episode horizon / window index")
    ax.set_ylabel("inflation radius [m]")
    ax.set_ylim(0, 4.3)
    ax.grid(True)
    below(ax, ncol=2, y=-0.30)


def panel_horizon(ax, d=0.05):
    hz = pd.read_csv(os.path.join(RES, "exp2_horizon.csv"))
    hh = hz[hz.delta == d]
    for meth, col, ls, lab in [
            ("optimal", "k", "-", "exact harmonic cutoff"),
            ("union_anyalloc", MCOL["union"], "-", "Union CP, any alloc."),
            ("aucb_lam0.01", MCOL["aucb"], "--", "AUC-B $\\lambda{=}0.01$"),
            ("aucb_lam0.05", MCOL["aucb"], ":", "AUC-B $\\lambda{=}0.05$"),
            ("auc", MCOL["auc"], "--", "AUC $\\delta/w(w{+}1)$")]:
        q = hh[hh.method == meth].sort_values("n")
        ax.plot(q.n, np.maximum(q.horizon, 0.5), color=col, ls=ls, lw=1.0, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(2094, color="#999999", lw=0.6, ls=":")
    ax.text(2400, 2.4e3, "our $n$", fontsize=5.2, color="#666666")
    ax.text(0.03, 0.95, "CS (ours): unbounded", transform=ax.transAxes,
            fontsize=5.6, color=MCOL["cs"], va="top")
    ax.set_xlabel("calibration scores $n$")
    ax.set_ylabel("certifiable windows")
    ax.grid(True, which="both")
    below(ax, ncol=2, y=-0.30)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.30))
    panel_validity(axes[0])
    panel_radius(axes[1])
    panel_horizon(axes[2])
    for ax, t in zip(axes, "abc"):
        ax.set_title(f"({t})", loc="left", fontsize=7)
    fig.tight_layout(pad=0.35, w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_core.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(FIG, "fig_core.pdf"))


if __name__ == "__main__":
    main()
