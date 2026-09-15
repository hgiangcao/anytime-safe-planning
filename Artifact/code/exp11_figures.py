"""exp11: figure for the three-level guarantee (G3), robustness, and matched
calibration -- the material added in the revision.

  fig_anytime : (a) cumulative violations vs the anytime budget B_W over a
                    120-window deployment, with the windows at which each
                    episode-level method stops having ANY statement;
                (b) the shift-robust CS: window violation rate against target in
                    two held-out stress regimes, plain vs robust under an
                    illustrative (not data-certified) budget rho;
                (c) matched calibration: with n large enough for the union bound
                    to certify all 120 windows, what the episode-level
                    certificate actually costs in throughput.

Run: .venv/bin/python code/exp11_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa: F401
from csq import violation_budget_curve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LONG = os.path.join(RES, "long")
FIG = os.path.join(RES, "figures")
CACHE = os.path.join(RES, "cache")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.0,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.35,
})
C = {"cs": "#d62728", "cs_r": "#7f2704", "union": "#1f77b4", "aucb": "#8c564b",
     "auc": "#ff7f0e", "aci": "#2ca02c", "aci_ep": "#17becf", "uncal": "#7f7f7f"}


def panel_a(ax, delta=0.05, regime="long_replay"):
    B = violation_budget_curve(delta, delta, 200, cache_dir=CACHE)
    W = 120
    ax.step(np.arange(1, W + 1), B[:W], where="post", color="k", lw=1.2,
            label=r"anytime budget $B_W$")
    ax.plot(np.arange(1, W + 1), delta * np.arange(1, W + 1), ":", color="k",
            lw=0.8, label=r"$\epsilon W$ (mean)")
    for m in ("cs", "aci", "aucb", "auc"):
        p = os.path.join(LONG, f"{regime}_{m}_{delta}.json")
        if not os.path.exists(p):
            continue
        recs = json.load(open(p))["records"]
        curves, first_unc = [], []
        for r in recs:
            v, seen = [], 0
            fu = None
            for w, (s, rad) in enumerate(r["windows"][:W], start=1):
                if rad is None:
                    fu = w if fu is None else fu
                else:
                    seen += int(s > rad)
                v.append(seen)
            curves.append(v + [v[-1]] * (W - len(v)))
            first_unc.append(fu)
        mean = np.mean(np.array(curves), axis=0)
        ax.plot(np.arange(1, W + 1), mean, color=C[m], lw=1.1,
                label={"cs": "CS (ours)", "aci": "ACI", "aucb": "AUC-B (ours)",
                       "auc": "AUC (ours)"}[m])
        fu = [x for x in first_unc if x is not None]
        if fu:
            wstar = int(np.median(fu))
            ax.plot([wstar], [mean[wstar - 1]], "x", color=C[m], ms=5, mew=1.2)
            ax.annotate(f"$w^\\ast{{=}}{wstar}$", (wstar, mean[wstar - 1]),
                        textcoords="offset points", xytext=(-6, -11),
                        fontsize=5.5, color=C[m])
    ax.set_xlabel("window $W$")
    ax.set_ylabel("cumulative violations $V_W$")
    ax.set_xlim(1, W)
    ax.grid(True)
    ax.set_ylim(bottom=-1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
              frameon=False, fontsize=5.4, handlelength=1.6,
              handletextpad=0.4, columnspacing=1.0, labelspacing=0.25,
              borderpad=0.0)
    ax.annotate(r"$\times$: certificate gone", (0.40, 0.10),
                xycoords="axes fraction", fontsize=5.2, color="0.35")


def panel_b(ax):
    veh = json.load(open(os.path.join(RES, "exp10_vehicle.json")))["conditions"]
    pr = json.load(open(os.path.join(RES, "exp8_robust.json")))["price"]
    labs, plain, rob, tgt = [], [], [], []
    for tag, nice in (("rob_orca", "ORCA"), ("rob_real", "real")):
        for d in (0.05, 0.10):
            k1, k2 = f"{tag}|cs|{d}", f"{tag}|cs_r|{d}"
            if k1 not in veh or k2 not in veh:
                continue
            labs.append(f"{nice}\n{d:g}")
            plain.append(veh[k1]["window_viol_rate"])
            rob.append(veh[k2]["window_viol_rate"])
            tgt.append(d)
    x = np.arange(len(labs))
    ax.bar(x - 0.19, plain, 0.36, color=C["cs"], label="CS")
    ax.bar(x + 0.19, rob, 0.36, color=C["cs_r"],
           label=r"CS-R ($\rho{=}12$, illustrative)")
    for i, t in enumerate(tgt):
        ax.plot([i - 0.42, i + 0.42], [t, t], "k--", lw=0.9,
                label="target $\\epsilon$" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=5)
    ax.set_ylabel("window violation rate")
    ax.set_ylim(0, max(plain) * 1.18)
    ax.grid(True, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
              frameon=False, fontsize=5.4, handlelength=1.6,
              handletextpad=0.4, columnspacing=1.0, labelspacing=0.25,
              borderpad=0.0)
    ax.set_xlabel(r"regime / target $\epsilon$", fontsize=6, labelpad=1)
    ax.tick_params(axis="x", pad=1)


def panel_c(ax):
    d = json.load(open(os.path.join(RES, "exp9_matched.json")))
    m = d["matched"]
    order = [k for k in ("cs", "union", "aci_ep", "aucb") if k in m]
    lab = {"cs": "CS\n(ours)", "union": "Union CP", "aci_ep": "ACI-ep",
           "aucb": "AUC-B\n(ours)"}
    x = np.arange(len(order))
    g = [m[k]["mean_goals"] for k in order]
    r = [m[k]["mean_radius"] for k in order]
    ax.bar(x, g, 0.55, color=[C[k] for k in order])
    for i, (gi, ri, k) in enumerate(zip(g, r, order)):
        ax.annotate(f"{ri:.2f} m\ncert {m[k]['cert_window_frac']:.2f}",
                    (i, gi), ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 1.5), fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab[k] for k in order], fontsize=6)
    ax.set_ylabel("goals per 120-window episode")
    ax.set_ylim(0, max(g) * 1.35)
    ax.grid(True, axis="y")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.92))
    panel_a(axes[0])
    panel_b(axes[1])
    for ax, t in zip(axes, "ab"):
        ax.set_title(f"({t})", loc="left", fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.7)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_anytime.{ext}"), dpi=200,
                    bbox_inches="tight")
    print("wrote", os.path.join(FIG, "fig_anytime.pdf"))


if __name__ == "__main__":
    main()
