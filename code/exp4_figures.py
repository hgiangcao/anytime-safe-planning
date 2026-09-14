"""exp4: paper-ready figures (PDF + PNG, single-column ~3.4 in, no titles).

  fig_validity    : synthetic time-uniform validity + the replanning loophole,
                    with the fair (episode-targeted) ACI arm
  fig_radius      : (a) radius vs horizon at MATCHED guarantee semantics,
                    (b) certifiable horizon vs calibration-set size,
                    (c) displayed mixture-CS finite-sample scaling
  fig_longrun     : long-horizon continuous deployment (120 windows, ~4 min):
                    certifiability, radius and throughput vs window index
  fig_calibration : empirical violation rate vs target delta (certified windows)
  fig_pareto      : safety-efficiency trade-off (success rate vs violations)
  fig_traj        : trajectory snapshots with certified inflated regions

Run: .venv/bin/python code/exp4_figures.py
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
import np2compat  # noqa: F401  (numpy>=2 pickles under numpy 1.x)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "results", "main")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.0,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.35,
})

MLAB = {"cs": "CS (ours)", "auc": "AUC (ours)", "aucb": "AUC-B (ours)",
        "union": "Union CP", "aci": "ACI", "aci_ep": "ACI-ep",
        "gauss": "Gauss CC", "uncal": "Uncal."}
MCOL = {"cs": "#d62728", "auc": "#ff7f0e", "aucb": "#8c564b",
        "union": "#1f77b4", "aci": "#2ca02c", "aci_ep": "#17becf",
        "gauss": "#9467bd", "uncal": "#7f7f7f"}
LONG = os.path.join(ROOT, "results", "long")
RMARK = {"replay": "o", "sf": "s", "orca": "^", "real": "D"}
DELTAS = [0.05, 0.1]


def load(regime, method, delta):
    with open(os.path.join(MAIN, f"{regime}_{method}_{delta}.json")) as f:
        return json.load(f)


def load_long(regime, method, delta):
    with open(os.path.join(LONG, f"{regime}_{method}_{delta}.json")) as f:
        return json.load(f)


REAL_SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def load_real_pooled(method, delta):
    """Pool the five leave-one-scene-out ETH/UCY folds into one summary."""
    tw = tu = tv = ne = nev = 0
    for sc in REAL_SCENES:
        d = load(f"real_{sc}", method, delta)
        recs = d["records"]
        tw += d["summary"]["total_windows"]
        tu += sum(r["n_uncert"] for r in recs)
        tv += sum(r["n_viol"] for r in recs)
        ne += len(recs)
        nev += sum(bool(r["episode_viol"]) for r in recs)
    tc = tw - tu
    return {"total_windows": tw, "cert_window_frac": tc / max(tw, 1),
            "window_viol_rate": (tv / tc) if tc else None, "n_viol": tv,
            "n_cert": tc, "n_episodes": ne, "episode_viol_rate": nev / max(ne, 1)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - hw, c + hw


def legend_below(ax, ncol=2, y=-0.30, fs=5.4, **kw):
    """Place the axis legend OUTSIDE the plot, centred below it."""
    opts = dict(handletextpad=0.4, columnspacing=1.0, labelspacing=0.25,
                borderpad=0.0, handlelength=1.6)
    opts.update(kw)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     frameon=False, fontsize=fs, **opts)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=250)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------- fig_calibration
def fig_calibration():
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.9))
    panels = [("window", ["cs", "aci", "gauss"], "per-window violation rate"),
              ("episode", ["union", "auc", "aucb", "aci_ep"],
               "per-episode violation rate")]
    regimes = ["replay", "sf", "orca", "real"]
    for ax, (lvl, methods, ylab) in zip(axes, panels):
        ax.plot([0.02, 0.13], [0.02, 0.13], "k--", lw=0.7, zorder=1,
                label="target ($y=\\delta$)")
        for mi, m in enumerate(methods):
            for ri, reg in enumerate(regimes):
                for d in DELTAS:
                    s = (load_real_pooled(m, d) if reg == "real"
                         else load(reg, m, d)["summary"])
                    if lvl == "window":
                        # certified windows only (a window with r=+inf carries no
                        # certificate and cannot be "violated")
                        n = round(s["total_windows"] * s["cert_window_frac"])
                        y = s["window_viol_rate"] or 0.0
                        k = round(y * n)
                    else:
                        n = s["n_episodes"]
                        k = round(s["episode_viol_rate"] * n)
                        y = s["episode_viol_rate"]
                    lo, hi = wilson(k, n)
                    x = d + (mi - (len(methods) - 1) / 2) * 0.006 + (ri - 1.5) * 0.0017
                    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], color=MCOL[m],
                                marker=RMARK[reg], ms=3, capsize=1.5, lw=0.7,
                                elinewidth=0.6, zorder=3)
        ax.set_ylabel(ylab)
        ax.set_xticks(DELTAS)
        ax.set_xlim(0.02, 0.13)
        ax.grid(True)
    axes[1].set_xlabel("target level $\\delta$")
    # legends: methods (colors) + regimes (markers)
    mh = [plt.Line2D([], [], color=MCOL[m], marker="s", ls="", ms=3.5, label=MLAB[m])
          for m in ["cs", "aci", "gauss", "union", "auc", "aucb", "aci_ep"]]
    rh = [plt.Line2D([], [], color="k", marker=RMARK[r], ls="", ms=3.5, label=r)
          for r in regimes]
    lh = plt.Line2D([], [], color="k", ls="--", lw=0.7, label="target $\\delta$")
    axes[0].legend(handles=mh[:3] + rh + [lh], ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, -0.16), frameon=False,
                   handletextpad=0.3, columnspacing=0.8, fontsize=5.4)
    axes[1].legend(handles=mh[3:], ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, -0.22), frameon=False,
                   handletextpad=0.3, columnspacing=0.8, fontsize=5.4)
    axes[0].set_ylim(0, 0.14)
    axes[1].set_ylim(0, 0.20)
    savefig(fig, "fig_calibration")


# ------------------------------------------------------------------ fig_radius
def fig_radius():
    """(a) radius vs horizon with the two guarantee families kept apart,
       (b) certifiable horizon vs calibration-set size."""
    df = pd.read_csv(os.path.join(ROOT, "results", "exp2_radius.csv"))
    hz = pd.read_csv(os.path.join(ROOT, "results", "exp2_horizon.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.55))
    SKIP_C = True
    d = 0.05

    # ---- (a) radius vs horizon
    ax = axes[0]
    u = df[(df.curve == "union") & (df.delta == d)].sort_values("x")
    fin = u[np.isfinite(u.radius)]
    Tblow = int(u[~np.isfinite(u.radius)].x.min())
    ax.plot(fin.x, fin.radius, color=MCOL["union"], label="Union CP ($T$ known)")
    ax.axvline(Tblow, color=MCOL["union"], ls=":", lw=0.8)
    ax.annotate(f"uncertifiable\n$T\\geq{Tblow}$", (Tblow * 1.2, 3.3), fontsize=5.5,
                color=MCOL["union"])
    for cur, col, ls, lab in [("auc", MCOL["auc"], "--", "AUC (ours)"),
                              ("aucb", MCOL["aucb"], "--", "AUC-B (ours)")]:
        a = df[(df.curve == cur) & (df.delta == d)].sort_values("x")
        a = a[np.isfinite(a.radius)]
        ax.plot(a.x, a.radius, color=col, ls=ls, marker=".", ms=2.0, label=lab)
        if len(a):  # mark where the certificate runs out
            ax.plot([a.x.iloc[-1]], [a.radius.iloc[-1]], marker="x", ms=4.5,
                    mew=1.1, color=col)
    g = df[(df.curve == "gauss") & (df.delta == d)].sort_values("x")
    ax.plot(g.x, g.radius, color=MCOL["gauss"], ls="-.", lw=0.8,
            label="Gauss CC (no guar.)")
    r_cs = float(df[(df.curve == "cs") & (df.delta == d)].radius.iloc[-1])
    r_pi = float(df[(df.curve == "plugin") & (df.delta == d)].radius.iloc[0])
    ax.axhline(r_cs, color=MCOL["cs"], lw=1.3, label="CS (ours)")
    ax.axhline(r_pi, color="k", ls=":", lw=0.8, label="plug-in $\\hat Q_{1-\\epsilon}$")
    ax.set_xscale("log")
    ax.set_xlabel("episode horizon / window index")
    ax.set_ylabel("inflation radius [m]")
    ax.set_ylim(0, 4.3)
    ax.grid(True)
    legend_below(ax, ncol=2, y=-0.34, fs=5.2)
    ax.text(0.97, 0.04, "solid/dashed: episode-level\nflat: per-window",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0,
            color="#444444")

    # ---- (b) certifiable horizon vs calibration-set size
    ax = axes[1]
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
    ax.text(2400, 2.6e3, "our $n$", fontsize=5.2, color="#666666")
    ax.text(0.03, 0.95, "CS (ours): unbounded", transform=ax.transAxes, fontsize=5.6,
            color=MCOL["cs"], va="top")
    ax.set_xlabel("calibration scores $n$")
    ax.set_ylabel("certifiable windows")
    ax.grid(True, which="both")
    legend_below(ax, ncol=1, y=-0.34, fs=4.8)

    # The former LIL overlay was removed: this uniform mixture has the slower
    # sqrt(log(t)/t) contraction rate.
    fig.tight_layout(w_pad=0.9)
    savefig(fig, "fig_radius")


# ----------------------------------------------------------------- fig_longrun
def fig_longrun(regime="long_replay", d=0.05):
    """Long-horizon continuous deployment (120 windows ~ 4 min per episode)."""
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.55))
    SKIP_C = True
    shown = ["cs", "aucb", "auc", "union"]
    ax = axes[0]
    for m in shown:
        pw = load_long(regime, m, d)["summary"]["per_window"]
        w = np.arange(1, len(pw["cert_frac"]) + 1)
        ax.plot(w, pw["cert_frac"], color=MCOL[m], label=MLAB[m], lw=1.1)
    ax.set_xlabel("window index within episode")
    ax.set_ylabel("fraction certified")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True)
    legend_below(ax, ncol=2, y=-0.34, fs=5.4)

    ax = axes[1]
    for m in shown:
        pw = load_long(regime, m, d)["summary"]["per_window"]
        w = np.arange(1, len(pw["mean_radius"]) + 1)
        r = np.array(pw["mean_radius"], dtype=float)
        ok = np.isfinite(r)
        ax.plot(w[ok], r[ok], color=MCOL[m], label=MLAB[m], lw=1.1)
    ax.set_xlabel("window index within episode")
    ax.set_ylabel("mean certified radius [m]")
    ax.grid(True)
    legend_below(ax, ncol=2, y=-0.34, fs=5.4)

    ax = axes[2]
    ms = ["cs", "aucb", "auc", "union", "aci", "aci_ep", "gauss", "uncal"]
    for m in ms:
        su = load_long(regime, m, d)["summary"]
        x = max(su["coll_step_frac"] * 1000.0, 0.05)
        y = su["mean_goals_per_episode"]
        c = su["cert_window_frac"]
        full = c > 0.99 and m != "uncal"
        ax.scatter(x, y, s=26, color=MCOL[m], zorder=3,
                   facecolor=MCOL[m] if full else "none",
                   edgecolor=MCOL[m], linewidths=1.0, label=MLAB[m])
        ax.annotate(f"{c:.2f}", (x, y), textcoords="offset points",
                    xytext=(4.5, -5.5 if m in ("aci", "aci_ep", "auc") else 1.5),
                    fontsize=4.6, color="#555555")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.set_xlim(1.1, 34)
    ax.set_ylim(-1.5, 41)
    ax.minorticks_off()
    ax.set_xlabel("collision-steps per 1000 steps")
    ax.set_ylabel("goals reached / episode")
    ax.grid(True)
    legend_below(ax, ncol=4, y=-0.34, fs=4.9, columnspacing=0.5,
                 handletextpad=0.15)
    fig.tight_layout(w_pad=1.2)
    savefig(fig, "fig_longrun")


# ------------------------------------------------------------------ fig_pareto
def fig_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    for ax, reg in zip(axes, ["sf", "orca"]):
        for m in ["cs", "auc", "aucb", "union", "aci", "aci_ep", "gauss", "uncal"]:
            s = load(reg, m, 0.05)["summary"]
            x = max(s["window_viol_rate"] or 0.0, 1.5e-3)
            y = s["success_rate"]
            ax.scatter(x, y, s=14, color=MCOL[m], zorder=3, label=MLAB[m])
        ax.set_xscale("log")
        ax.set_xlabel("per-window violation rate")
        ax.grid(True)
        ax.axvline(0.05, color="k", ls="--", lw=0.6)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=4, frameon=False, fontsize=5.2, handletextpad=0.2,
               columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    savefig(fig, "fig_pareto")


# -------------------------------------------------------------------- fig_traj
def fig_traj():
    import zlib
    from envloop import make_calibrator, run_episode, H
    from predictor import Predictor
    warm = np.load(os.path.join(ROOT, "results", "data", "cal_sf.npy"))
    pred = Predictor(os.path.join(ROOT, "results", "models", "gru_sf.pt"))
    cal = make_calibrator("cs", 0.05, warm,
                          cache_dir=os.path.join(ROOT, "results", "cache"))
    rng = np.random.default_rng(zlib.crc32(b"sf|3"))
    m = run_episode("sf", cal, "cs", pred, rng, record_traj=True)
    traj = m["traj"]
    snaps = [8, 22]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6))
    for ax, ts in zip(axes, snaps):
        fr = traj[min(ts, len(traj) - 1)]
        r = fr["r"]
        # robot path so far
        path = np.array([f["robot"] for f in traj[: ts + 1]])
        ax.plot(path[:, 0], path[:, 1], color="#d62728", lw=1.2)
        ax.plot(*fr["robot"], marker="o", color="#d62728", ms=5)
        ax.plot(0, -4.6, marker="*", color="#d62728", ms=8, mew=0)
        for i in range(fr["agents"].shape[0]):
            ax.plot(*fr["agents"][i], marker="o", color="#1f77b4", ms=3.5)
            pr = fr["preds"][:, i]  # [H,2]
            ax.plot(pr[:, 0], pr[:, 1], color="#1f77b4", lw=0.6, alpha=0.6)
            for j in [3, 7]:
                circ = plt.Circle(pr[j], r * (j + 1) / H, fill=True,
                                  color="#1f77b4", alpha=0.10, lw=0)
                ax.add_patch(circ)
                circ = plt.Circle(pr[j], r * (j + 1) / H, fill=False,
                                  color="#1f77b4", alpha=0.45, lw=0.5)
                ax.add_patch(circ)
        ax.set_xlim(-7.6, 7.6)
        ax.set_ylim(-5.6, 5.6)
        ax.set_aspect("equal")
        ax.text(0.02, 0.96, f"$t={ts * 0.25:.1f}$ s,  $r_t={r:.2f}$ m",
                transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(h_pad=0.4)
    savefig(fig, "fig_traj")


# ---------------------------------------------------------------- fig_validity
def fig_validity():
    with open(os.path.join(ROOT, "results", "exp1_validity.json")) as f:
        E = json.load(f)
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.15),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    ax = axes[0]
    for rec in E["cs_time_uniform"]:
        t, rate = np.array(rec["profile_t_rate"]).T
        lab = f"CS $\\beta$={rec['beta']}, $\\delta$={rec['delta']}"
        ax.plot(t, rate, label=lab, lw=1.0)
        ax.axhline(rec["delta"], color="k", ls=":", lw=0.5)
    for rec in E["plugin_time_uniform"]:
        if rec["beta"] == 0.9 and rec["delta_nominal"] == 0.05:
            ax.axhline(rec["anytime_undercoverage_rate"], color="#7f7f7f", ls="--",
                       lw=1.0)
            ax.text(12, rec["anytime_undercoverage_rate"] - 0.16,
                    "plug-in split-CP quantile,\nrecomputed each step", fontsize=6,
                    color="#444444")
    ax.set_xscale("log")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("P(any undercoverage by $t$)")
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    legend_below(ax, ncol=2, y=-0.30, fs=5.5)

    ax = axes[1]
    labels, vals, cols, targets = [], [], [], []
    for rec in E["episode_level"]:
        d = rec["delta"]
        for key, name, col in [("auc_episode_viol", "AUC", MCOL["auc"]),
                               ("aucb_episode_viol", "AUC-B", MCOL["aucb"]),
                               ("union_episode_viol", "Union", MCOL["union"]),
                               ("aci_ep_episode_viol", "ACI-ep", MCOL["aci_ep"]),
                               ("aci_episode_viol", "ACI", MCOL["aci"]),
                               ("cs_episode_viol", "CS", MCOL["cs"])]:
            labels.append(f"{name}\n$\\delta$={d}")
            vals.append(rec[key])
            cols.append(col)
            targets.append(d)
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=cols, width=0.72)
    for x, tg in zip(xs, targets):
        ax.plot([x - 0.4, x + 0.4], [tg, tg], "k--", lw=0.8)
    # the two per-window entries (ACI at its window tuning, and our own CS) sit at
    # the compounding ceiling by construction, not by failure
    for rec, off in zip(E["episode_level"], [0, 6]):
        c = rec["compounding_ceiling_1m(1md)^W"]
        ax.plot([off + 3.6, off + 5.4], [c, c], color="#888888", lw=1.2)
    ax.axvline(5.5, color="#cccccc", lw=0.8)
    short = [l.split("\n")[0] for l in labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(short, fontsize=5.4, rotation=38, ha="right")
    ax.set_xlim(-0.7, len(xs) - 0.3)
    ax.set_ylabel("episode viol. rate (i.i.d.)")
    ax.set_ylim(0, 0.66)
    for off, d in zip([2.5, 8.5], [0.05, 0.1]):
        ax.text(off, 0.62, f"$\\delta={d}$", ha="center", fontsize=6)
    ax.grid(True, axis="y")
    h = [plt.Line2D([], [], color="k", ls="--", lw=0.8, label="target $\\delta$"),
         plt.Line2D([], [], color="#888888", lw=1.2,
                    label="$1-(1-\\delta)^W$ ceiling (per-window methods)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              frameon=False, fontsize=5.4, handletextpad=0.4, columnspacing=1.0,
              handlelength=1.6)
    fig.tight_layout(h_pad=0.8)
    savefig(fig, "fig_validity")


if __name__ == "__main__":
    fig_validity()
    fig_radius()
    fig_longrun()
    fig_calibration()
    fig_pareto()
    fig_traj()
    print("exp4 complete")
