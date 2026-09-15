"""Aggregate results/ into results/summary.json (headline numbers, per-experiment
records, caveats).  Run after exp1-exp4."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa: F401  (numpy>=2 pickles under numpy 1.x)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def jload(p):
    with open(p) as f:
        return json.load(f)


exp1 = jload(os.path.join(R, "exp1_validity.json"))
exp2 = jload(os.path.join(R, "exp2_headline.json"))
meta = jload(os.path.join(R, "data", "meta.json"))

METHODS = ["cs", "auc", "aucb", "union", "aci", "aci_ep", "gauss", "uncal"]


def _parse(fn):
    """split '<regime>_<method>_<delta>.json'; method names contain '_'."""
    stem = fn[:-5]
    for m in sorted(METHODS, key=len, reverse=True):
        tag = f"_{m}_"
        if tag in stem:
            reg, d = stem.split(tag)
            return reg, m, float(d)
    raise ValueError(fn)


main, longr = {}, {}
for fn in sorted(os.listdir(os.path.join(R, "main"))):
    if fn.endswith(".json"):
        main[_parse(fn)] = jload(os.path.join(R, "main", fn))["summary"]
LONGD = os.path.join(R, "long")
if os.path.isdir(LONGD):
    for fn in sorted(os.listdir(LONGD)):
        if fn.endswith(".json"):
            s_ = jload(os.path.join(LONGD, fn))["summary"]
            s_.pop("per_window", None)
            longr[_parse(fn)] = s_


def g(reg, meth, d, key):
    return main[(reg, meth, d)][key]


def gl(reg, meth, d, key):
    return longr[(reg, meth, d)][key]


REAL_SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def real_pool(meth, d):
    """Pool the five leave-one-scene-out folds (windows and episodes)."""
    tw = tu = tv = ne = nev = nc = nsu = 0
    for sc in REAL_SCENES:
        fp = os.path.join(R, "main", f"real_{sc}_{meth}_{d}.json")
        if not os.path.exists(fp):
            return None
        rec = jload(fp)
        recs = rec["records"]
        tw += rec["summary"]["total_windows"]
        tu += sum(r["n_uncert"] for r in recs)
        tv += sum(r["n_viol"] for r in recs)
        ne += len(recs)
        nev += sum(bool(r["episode_viol"]) for r in recs)
        nc += sum(bool(r["collision"]) for r in recs)
        nsu += sum(bool(r["reached"]) for r in recs)
    tc = tw - tu
    return {"n_episodes": ne, "cert_window_frac": round(tc / max(tw, 1), 3),
            "window_viol_rate": round(tv / tc, 4) if tc else None,
            "episode_viol_rate": round(nev / max(ne, 1), 4),
            "success_rate": round(nsu / max(ne, 1), 3),
            "collision_rate": round(nc / max(ne, 1), 3)}


cs_anytime = {f"beta{r['beta']}_delta{r['delta']}": r["anytime_undercoverage_rate"]
              for r in exp1["cs_time_uniform"]}
plugin = {f"beta{r['beta']}": r["anytime_undercoverage_rate"]
          for r in exp1["plugin_time_uniform"]}

headline = {
    "theorem1_synthetic_anytime_undercoverage_rate_vs_delta": cs_anytime,
    "naive_plugin_conformal_anytime_undercoverage_rate": plugin,
    "aci_episode_violation_iid_despite_perfect_avg_coverage": {
        f"delta{r['delta']}": r["aci_episode_viol"] for r in exp1["episode_level"]},
    "radius_separation_delta0.05": exp2["delta_0.05"],
    "radius_separation_delta0.1": exp2["delta_0.1"],
    "long_horizon_120_windows_delta0.05": {
        f"{reg}/{m}": {"cert_window_frac": round(gl(reg, m, 0.05, "cert_window_frac"), 3),
                       "median_first_uncertified_window": gl(reg, m, 0.05, "first_uncertified_window_median"),
                       "window_viol_rate": gl(reg, m, 0.05, "window_viol_rate"),
                       "goals_per_episode": round(gl(reg, m, 0.05, "mean_goals_per_episode"), 2),
                       "collision_step_frac": round(gl(reg, m, 0.05, "coll_step_frac"), 5)}
        for reg in ["long_replay", "long_sf"] for m in METHODS},
    "replay_calibration": {
        "cs_window_viol_rate_target0.05": g("replay", "cs", 0.05, "window_viol_rate"),
        "cs_window_viol_rate_target0.1": g("replay", "cs", 0.1, "window_viol_rate"),
        "union_episode_viol_rate_target0.05": g("replay", "union", 0.05, "episode_viol_rate"),
        "auc_episode_viol_rate_target0.05": g("replay", "auc", 0.05, "episode_viol_rate"),
        "gauss_window_viol_rate_target0.1": g("replay", "gauss", 0.1, "window_viol_rate")},
    "closed_loop_stress_window_viol_cs": {
        "sf_target0.05": g("sf", "cs", 0.05, "window_viol_rate"),
        "sf_target0.1": g("sf", "cs", 0.1, "window_viol_rate"),
        "orca_target0.05": g("orca", "cs", 0.05, "window_viol_rate"),
        "orca_target0.1": g("orca", "cs", 0.1, "window_viol_rate")},
    "efficiency_sf_delta0.05": {
        m: {"success_rate": g("sf", m, 0.05, "success_rate"),
            "mean_task_time_s": g("sf", m, 0.05, "mean_task_time_s"),
            "fallback_step_frac": g("sf", m, 0.05, "fallback_step_frac"),
            "mean_radius_m": g("sf", m, 0.05, "mean_radius"),
            "collision_rate": g("sf", m, 0.05, "collision_rate")}
        for m in METHODS},
    "real_ethucy_leave_one_scene_out": {
        "folds": REAL_SCENES,
        "n_cal_scores_per_fold": {sc: meta["real"][sc]["n_cal_scores"] for sc in REAL_SCENES}
                                  if isinstance(meta.get("real"), dict) and REAL_SCENES[0] in meta.get("real", {}) else None,
        **{f"{m}_delta{d}": real_pool(m, d) for m in METHODS for d in (0.05, 0.1)}},
    "long_real_60_windows": {
        f"{m}_delta{d}": {
            "cert_window_frac": round(longr[("long_real", m, d)]["cert_window_frac"], 3),
            "median_first_uncertified_window": longr[("long_real", m, d)]["first_uncertified_window_median"],
            "window_viol_rate": longr[("long_real", m, d)]["window_viol_rate"],
            "goals_per_episode": round(longr[("long_real", m, d)]["mean_goals_per_episode"], 2),
            "collision_step_frac": round(longr[("long_real", m, d)]["coll_step_frac"], 5)}
        for m in METHODS for d in (0.05, 0.1)},
    "ablations": jload(os.path.join(R, "exp6_ablations.json")) if os.path.exists(os.path.join(R, "exp6_ablations.json")) else None,
    "eth_single_scene_smallN": {
        "n_cal_scores": meta["eth"].get("n_cal_scores"),
        **{f"cert_window_frac_{m}_{d}": round(g("eth", m, d, "cert_window_frac"), 3)
           for m in METHODS for d in (0.05, 0.1)},
        "cs_delta0.1_window_viol": g("eth", "cs", 0.1, "window_viol_rate"),
        "cs_delta0.1_success": g("eth", "cs", 0.1, "success_rate")},
}

experiments = [
    {"name": "test_core", "config": "unit tests of e-value/CS/CP/ACI/certifiable-horizon/planner core",
     "files": ["code/test_core.py"], "key_numbers": "all pass"},
    {"name": "exp1_validity", "config": exp1["config"],
     "n_runs": exp1["config"]["n_runs"],
     "key_numbers": {"cs_anytime_undercoverage": cs_anytime, "plugin_loophole": plugin,
                     "episode_level_iid": exp1["episode_level"]},
     "files": ["results/exp1_validity.json", "results/figures/fig_validity.pdf"]},
    {"name": "exp2_radius", "config": "real sf calibration scores n=2094 + synthetic Rayleigh t=1e5",
     "key_numbers": exp2,
     "files": ["results/exp2_radius.csv", "results/figures/fig_radius.pdf"]},
    {"name": "exp3_planning",
     "config": {"regimes": ["replay", "sf", "orca", "eth"],
                "methods": METHODS,
                "delta": [0.05, 0.1], "episodes_per_condition": {"default": 200, "eth": 11},
                "windows_per_episode_max": 8, "dt_s": 0.25, "horizon_steps": 8,
                "agents": "5-11", "seeds": "crc32(regime|episode), paired across methods"},
     "n_episodes_total": sum(s["n_episodes"] for s in main.values()),
     "key_numbers": {f"{k[0]}/{k[1]}/{k[2]}": {
         "cert_window_frac": round(s["cert_window_frac"], 3),
         "window_viol": None if s["window_viol_rate"] is None else round(s["window_viol_rate"], 4),
         "episode_viol": round(s["episode_viol_rate"], 4),
         "collision": round(s["collision_rate"], 3),
         "fallback_frac": round(s["fallback_step_frac"], 3),
         "success": round(s["success_rate"], 3)} for k, s in main.items()},
     "files": ["results/main/", "results/figures/fig_calibration.pdf",
               "results/figures/fig_pareto.pdf", "results/figures/fig_traj.pdf"]},
    {"name": "exp6_ablations",
     "config": "sf, delta=0.05, 120 episodes: CV-only vs CV+GRU predictor; 50 vs 162 planner candidates",
     "key_numbers": jload(os.path.join(R, "exp6_ablations.json")) if os.path.exists(os.path.join(R, "exp6_ablations.json")) else None,
     "files": ["results/exp6_ablations.json"]},
    {"name": "exp5_longrun",
     "config": {"regimes": ["long_replay", "long_sf", "long_real"], "methods": METHODS,
                "delta": [0.05, 0.1],
                "episodes_per_condition": {"long_replay": 30, "long_sf": 30, "long_real": 14},
                "windows_per_episode": {"sim": 120, "long_real": 60}, "task": "patrol (goal flips on arrival)",
                "calibrator": "fresh per episode from the same n=2094 sf pool"},
     "n_episodes_total": sum(s["n_episodes"] for s in longr.values()),
     "key_numbers": {f"{k[0]}/{k[1]}/{k[2]}": {
         "cert_window_frac": round(s["cert_window_frac"], 3),
         "median_first_uncertified_window": s["first_uncertified_window_median"],
         "window_viol": None if s["window_viol_rate"] is None else round(s["window_viol_rate"], 4),
         "goals_per_episode": round(s["mean_goals_per_episode"], 2),
         "collision_step_frac": round(s["coll_step_frac"], 5),
         "fallback_frac": round(s["fallback_step_frac"], 3)} for k, s in longr.items()},
     "files": ["results/long/", "results/figures/fig_longrun.pdf"]},
]

caveats = [
    "All baselines and simulators are compact reimplementations in this codebase (Lindemann-style union-bound CP, ACI planner, social-force, ORCA-style reciprocal avoidance), not the original authors' code.",
    "Guarantee semantics differ by construction and every figure/table separates them: CS certifies per-window risk <= eps simultaneously for ALL windows (with CS confidence delta); union CP, AUC and AUC-B certify per-episode any-violation <= delta. CS is not expected to satisfy the episode-level bound; its episode rate sits at the compounding ceiling 1-(1-delta)^W, and so does a window-targeted ACI's -- that number is arithmetic, not a discovered failure, which is why an episode-targeted ACI (alpha = delta/W) is included as the fair episode-level ACI arm.",
    "Certifiability is reported separately from violation: 'cert_window_frac' is the fraction of scored windows with a FINITE radius, and legacy window violation rates are computed over finite-radius windows only. A radius of +inf is statistically valid but operationally uninformative; it triggers a separately labelled uncertified fallback.",
    "The iid premise is exact only in synthetic tests (exp1). Replay windows within a trajectory are dependent, and replay format alone does not establish marginal exchangeability with the calibration pool. Replay rates are descriptive and long-trajectory uncertainty clusters complete episodes and source scenes. Closed-loop sf/orca additionally have interaction-induced shift.",
    "Archived planning records used a legacy fresh-prediction recentering contract. They are retained for provenance, but their throughput/collision fields do not validate the corrected window_anchor_v1 controller. New records anchor one prediction at the score-window start; score-selected identities, containment, tracking status, selected-agent endpoint-clearance implication, fallback, and physical collision are distinct labels.",
    "Robust e-value mixing is implemented, but its theorem requires an externally justified conditional pathwise shift budget. Recorded marginal frequencies do not certify that budget. Other scope omissions include the adversarially robust CP baseline, SIPP, and delta=0.01.",
    "ETH regime is real data but tiny (11 replay chunks, 72 calibration scores from the fetched biwi_eth.txt); treated as a small pilot, not a headline. It is, however, exactly the small-n regime in which the certifiable-horizon results bite.",
    "200 episodes per condition for the 8-window experiment and 30 per condition for the 120-window experiment. At n=200 a per-episode rate has a ~+/-0.03 95% CI, at n=30 a ~+/-0.15 CI; Wilson intervals are stored with every rate in results/main and results/long.",
]

summary = {"headline": headline, "experiments": experiments, "caveats": caveats,
           "data_meta": meta}
with open(os.path.join(R, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("wrote results/summary.json")
