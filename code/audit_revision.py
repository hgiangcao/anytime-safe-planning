"""Focused acceptance audit for the corrected manuscript and current contract.

This reads checked-in artifacts only.  It verifies every quantitative value that
the revised paper highlights, required caveat language, and the new record
schema.  It does not rerun any planning experiment.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from csq import (QuantileCS, ViolationBudgetCS, cert_horizon_fixed_T,
                 cert_horizon_minimal, harmonic_spend)


ROOT = Path(__file__).resolve().parents[1]
passed = 0
failed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed
    if condition:
        passed += 1
        print(f"PASS {name}" + (f": {detail}" if detail else ""))
    else:
        failed.append(f"{name}: {detail}")
        print(f"FAIL {name}: {detail}")


def close(got: float, want: float, tol: float = 5e-4) -> bool:
    return math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol)


def main() -> None:
    exp1 = json.loads((ROOT / "results/exp1_validity.json").read_text())
    g1 = {(r["beta"], r["delta"]): r["anytime_undercoverage_rate"]
          for r in exp1["cs_time_uniform"]}
    check("G1 synthetic rates",
          all(close(g1[k], v, 5e-4) for k, v in {
              (0.9, 0.05): .0421, (0.9, .1): .0883,
              (.95, .05): .04395, (.95, .1): .08405}.items()), str(g1))
    plug = [r["anytime_undercoverage_rate"] for r in exp1["plugin_time_uniform"]]
    check("repeated marginal range", min(plug) >= .95 and max(plug) <= .97,
          f"range=[{min(plug):.4f},{max(plug):.4f}]")
    g3 = {r["delta"]: r for r in exp1["episode_level"]}
    check("G3 same-contract rates",
          close(g3[.05]["auc_episode_viol"], .0435)
          and close(g3[.05]["aucb_episode_viol"], .0179)
          and close(g3[.05]["union_episode_viol"], .04785)
          and close(g3[.1]["auc_episode_viol"], .0876)
          and close(g3[.1]["aucb_episode_viol"], .0313)
          and close(g3[.1]["union_episode_viol"], .0962), str(g3))

    exp2 = json.loads((ROOT / "results/exp2_headline.json").read_text())["delta_0.05"]
    exact = cert_horizon_minimal(2094, .05)
    check("fixed-pool wall", cert_horizon_fixed_T(2094, .05) == 104)
    check("exact harmonic cutoff", exact == 107
          and harmonic_spend(2094, exact) <= .05
          and harmonic_spend(2094, exact + 1) > .05,
          f"W={exact}")
    check("stored horizon values", exp2["cert_horizon_union_any_allocation"] == 104
          and exp2["cert_horizon_auc_wsq"] == 9
          and exp2["cert_horizon_optimal"] == 107)
    check("stored radius values", close(exp2["union_radius_T50"], 2.386, .001)
          and close(exp2["cs_radius_final_t~2100"], 1.721, .001)
          and close(exp2["split_quantile_1-delta"], 1.616, .001))

    dep = json.loads((ROOT / "results/dependence_aware_long_real.json").read_text())
    dm = dep["conditions"]["long_real_cs_0.05"]["metrics"][
        "certified_window_violation_rate"]
    check("dependence-aware estimate", close(dm["estimate"], .025, 1e-12))
    check("episode-cluster interval",
          np.allclose(dm["episode_cluster_bootstrap_ci95"],
                      [.0047619048, .0523809524]))
    check("scene-cluster interval",
          np.allclose(dm["scene_cluster_bootstrap_ci95"],
                      [.0009803922, .0757575758]))
    check("leave-one-scene sensitivity",
          np.allclose(dm["leave_one_scene_out_range"],
                      [.0141025641, .0388888889]))

    shift = json.loads((ROOT / "results/exp12_rho_loso.json").read_text())
    misses = [r for r in shift["folds"] if not r["point_transfer_covered"]]
    check("shift is not theorem-certified",
          shift["formal_conditional_shift_budget_certified"] is False)
    check("held-out point transfer", shift["point_transfers_covered"] == 9
          and shift["n_total"] == 10 and len(misses) == 1
          and misses[0]["held_out"] == "real_zara1"
          and close(misses[0]["delta"], .1, 1e-12), str(misses))

    exp7 = json.loads((ROOT / "results/exp7_budget.json").read_text())
    dep7 = exp7["deployment"]
    check("G2/G3 artifact labels are semantic",
          dep7["long_replay|cs|0.05"]["G2_contract_under_stated_iid_assumptions"]
          and not dep7["long_replay|cs|0.05"][
              "G3_contract_under_stated_exchangeability_assumptions"]
          and dep7["long_replay|auc|0.05"][
              "G3_contract_under_stated_exchangeability_assumptions"]
          and not dep7["long_replay|aci|0.05"][
              "G3_contract_under_stated_exchangeability_assumptions"]
          and all("has_G3_statement_throughout" not in r for r in dep7.values()))
    summary = json.loads((ROOT / "results/summary.json").read_text())
    check("summary uses G2 count terminology",
          "violation_budget_G2" in summary["headline"]
          and "violation_budget_G3" not in summary["headline"])
    exp9 = json.loads((ROOT / "results/exp9_matched.json").read_text())
    check("allocation artifact disclaims global optimum",
          "not certified global optima" in exp9["prop1_test"]["method_note"])

    smoke = json.loads((ROOT / "results/window_anchor_smoke.json").read_text())
    rec = smoke["record"]
    fields = (
        "predictor_update_steps", "constraint_anchor_steps", "fallback_labels",
        "collision_step_labels", "anchored_constraint_feasible_labels",
        "tracking_bound_labels", "containment_labels_by_control_step",
        "selected_agent_endpoint_clearance_implication_labels",
    )
    check("current execution contract", rec["execution_contract"] == "window_anchor_v1")
    check("complete per-step labels", all(len(rec[k]) == rec["steps"] for k in fields))
    check("exact smoke provenance", smoke["purpose"] ==
          "contract_smoke_not_performance_evidence"
          and bool(smoke["started_utc"]) and bool(smoke["completed_utc"])
          and len(smoke["source_sha256"]) == 6)
    check("selected-agent scope explicit",
          all(len(w["score_selected_agent_indices"]) == w["clean_agent_count"]
              and isinstance(w["all_window_agents_in_score_event"], bool)
              for w in rec["window_audit"])
          and "discrete_clearance_implication_labels" not in rec)

    fresh = json.loads((ROOT / "results/window_anchor_v1_study/summary.json").read_text())
    ledger = json.loads((ROOT / "results/window_anchor_v1_study/ledger.json").read_text())
    same = json.loads((ROOT / "results/window_anchor_v1_study/same_frozen_stream.json").read_text())
    check("fresh anchored matrix complete",
          fresh["n_predeclared_attempts"] == fresh["n_completed_attempts"] == 320
          and fresh["n_failed_executions_retained"] == 0
          and fresh["n_interrupted_executions_retained"] == 0
          and all(a["status"] == "completed" for a in ledger["attempts"]))
    check("fresh matrix has eight 40-episode cells",
          len(fresh["cells"]) == 8
          and all(c["n_episodes"] == 40 and c["n_attempted_windows"] == 320
                  for c in fresh["cells"].values()))
    fallback = [c["active_window_fallback_fraction"]
                for c in fresh["cells"].values()]
    collision = [c["observed_collision_episode_fraction"]
                 for c in fresh["cells"].values()]
    selected = [c["selected_agent_fraction_over_attempted_window_agents"]
                for c in fresh["cells"].values()]
    check("negative controller outcomes retained",
          close(min(fallback), .566796875, 1e-12)
          and close(max(fallback), .662109375, 1e-12)
          and close(min(collision), .025, 1e-12)
          and close(max(collision), .175, 1e-12)
          and min(selected) < .80 and max(selected) < .81)
    check("bounded tracking stress respected",
          all(c["tracking_bound_violation_count"] == 0
              and c["tracking_error_max_m"] <= .1000000001
              for c in fresh["cells"].values()))
    check("literal same frozen stream present",
          all(block["n_scores"] == 100 and len(block["rows"]) == 100
              for block in same["regimes"].values())
          and same["purpose"] ==
          "literal_same_score_stream_same_G1_radius_comparison")
    qgrow = QuantileCS(.9, .1, [-1., 0., 1.], t_max=2)
    qgrow.radius()
    bgrow = ViolationBudgetCS(.05, .05, w_max=2)
    for _ in range(3):
        bgrow.add(False)
    check("initial capacities are not horizons",
          len(qgrow._u) >= qgrow.t and len(bgrow._u) >= bgrow.W)

    tex = (ROOT / "paper/root.tex").read_text()
    required = (
        "not an iterated-logarithm rate",
        "exact integer cutoff",
        "not an impossibility theorem for all episode certificates",
        "fixed horizon covers every prefix",
        "cross-contract",
        "legacy score streams",
        "continuous-time collision avoidance",
        "externally justified conditional-probability budget",
        "same-g1 comparator",
        "score-value-independent",
        "current-observation speed envelope",
        "no 320-run portfolio guarantee",
        "W_{\\mathrm{AUC}}",
        "slot-generation UIDs",
        "double on demand",
        "not claimed globally optimal",
    )
    check("required caveats present", all(x.lower() in tex.lower() for x in required))

    # Round-4 item 20.P1.1: the CS-R closed-loop replay. Every number the manuscript prints
    # about it must equal results/closedloop_shift/summary.json, and the manuscript must not
    # regress to claiming CS-R was only ever exercised on legacy streams.
    cl_path = ROOT / "results/closedloop_shift/summary.json"
    check("closed-loop shift study present", cl_path.exists())
    if cl_path.exists():
        cl = json.loads(cl_path.read_text())
        g = {round(r["rho"], 3): r for r in cl["grid_totals"]}
        v0 = 100 * g[0.0]["violation_frac_of_certified"]
        v5 = 100 * g[5.0]["violation_frac_of_certified"]
        d5 = g[5.0]["eff_delta"]
        check("closed-loop rho=0 rate matches text",
              f"{v0:.1f}\\%" in tex, f"{v0:.1f}%")
        check("closed-loop rho=5 rate matches text",
              f"{v5:.1f}\\%" in tex, f"{v5:.1f}%")
        check("closed-loop budget ratio matches text",
              f"{round(0.1 / d5):d}" in tex.replace(",", ""), f"{0.1/d5:.0f}x")
        check("closed-loop attempt split matches text",
              all(f"${a}/{b}$" in tex or f"{a}/{b}" in tex for a, b in
                  ((cl["n_clean_at_rho0"], cl["n_scored_attempts"]),)),
              f"{cl['n_clean_at_rho0']}/{cl['n_scored_attempts']}")
        check("closed-loop unfixable count in text", str(cl["n_unfixable_on_grid"]) in tex)
        check("CS-R no longer claimed legacy-only",
              "closed-loop distribution, which is where it matters" in tex.lower()
              or "closed-loop scores" in tex.lower())
    check("no continuous collision overclaim", "collision-free" not in tex.lower())
    check("no legacy throughput claim",
          "17.8 goals" not in tex.lower() and "2.5x" not in tex.lower())
    check("AUC cutoff has a distinct symbol",
          "W_{\\mathrm{AUC}}" in tex
          and "AUC schedule" in tex
          and "ten calibrators" not in tex)
    check("fresh controller evidence replaces future-study claim",
          "320-episode anchored study" in tex
          and "full performance rerun remains future evidence" not in tex
          and "not a deployment-safety claim" in tex)
    table_rows = (
        "Social Force & mixture & 0.00 & 7.2 & 58.8 & 79.5 & 29.1 & 10.0 & 21.4",
        "Social Force & spending & 0.10 & 6.2 & 63.9 & 79.5 & 25.8 & 17.5 & 20.3",
        "ORCA-lite & mixture & 0.00 & 17.8 & 57.5 & 79.7 & 27.7 & 7.5 & 21.5",
        "ORCA-lite & spending & 0.10 & 17.5 & 63.9 & 80.2 & 24.5 & 2.5 & 21.1",
    )
    check("fresh table values bind summary", all(row in tex for row in table_rows))

    # Round-5 item 20.P0.2: the coverage contracts and the operational outcome must be
    # readable in one table, and every value in it must come from stored results rather
    # than from a hand-typed edit.  make_main_table.py regenerates the block; --check
    # fails if root.tex has drifted from what the JSON artifacts imply.
    import make_main_table as mmt
    check("main table generated from stored results",
          mmt.current_block(tex) == mmt.render() if mmt.BEGIN in tex else False)
    cov = {ctor: (a, b, ok) for _, ctor, a, b, ok in mmt.coverage_rows()}
    g1 = f"{100 * exp1['cs_time_uniform'][0]['anytime_undercoverage_rate']:.1f}"
    g3 = f"{100 * exp1['episode_level'][0]['aucb_episode_viol']:.1f}"
    g2 = f"{100 * exp7['synthetic'][0]['anytime_exceed_rate']:.1f}"
    check("coverage panel prints G1/G2/G3 next to the operational rows",
          all(v in tex for v in (g1, g2, g3))
          and "Certified statistical layer" in tex
          and "Uncertified operational layer" in tex,
          f"G1 {g1}, G2 {g2}, G3 {g3}")
    plug = cov["plug-in split-CP, $\\beta{=}0.9$"]
    check("coverage panel keeps the plug-in negative control",
          plug[:2] == (f"{100 * exp1['plugin_time_uniform'][0]['anytime_undercoverage_rate']:.1f}",
                       f"{100 * exp1['plugin_time_uniform'][1]['anytime_undercoverage_rate']:.1f}")
          and plug[2] == "\\textbf{no}"
          and all(cov[c][2] == "yes" for c in cov if not c.startswith("plug-in")),
          f"plug-in {plug[0]}/{plug[1]}")
    check("table states that validity is not collision safety",
          "statistical validity is not collision safety" in tex)

    # Round-5 item 20.P0.1: no "safe planning" framing survives outside an explicit denial.
    banned = ("a safe planner", "safety-critical calibration",
              "Certified receding-horizon planning", "safe planning")
    check("no safe-planning framing", not any(b.lower() in tex.lower() for b in banned),
          ", ".join(b for b in banned if b.lower() in tex.lower()))
    check("paper names the certified-occupancy-tube framing",
          tex.lower().count("planning with certified occupancy-tube contracts") >= 2)

    # Round-5 item 20.P1: the clearance conjunction and the universal/family-specific split.
    check("clearance conjunction schematic present",
          "discrete endpoint clearance} only" in tex
          and "no\\\\fallback" in tex and "selected\\\\clean UID" in tex)
    check("rank-horizon scope stated explicitly",
          "Universal vs.\\ family-specific" in tex
          and "is not bound by the harmonic cutoff" in tex)

    # Round-6 item 20.P1.1: every reviewer asked what fraction of the observed collisions
    # happened under fallback rather than under an anchored control.  The breakdown is a
    # regrouping of the same 320 frozen records the main table uses; collision_attribution
    # --check re-derives it and fails if either the stored JSON or root.tex has drifted.
    import collision_attribution as ca
    _ca = ca.compute()
    check("collision attribution recomputes from the frozen records",
          json.loads(ca.OUT.read_text())["fallback_collision_controls"]
          == _ca["fallback_collision_controls"] if ca.OUT.exists() else False)
    check("collision attribution printed beside the operational rows",
          all(needle in tex for _, needle in ca.paper_numbers(_ca))
          and "without attributing cause" in tex,
          f"fallback {_ca['fallback_collision_controls']}/{_ca['colliding_controls']}, "
          f"anchored {_ca['anchored_collision_controls']}/{_ca['colliding_controls']}")
    response = (ROOT / "EXTERNAL_REVIEW_RESPONSE.md").read_text()
    check("revised review response mapped",
          "2026-09-08 revised-review closure" in response
          and "Nothing here promises acceptance" in response
          and "23/23" in response)
    stale_docs = ((ROOT / "code/csq.py").read_text()
                  + (ROOT / "code/exp5_longrun.py").read_text())
    check("replay dependence caveat in released code",
          "clean, exchangeable" not in stale_docs
          and "episodes are i.i.d." not in stale_docs
          and "replay format alone\n    does not establish it" in stale_docs)

    print(f"\n{passed} acceptance checks passed; {len(failed)} failed")
    for item in failed:
        print(" -", item)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
