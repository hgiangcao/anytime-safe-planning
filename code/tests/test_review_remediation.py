"""Regression checks for the 2026-09-07 review remediation.

Historical checks read frozen artifacts; focused round-two regressions use only
small deterministic simulations and never launch the prospective matrix.
"""
from pathlib import Path
import json
import sys

import pytest
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
import audit_frozen_evidence as audit  # noqa: E402
import review_round2 as rr2  # noqa: E402
import make_review_round2_paper as rr2paper  # noqa: E402
from mapf import flow, maps, sim  # noqa: E402
from mapf.pibt import dist_table  # noqa: E402


def test_active_frozen_ledger_is_complete_and_unchanged():
    records, counts = audit.load_raw()
    assert counts == audit.EXPECTED_GROUP_COUNTS
    assert len(records) == 1525
    assert {int(rec["steps"]) for _, rec in records} == {1000}
    assert all(path.name == audit.expected_raw_name(rec["cond"])
               for path, rec in records)


def test_posthoc_tail_audit_covers_every_headline_cell_without_overclaim():
    records, _ = audit.load_raw()
    report = audit.headline_tail_audit(records)
    assert len(report["cells"]) == 15
    assert all(set(methods) == set(audit.HEADLINE_METHODS)
               for methods in report["cells"].values())
    assert report["aggregate"]["runs"] == 600
    assert report["aggregate"]["terminal_zero_100_step_events"] == 0
    assert report["aggregate"]["min_last_half_over_overall"] == pytest.approx(
        0.5049, abs=1e-6)
    assert "not substantially longer-horizon" in report["interpretation_limit"]


def test_released_headline_and_compute_numbers_match_raw_json():
    records, _ = audit.load_raw()
    report = audit.summary_crosscheck(records)
    assert report["status"] == "exact_match_to_raw_json"
    assert report["headline_cells_checked"] == 60
    assert report["headline_fields_checked"] == 180
    assert report["compute_cells_checked"] == 8
    assert report["compute_fields_checked"] == 24


def test_search_reclassification_exposes_unretained_start_and_terminal_mean():
    records, _ = audit.load_raw()
    report = audit.search_audit(records)
    assert report["current_source_retains_new_starting_candidates"] is True
    fullwarm = report["runs"]["fullwarm"]
    assert fullwarm["starting_field_exact_after_parameterization"] is True
    assert fullwarm["starting_candidate_evaluated_before_sampling"] is False
    assert fullwarm["starting_candidate_retained_as_incumbent"] is False
    assert fullwarm["terminal_mean_performance"] == "not_evaluated_in_frozen_artifacts"
    assert fullwarm["starting_candidate_heldout_proxy"]["mean"] == pytest.approx(15.1439)
    assert fullwarm["selected_heldout_throughput"]["mean"] == pytest.approx(14.4765)


def test_manuscript_keeps_theory_and_deployment_boundaries_explicit():
    text = (PROJECT / "paper" / "root.tex").read_text()
    required = (
        "b_u+b_v",
        r"Bh=2(\mathbf 1_u-\mathbf 1_v)\ne0",
        "network optimality theorem",
        "a guarantee about discrete PIBT",
        "incumbent fitness to infinity",
        "Thus even 5,000 steps do not establish steady state",
        "execution pauses for a cold solve",
        r"(w,\varepsilon)\mapsto(kw,k\varepsilon)",
        "aggregate fixed-gradient routing-cost residual",
        "every positive-flow path is exactly shortest",
        "A screen from the map alone",
        "a registered prediction, not a validated classifier",
    )
    for phrase in required:
        assert phrase in text
    forbidden = (
        "exact threshold at which head-on pricing makes one-way lanes system-optimal",
        "the same condition delimits the program's convexity domain",
        "the only method that holds up under task-distribution shift",
        "tie tolerance is a correctness condition",
        "seeded at the toll field \\emph{exactly}, it still ends below it",
    )
    for phrase in forbidden:
        assert phrase not in text


def test_nonzero_fw_gap_has_the_claimed_aggregate_cost_meaning():
    m = maps.get_map("empty-32x32")
    net = flow.FlowNet(m, gamma=0.0, mu=1.0)
    dests, demand = flow.uniform_demand(net, lam=1.0, K=8, seed=17)
    sol = net.solve(dests, demand, max_iter=4, tol=0.0)
    r = net.linearization_residual(sol["f"], dests, demand)
    assert sol["gap"] == pytest.approx(r["relative_gap"])
    assert r["absolute_residual"] >= 0
    assert r["returned_linear_cost"] + 1e-10 >= r["minimum_linear_cost"]
    if r["relative_gap"] < 1:
        rhs = r["minimum_linear_cost"] / (1 - r["relative_gap"])
        assert r["returned_linear_cost"] <= rhs * (1 + 1e-10) + 1e-10


def test_warehouse_resolution_parameter_changes_only_pickup_count():
    m = maps.get_map("warehouse-10-20-10-2-1")
    net = flow.FlowNet(m)
    n_stations = len(m["meta"]["stations"])
    for k in (32, 64, 128):
        dests, demand = flow.make_demand(net, lam=3.0, K=k, seed=0)
        assert len(dests) == n_stations + k
        assert demand.sum() == pytest.approx(3.0)


def test_locked_round_two_corpus_and_paper_exports_preserve_negative_results():
    out = PROJECT / "results" / "review_round2"
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert summary["complete"] is True
    assert summary["observed_counts"] == {
        "attribution": 30,
        "independent_planner": 9,
        "long_horizon": 27,
        "search_test": 6,
        "stale": 9,
        "warehouse_resolution": 8,
    }
    assert manifest["status"] == "PASS"
    assert manifest["n_trace_records"] == 89
    assert len(manifest["artifacts"]) == 198
    assert summary["search"]["initial_candidate_evaluated_and_retained"] is True
    assert summary["search"]["validation_selected_index"] == 0
    cells = summary["families"]["long_horizon"]["by_map_label"]
    assert cells["empty-32-32::tolls"]["throughput_mean"] > cells[
        "empty-32-32::unweighted"]["throughput_mean"]
    assert cells["maze-32-32-4::tolls"]["throughput_mean"] < cells[
        "maze-32-32-4::unweighted"]["throughput_mean"]
    assert cells["warehouse-10-20-10-2-1::tolls"]["throughput_mean"] < cells[
        "warehouse-10-20-10-2-1::unweighted"]["throughput_mean"]
    macros, table = rr2paper.render()
    assert macros == (PROJECT / "paper" /
                      "review_round2_numbers.tex").read_text()
    assert table == (PROJECT / "paper" /
                     "review_round2_table.tex").read_text()


def test_full_trace_contract_matches_completion_events():
    m = maps.get_map("empty-32x32")
    r = sim.run_lifelong(m, 20, 80, seed=190, record_full=True)
    assert r["position_trace"].shape == (81, 20)
    assert r["goal_trace"].shape == (81, 20)
    assert r["goal_events"].shape[1] == 7
    assert len(r["goal_events"]) == r["completed"]
    assert (r["goal_events"][:, 3] > 0).all()


def test_joint_weight_and_tie_scaling_preserves_realized_trajectory():
    m = maps.get_map("empty-32x32")
    rng = np.random.default_rng(20260908)
    edges, _ = maps.directed_edges(m)
    weights = 0.75 + rng.random(len(edges))
    scale = 4.25
    d1 = dist_table(m, weights)
    d2 = dist_table(m, scale * weights)
    a = sim.run_lifelong(m, 35, 120, seed=191, D=d1,
                         eps_tie=0.6, record_full=True)
    b = sim.run_lifelong(m, 35, 120, seed=191, D=d2,
                         eps_tie=scale * 0.6, record_full=True)
    assert np.array_equal(a["position_trace"], b["position_trace"])
    assert np.array_equal(a["goal_trace"], b["goal_trace"])
    assert np.array_equal(a["goal_events"], b["goal_events"])


def test_round2_protocol_is_locked_to_disjoint_bounded_families():
    p = rr2.protocol_object()
    assert p["common"]["workers"] == 1
    assert p["search"]["max_training_candidates_including_initial"] == 49
    assert p["long_horizon"]["steps"] == 5000
    assert p["warehouse_resolution"]["K_pick"] == [32, 64, 128, 256]
    assert p["stale"]["control_period_s"] == pytest.approx(0.1)
    assert "not native" in p["independent_planner"]["implementation"]


def test_topology_screen_macros_match_the_frozen_evidence():
    """The map-only pre-screen (Sec. IV-F) must be re-derivable from the map
    files plus the frozen native check -- no hand-typed number, and no new
    simulation."""
    import topology_screen as ts  # noqa: E402

    rec = json.loads((PROJECT / "results" / "topology_screen.json").read_text())
    model, _ = ts.model_term_by_map()
    assert set(model) == set(rec["stats"]) and len(model) == 5
    assert sum(1 for v in model.values() if v > 0) == 1, "one open map"

    # corridor share separates the sign of the model-level term; the two
    # mechanisms reviewers name do not.
    for name, stats in rec["stats"].items():
        got = ts.map_stats(name)
        for key in ("rho1", "rho2", "art"):
            assert got[key] == pytest.approx(stats[key], abs=5e-5), (name, key)
    assert rec["features"]["rho2"]["separates"] is True
    assert rec["features"]["rho1"]["separates"] is False
    assert rec["features"]["art"]["separates"] is False
    wh = rec["stats"]["warehouse-10-20-10-2-1"]
    assert wh["rho1"] == 0.0 and wh["art"] == 0.0
    assert model["warehouse-10-20-10-2-1"] < 0

    tex = (PROJECT / "paper" / "topology_screen_numbers.tex").read_text()
    r2 = rec["features"]["rho2"]["values"]
    obs = [n for n in r2 if model[n] < 0]
    for macro, value in (
        ("ScrMaps", "5"),
        ("ScrObs", str(len(obs))),
        ("ScrOpenRho", f"{r2['empty-32-32']:.3f}"),
        ("ScrObsRhoMin", f"{min(r2[n] for n in obs):.3f}"),
        ("ScrObsRhoMax", f"{max(r2[n] for n in obs):.3f}"),
        ("ScrWhModel", f"{model['warehouse-10-20-10-2-1']:+.2f}"),
    ):
        assert "\\newcommand{\\%s}{%s}" % (macro, value) in tex, macro
