"""Independent, read-only audit of the frozen Paper 17 evidence corpus.

This script never imports or runs the simulator.  It validates the active raw
JSON ledger, derives clearly labelled 100-step/tail summaries from already
stored checkpoints, inventories optimizer states, and hashes the release
inputs.  Its sole write is the deterministic JSON report requested with
``--output``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev, stdev

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
RAW = RESULTS / "raw"
CACHE = RESULTS / "cache"

EXPECTED_GROUP_COUNTS = {
    "exp1b": 600,
    "exp2c_eval_cold": 10,
    "exp2c_eval_fullcold": 10,
    "exp2c_eval_fullwarm": 10,
    "exp2c_eval_warm": 10,
    "exp3b_agentcount": 250,
    "exp3b_demandshift": 160,
    "exp4b": 15,
    "exp6b": 260,
    "exp7": 80,
    "exp8": 120,
}
HEADLINE_CELLS = {
    "empty-32-32": (100, 250, 400),
    "random-32-32-20": (100, 200, 320),
    "room-32-32-4": (70, 170, 270),
    "maze-32-32-4": (80, 200, 310),
    "warehouse-10-20-10-2-1": (400, 800, 1600),
}
HEADLINE_METHODS = ("unweighted", "lanes", "tolls", "tolls-online")
SEARCH_TAGS = ("cold", "warm", "fullcold", "fullwarm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def expected_raw_name(cond: dict) -> str:
    encoded = json.dumps({k: v for k, v in sorted(cond.items())}, sort_keys=True)
    suffix = hashlib.md5(encoded.encode()).hexdigest()[:10]
    return (f"{cond['map']}_{cond['method']}_N{cond['N']}_s{cond['seed']}_"
            f"{suffix}.json")


def load_raw() -> tuple[list[tuple[Path, dict]], dict]:
    records = []
    group_counts = {}
    for directory in sorted(path for path in RAW.iterdir() if path.is_dir()):
        files = sorted(directory.glob("*.json"))
        group_counts[directory.name] = len(files)
        for path in files:
            rec = json.loads(path.read_text())
            required = {"throughput", "completed", "steps", "series", "cond"}
            missing = sorted(required - rec.keys())
            if missing:
                raise AssertionError(f"{path}: missing {missing}")
            if path.name != expected_raw_name(rec["cond"]):
                raise AssertionError(f"{path}: condition hash/name mismatch")
            times = [point["t"] for point in rec["series"]]
            if times != list(range(100, rec["steps"] + 1, 100)):
                raise AssertionError(f"{path}: noncanonical 100-step series")
            if rec["series"][-1]["completed"] != rec["completed"]:
                raise AssertionError(f"{path}: terminal completion mismatch")
            records.append((path, rec))
    if group_counts != EXPECTED_GROUP_COUNTS:
        raise AssertionError(
            f"active raw counts changed: {group_counts} != {EXPECTED_GROUP_COUNTS}")
    return records, group_counts


def run_windows(rec: dict) -> list[float]:
    previous = 0
    rates = []
    for point in rec["series"]:
        rates.append((point["completed"] - previous) / 100.0)
        previous = point["completed"]
    return rates


def mean_std(values: list[float]) -> dict:
    return {
        "mean": round(fmean(values), 6),
        "sample_std": round(stdev(values), 6) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def headline_tail_audit(records: list[tuple[Path, dict]]) -> dict:
    selected = [rec for path, rec in records if path.parent.name == "exp1b"]
    by_key = {}
    for rec in selected:
        cond = rec["cond"]
        key = (cond["map"], int(cond["N"]), cond["method"])
        by_key.setdefault(key, []).append(rec)

    expected = {(m, n, method) for m, ns in HEADLINE_CELLS.items()
                for n in ns for method in HEADLINE_METHODS}
    if set(by_key) != expected:
        raise AssertionError("headline condition grid is incomplete or has extras")

    cells = {}
    terminal_zero_total = 0
    any_zero_total = 0
    total_runs = 0
    ratios = []
    for key in sorted(by_key):
        map_name, n_agents, method = key
        runs = sorted(by_key[key], key=lambda rec: int(rec["cond"]["seed"]))
        seeds = [int(rec["cond"]["seed"]) for rec in runs]
        if seeds != list(range(10)):
            raise AssertionError(f"{key}: expected seeds 0..9, got {seeds}")
        overall, first_half, last_half, terminal = [], [], [], []
        terminal_zero, any_zero = 0, 0
        for rec in runs:
            rates = run_windows(rec)
            overall.append(float(rec["throughput"]))
            first_half.append(fmean(rates[:5]))
            last_half.append(fmean(rates[5:]))
            terminal.append(rates[-1])
            terminal_zero += int(rates[-1] == 0.0)
            any_zero += int(any(rate == 0.0 for rate in rates))
        ratio = fmean(last_half) / max(fmean(overall), 1e-12)
        ratios.append(ratio)
        terminal_zero_total += terminal_zero
        any_zero_total += any_zero
        total_runs += len(runs)
        cell_key = f"{map_name}@N{n_agents}"
        cells.setdefault(cell_key, {})[method] = {
            "overall_1000_step": mean_std(overall),
            "first_500_step": mean_std(first_half),
            "last_500_step": mean_std(last_half),
            "terminal_100_step": mean_std(terminal),
            "last_half_over_overall": round(ratio, 6),
            "terminal_zero_100_step_count": terminal_zero,
            "any_zero_100_step_count": any_zero,
        }
    return {
        "status": "post_hoc_derived_from_frozen_cumulative_checkpoints",
        "interpretation_limit": (
            "These are within-run diagnostics, not substantially longer-horizon "
            "evidence and not estimates of eventual absorbing-state probability."),
        "horizon_steps": 1000,
        "checkpoint_period_steps": 100,
        "cells": cells,
        "aggregate": {
            "runs": total_runs,
            "terminal_zero_100_step_events": terminal_zero_total,
            "any_zero_100_step_events": any_zero_total,
            "min_last_half_over_overall": round(min(ratios), 6),
            "max_last_half_over_overall": round(max(ratios), 6),
        },
    }


def search_audit(records: list[tuple[Path, dict]]) -> dict:
    source = (PROJECT / "code" / "exp2c_es_strong.py").read_text()
    if "EVAL_SEEDS = (10007, 10009)" not in source or "EVAL_STEPS = 500" not in source:
        raise AssertionError("optimizer training protocol changed")
    source_now_retains_start = (
        "initial_candidate_retained = True" in source
        and "initial_fit, initial_eval_time = eval_theta(x0)" in source)

    toll_path = (CACHE /
                 "tolls_empty-32-32_N400_a1.0_b2_g2.0_KNone_s0_L_m1.0.npz")
    with np.load(toll_path, allow_pickle=True) as z:
        toll_weights = z["w"].astype(float)
    exact_full_start = np.exp(np.clip(np.log(toll_weights), -2.0, 2.0))

    raw_by_exp = {}
    for path, rec in records:
        raw_by_exp.setdefault(path.parent.name, []).append(rec)
    toll_runs = [rec for rec in raw_by_exp["exp1b"]
                 if rec["cond"]["map"] == "empty-32-32"
                 and rec["cond"]["N"] == 400
                 and rec["cond"]["method"] == "tolls"]
    toll_heldout = mean_std([float(rec["throughput"]) for rec in toll_runs])

    out = {}
    for tag in SEARCH_TAGS:
        state_path = CACHE / f"es_strong_{tag}.npz"
        best_path = CACHE / f"es_strong_best_{tag}.npz"
        with np.load(state_path, allow_pickle=True) as state, \
                np.load(best_path, allow_pickle=True) as best:
            history = json.loads(str(state["history"]))
            legacy_unretained = "initial_fit" not in state.files
            if not np.array_equal(state["best_theta"], best["theta"]):
                raise AssertionError(f"{tag}: state/best incumbent mismatch")
            eval_records = raw_by_exp[f"exp2c_eval_{tag}"]
            eval_seeds = sorted(int(rec["cond"]["seed"]) for rec in eval_records)
            if eval_seeds != list(range(10)):
                raise AssertionError(f"{tag}: held-out seeds are not 0..9")
            entry = {
                "parameterization": str(best["param"]),
                "dimension": int(best["dim"]),
                "generations": int(state["gen"]),
                "training_rollout_evaluations": int(state["n_evals"]),
                "training_objective": "mean throughput; seeds 10007 and 10009; 500 steps",
                "selected_candidate": "best sampled candidate by training objective",
                "selected_training_throughput": round(-float(best["fit"]), 6),
                "selected_heldout_protocol": "seeds 0..9; 1000 steps",
                "selected_heldout_throughput": mean_std(
                    [float(rec["throughput"]) for rec in eval_records]),
                "last_generation_sample_best": round(float(history[-1]["best"]), 6),
                "best_ever_history_final": round(float(history[-1]["best_ever"]), 6),
                "best_candidate_array_sha256": sha256_array(best["theta"]),
                "terminal_mean_array_sha256": sha256_array(state["mean"]),
                "terminal_mean_performance": "not_evaluated_in_frozen_artifacts",
                "starting_candidate_evaluated_before_sampling": not legacy_unretained,
                "starting_candidate_retained_as_incumbent": not legacy_unretained,
                "policy_evidence": (
                    "legacy frozen state has no initial_fit/initial_theta fields and "
                    "its history begins at generation 1" if legacy_unretained else
                    "state records an explicit generation-0 incumbent"),
                "state_file_sha256": sha256_file(state_path),
                "best_file_sha256": sha256_file(best_path),
            }
            if tag == "fullwarm":
                entry.update({
                    "starting_candidate": "exact toll field",
                    "starting_field_exact_after_parameterization": bool(
                        np.array_equal(exact_full_start, toll_weights)),
                    "starting_candidate_heldout_proxy": toll_heldout,
                    "starting_candidate_note": (
                        "The same frozen toll field was evaluated under the 10-seed "
                        "headline protocol, but it was not protected by the CMA-ES "
                        "incumbent logic."),
                })
            elif tag == "warm":
                entry["starting_candidate"] = "least-squares projection of toll field"
                entry["projection_mean_relative_error"] = round(float(best["proj_err"]), 8)
                entry["starting_candidate_performance"] = "not_saved_or_evaluated"
            else:
                entry["starting_candidate"] = "zero log-weight vector (unit weights)"
                entry["starting_candidate_performance"] = "not_saved_or_evaluated_by_search"
            out[tag] = entry
    return {
        "status": "frozen_optimizer_state_reclassification",
        "current_source_retains_new_starting_candidates": source_now_retains_start,
        "selection_split": (
            "Training uses seeds 10007/10009 at 500 steps; the selected sampled "
            "candidate uses disjoint seeds 0..9 at 1000 steps for reporting. There "
            "is no separate validation split."),
        "incumbent_conclusion": (
            "The frozen implementation did not evaluate or preserve its initial "
            "candidate. Therefore the warm-start result cannot support a claim that "
            "incumbent-preserving search degraded the toll field."),
        "runs": out,
    }


def summary_crosscheck(records: list[tuple[Path, dict]]) -> dict:
    """Recompute every headline and compute-table throughput cell from raw JSON."""
    summary = json.loads((RESULTS / "summary_v2.json").read_text())
    raw_by_key: dict[tuple[str, str, int, str], list[float]] = {}
    for path, rec in records:
        cond = rec["cond"]
        key = (path.parent.name, str(cond["map"]), int(cond["N"]),
               str(cond["method"]))
        raw_by_key.setdefault(key, []).append(float(rec["throughput"]))

    headline_fields = 0
    for row in summary["headline"]:
        map_name, n_agents = str(row["map"]), int(row["N"])
        for method in HEADLINE_METHODS:
            stored = row[method]
            values = raw_by_key[("exp1b", map_name, n_agents, method)]
            expected = {
                "mean": round(fmean(values), 4),
                # The released summary/table convention is population standard
                # deviation.  The remediation prose labels that convention.
                "std": round(pstdev(values), 4),
                "n": len(values),
            }
            actual = {field: stored[field] for field in expected}
            if actual != expected:
                raise AssertionError(
                    f"headline mismatch {map_name}@N{n_agents}/{method}: "
                    f"{actual} != {expected}")
            headline_fields += len(expected)

    compute_fields = 0
    compute = summary["compute_vs_tput"]
    for method in (*HEADLINE_METHODS,
                   "es-strong-cold", "es-strong-warm",
                   "es-strong-fullcold", "es-strong-fullwarm"):
        group = ("exp1b" if method in HEADLINE_METHODS else
                 f"exp2c_eval_{method.removeprefix('es-strong-')}")
        values = raw_by_key[(group, "empty-32-32", 400, method)]
        expected = {
            "tput": round(fmean(values), 4),
            "std": round(pstdev(values), 4),
            "n": len(values),
        }
        actual = {field: compute[method][field] for field in expected}
        if actual != expected:
            raise AssertionError(
                f"compute-table mismatch for {method}: {actual} != {expected}")
        compute_fields += len(expected)

    return {
        "status": "exact_match_to_raw_json",
        "headline_cells_checked": len(summary["headline"]) * len(HEADLINE_METHODS),
        "headline_fields_checked": headline_fields,
        "compute_cells_checked": 8,
        "compute_fields_checked": compute_fields,
        "standard_deviation_convention": "population (ddof=0), as released",
    }


def artifact_manifest(records: list[tuple[Path, dict]]) -> dict:
    raw_files = []
    tree = hashlib.sha256()
    for path, _ in records:
        rel = path.relative_to(PROJECT).as_posix()
        digest = sha256_file(path)
        raw_files.append({"path": rel, "sha256": digest})
        tree.update(rel.encode("utf-8") + b"\0" + bytes.fromhex(digest))
    maps = {}
    for path in sorted((PROJECT / "data" / "maps").glob("*.map")):
        maps[path.name] = sha256_file(path)
    core = {}
    for rel in ("results/summary_v2.json", "results/certificate.json",
                "results/tie_generality.json", "results/directionality.csv",
                "paper/tables.tex", "paper/numbers.tex"):
        path = PROJECT / rel
        core[rel] = sha256_file(path)
    return {
        "active_raw_tree_sha256": tree.hexdigest(),
        "active_raw_files": raw_files,
        "map_sha256": maps,
        "core_artifact_sha256": core,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records, group_counts = load_raw()
    steps = sorted({int(rec["steps"]) for _, rec in records})
    report = {
        "audit": "paper17_frozen_evidence_v1",
        "policy": "read existing artifacts only; no simulation or optimization",
        "active_raw_group_counts": group_counts,
        "active_raw_total": len(records),
        "stored_horizons_steps": steps,
        "longer_than_1000_step_run_present": any(step > 1000 for step in steps),
        "stored_trace_fields": ["100-step cumulative completions"],
        "not_stored": [
            "per-timestep positions",
            "goal-assignment event streams",
            "asynchronous/stale-guidance execution",
            "warehouse commodity-resolution sweep",
            "randomized or alternative lane initializations",
            "seed-only (factor 1.5) no-optimization control",
            "native/reference-implementation PIBT comparison",
        ],
        "protocol_facts": {
            "headline": "15 cells x 4 methods x 10 seeds x 1000 steps",
            "task_generator": "deterministic from condition seed and code; event stream not logged",
            "online_execution": (
                "synchronous: simulation waits while the flow solve and distance-table "
                "rebuild execute; tasks/step excludes wall-clock delay"),
            "warehouse_demand": (
                "all 122 station destinations plus a fixed-seed subset of 64 among "
                "2654 pickup destinations (186 destination commodities total)"),
            "lane_initialization": (
                "factor-1.5 crisscross bias for the initial AON assignment; subsequent "
                "FW steps use the objective gradient"),
            "pibt_scope": "repository-local PIBT-style reimplementation with swap extension",
        },
        "headline_tail_audit": headline_tail_audit(records),
        "summary_crosscheck": summary_crosscheck(records),
        "search_audit": search_audit(records),
        "manifest": artifact_manifest(records),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
        print(f"wrote {output} ({len(records)} raw records checked)")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
