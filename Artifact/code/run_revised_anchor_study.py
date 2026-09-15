"""Locked, serial, restart-safe Paper-20 anchored-controller study.

This runner is intentionally isolated from every legacy planning artifact.  It
pre-enumerates all 320 attempt IDs, records a durable ``started`` event before
each episode, writes full JSON/NPZ traces atomically, and refuses to resume when
the locked specification or any computational input changes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import tempfile
import traceback
from pathlib import Path

import numpy as np

import np2compat  # noqa: F401  (read historical NumPy pickles compatibly)
from csq import AlphaSpendingQuantileCS, QuantileCS
from envloop import (EXECUTION_CONTRACT, IDENTITY_POLICY,
                     IDENTITY_SPEED_CAP_MPS, TRACKING_STRESS_CONTRACT,
                     make_calibrator, run_episode, window_scores_from_rollout)
from predictor import H, L, Predictor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "window_anchor_v1_study"
ATTEMPT_DIR = OUT / "attempts"
TRACE_DIR = OUT / "traces"
SPEC_PATH = OUT / "locked_spec.json"
LEDGER_PATH = OUT / "ledger.json"
SUMMARY_PATH = OUT / "summary.json"
SAME_STREAM_PATH = OUT / "same_frozen_stream.json"
DATA = ROOT / "results" / "data"
MODELS = ROOT / "results" / "models"

STUDY_ID = "p20_window_anchor_v1_revised_20260908"
SCORE_CONTRACT_ID = "window_max_lookahead_normalized_retrospective_clean_uid_v1"
DELTA = 0.10
BETA = 0.90
N_WINDOWS = 8
N_SEEDS = 40
TRACKING_BOUNDS = (0.0, 0.10)
REGIMES = ("sf", "orca")
METHODS = ("cs", "spending_cs")
SAME_STREAM_N = 100
SEED_RULE = "uint64(first8(SHA256('p20-window-anchor-v1|regime|seed_index')))"
PHASE_RULE = "2*pi*(seed_index+0.5)/40"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                    dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                    dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def episode_seed(regime: str, seed_index: int) -> int:
    token = f"p20-window-anchor-v1|{regime}|{seed_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def attempt_id(regime: str, method: str, tracking_bound: float,
               seed_index: int) -> str:
    return (f"{regime}__{method}__track-{tracking_bound:.2f}m__"
            f"seed-{seed_index:02d}")


def attempts() -> list[dict]:
    rows = []
    for regime in REGIMES:
        for tracking_bound in TRACKING_BOUNDS:
            for seed_index in range(N_SEEDS):
                for method in METHODS:  # paired methods are adjacent
                    rows.append({
                        "attempt_id": attempt_id(
                            regime, method, tracking_bound, seed_index),
                        "regime": regime,
                        "method": method,
                        "tracking_error_bound_m": tracking_bound,
                        "tracking_margin_m": tracking_bound,
                        "tracking_error_phase_rad": (
                            2.0 * math.pi * (seed_index + 0.5) / N_SEEDS),
                        "seed_index": seed_index,
                        "pcg64_seed": episode_seed(regime, seed_index),
                        "deployment_epoch_id": (
                            f"{STUDY_ID}/" + attempt_id(
                                regime, method, tracking_bound, seed_index)),
                    })
    return rows


def build_spec() -> dict:
    input_paths = {
        "cal_sf": DATA / "cal_sf.npy",
        "cal_orca": DATA / "cal_orca.npy",
        "gru_sf": MODELS / "gru_sf.pt",
        "gru_orca": MODELS / "gru_orca.pt",
        "replay_sf": DATA / "replay_sf.pkl",
        "replay_orca": DATA / "replay_orca.pkl",
    }
    source_names = (
        "csq.py", "envloop.py", "planner.py", "predictor.py", "sims.py",
        "run_revised_anchor_study.py", "check_revised_anchor_study.py")
    source_paths = {name: ROOT / "code" / name for name in source_names}
    return {
        "study_id": STUDY_ID,
        "artifact_schema_version": 1,
        "execution_contract": EXECUTION_CONTRACT,
        "score_contract_id": SCORE_CONTRACT_ID,
        "identity_policy": IDENTITY_POLICY,
        "tracking_stress_contract": TRACKING_STRESS_CONTRACT,
        "risk": {"epsilon": DELTA, "delta": DELTA, "beta": BETA,
                 "contract": "G1_time_uniform_quantile_domination"},
        "matrix": {
            "regimes": list(REGIMES), "methods": list(METHODS),
            "tracking_error_bounds_m": list(TRACKING_BOUNDS),
            "seeds_per_cell": N_SEEDS, "attempted_windows": N_WINDOWS,
            "vehicle": "double_integrator", "patrol": True,
            "total_attempts": len(attempts()), "workers": 1,
        },
        "randomness": {"episode_seed_rule": SEED_RULE,
                       "tracking_phase_rule": PHASE_RULE,
                       "paired_scope": "same initial PCG64 state across methods and tracking cells"},
        "comparator": {
            "name": "exact_binomial_alpha_spending",
            "monitoring_starts": "after fixed warm pool",
            "query_level": "delta/[k(k+1)]",
            "sample_count": "N_k=warm_size+k-1 for scored deployment queries",
            "same_G1_union_bound": True,
            "novelty_claimed": False,
        },
        "confidence_lifetime": {
            "reset_at_each_attempt": True,
            "new_lifetime_if_changed": [
                "immutable_base_pool_identity_or_bytes", "predictor_bytes",
                "score_or_selection_contract", "deployment_epoch"],
            "within_epoch_appending": "intended online update; does not reset",
            "portfolio_guarantee_over_320_attempts": False,
        },
        "interpretation": (
            "Fresh simulated descriptive study. Closed-loop methods share initial "
            "conditions but can induce different score streams. Same-stream radius "
            "comparison is separately frozen. Exchangeability is not established."),
        "same_frozen_stream": {
            "n_scores_per_regime": SAME_STREAM_N,
            "source": "first chronological valid window scores from frozen replay_<regime>.pkl",
            "base_pool": "full frozen cal_<regime>.npy",
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha_file(path)
                         for path in input_paths.values()},
        "source_sha256": {str(path.relative_to(ROOT)): sha_file(path)
                          for path in source_paths.values()},
        "attempts": attempts(),
    }


def trace_arrays(traj: list[dict]) -> dict[str, np.ndarray]:
    steps = len(traj)
    n = traj[0]["agents"].shape[0]
    constraint = np.full((steps, H, n, 2), np.nan, dtype=np.float64)
    scales = np.full((steps, H), np.nan, dtype=np.float64)
    horizon = np.zeros(steps, dtype=np.int64)
    for t, row in enumerate(traj):
        h = len(row["constraint_preds"])
        constraint[t, :h] = row["constraint_preds"]
        scales[t, :h] = row["constraint_region_scales"]
        horizon[t] = h
    return {
        "robot_before_control": np.stack([r["robot"] for r in traj]),
        "robot_planned_endpoint": np.stack([
            r["planned_robot_endpoint"] for r in traj]),
        "robot_realized_endpoint": np.stack([
            r["realized_robot_endpoint"] for r in traj]),
        "agents_before_control": np.stack([r["agents"] for r in traj]),
        "agents_after_control": np.stack([
            r["agents_after_control"] for r in traj]),
        "agent_substep_positions": np.stack([
            r["agent_substep_positions"] for r in traj]),
        "fresh_predictions": np.stack([r["fresh_preds"] for r in traj]),
        "constraint_predictions_padded": constraint,
        "constraint_region_scales_padded": scales,
        "constraint_horizon": horizon,
        "constraint_anchor_step": np.array([
            -1 if r["constraint_anchor_step"] is None
            else r["constraint_anchor_step"] for r in traj], dtype=np.int64),
        "radius": np.array([r["r"] for r in traj], dtype=np.float64),
        "observed_step_min_distance_m": np.array([
            r["observed_step_min_distance_m"] for r in traj]),
        "observed_substep_collision": np.array([
            r["observed_substep_collision"] for r in traj], dtype=np.bool_),
    }


def same_frozen_stream(spec_sha: str, predictors: dict[str, Predictor]) -> dict:
    result = {
        "artifact_schema_version": 1,
        "study_id": STUDY_ID,
        "locked_spec_sha256": spec_sha,
        "purpose": "literal_same_score_stream_same_G1_radius_comparison",
        "interpretation": (
            "Frozen agent-only replay analysis; no legacy robot-control outcome "
            "is used and no closed-loop exchangeability claim is made."),
        "regimes": {},
    }
    for regime in REGIMES:
        warm_path = DATA / f"cal_{regime}.npy"
        replay_path = DATA / f"replay_{regime}.pkl"
        warm = np.load(warm_path)
        with replay_path.open("rb") as handle:
            replay = pickle.load(handle)
        scores: list[float] = []
        for P, ev in replay:
            scores.extend(window_scores_from_rollout(P, ev, predictors[regime]))
            if len(scores) >= SAME_STREAM_N:
                break
        if len(scores) < SAME_STREAM_N:
            raise RuntimeError(f"only {len(scores)} frozen {regime} scores")
        scores = [float(x) for x in scores[:SAME_STREAM_N]]
        cs = QuantileCS(BETA, DELTA, warm)
        spending = AlphaSpendingQuantileCS(BETA, DELTA, warm)
        rows = []
        for k, score in enumerate(scores, 1):
            r_cs = float(cs.radius())
            r_sp = float(spending.radius())
            n_total = len(cs.sorted)
            cs._ensure_capacity(n_total)
            rows.append({
                "query_index": k,
                "total_score_count": n_total,
                "score": score,
                "mixture_cs_radius": r_cs if np.isfinite(r_cs) else None,
                "spending_cs_radius": r_sp if np.isfinite(r_sp) else None,
                "mixture_cs_rank": int(cs._u[n_total - 1]),
                "spending_cs_rank": int(spending.last_rank),
                "spending_alpha": float(spending.last_alpha),
                "mixture_cs_violation": bool(np.isfinite(r_cs) and score > r_cs),
                "spending_cs_violation": bool(np.isfinite(r_sp) and score > r_sp),
            })
            cs.add(score)
            spending.add(score)
        arr = np.asarray(scores, dtype="<f8")
        result["regimes"][regime] = {
            "base_pool_path": str(warm_path.relative_to(ROOT)),
            "base_pool_sha256": sha_file(warm_path),
            "base_pool_size": int(len(warm)),
            "replay_path": str(replay_path.relative_to(ROOT)),
            "replay_sha256": sha_file(replay_path),
            "score_stream_float64_le_sha256": hashlib.sha256(
                arr.tobytes()).hexdigest(),
            "n_scores": len(scores),
            "rows": rows,
        }
    return result


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - half, mid + half]


def bootstrap_mean_ci(values: list[float], seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(4000, len(x)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def completed_records(ledger: dict) -> list[dict]:
    out = []
    for attempt in ledger["attempts"]:
        if attempt["status"] != "completed":
            continue
        path = OUT / attempt["record_path"]
        out.append(json.loads(path.read_text()))
    return out


def build_summary(ledger: dict, same_stream_payload: dict) -> dict:
    records = completed_records(ledger)
    cells = {}
    for regime in REGIMES:
        for method in METHODS:
            for bound in TRACKING_BOUNDS:
                key = f"{regime}|{method}|track={bound:.2f}m"
                rs = [x for x in records if x["attempt"]["regime"] == regime
                      and x["attempt"]["method"] == method
                      and x["attempt"]["tracking_error_bound_m"] == bound]
                ep = [x["record"] for x in rs]
                active_steps = sum(r["steps"] - r["fallback_steps_warmup"] for r in ep)
                attempts_w = sum(r["n_attempted_windows"] for r in ep)
                scored = [w for r in ep for w in r["window_audit"]
                          if w["score_status"] == "scored"]
                finite = [w for w in scored if w["radius"] is not None]
                n_coll = sum(bool(r["collision"]) for r in ep)
                n_epviol = sum(bool(r["episode_viol"]) for r in ep)
                n_identity = sum(any(w["identity_policy_triggered"]
                                     for w in r["window_audit"]) for r in ep)
                cells[key] = {
                    "n_episodes": len(ep),
                    "n_attempted_windows": attempts_w,
                    "n_scored_windows": len(scored),
                    "finite_radius_window_fraction": len(finite) / max(len(scored), 1),
                    "finite_radius_containment_violation_fraction": (
                        sum(w["containment_violation"] is True for w in finite)
                        / max(len(finite), 1)),
                    "episode_containment_violation_fraction": n_epviol / max(len(ep), 1),
                    "episode_containment_violation_wilson95": wilson(n_epviol, len(ep)),
                    "observed_collision_episode_fraction": n_coll / max(len(ep), 1),
                    "observed_collision_episode_wilson95": wilson(n_coll, len(ep)),
                    "observed_collision_control_fraction": (
                        sum(r["n_coll_steps"] for r in ep)
                        / max(sum(r["steps"] for r in ep), 1)),
                    "active_window_fallback_fraction": (
                        sum(r["fallback_steps_active_window"] for r in ep)
                        / max(active_steps, 1)),
                    "identity_event_episode_fraction": n_identity / max(len(ep), 1),
                    "identity_event_episode_wilson95": wilson(n_identity, len(ep)),
                    "identity_policy_window_fraction": (
                        sum(w["identity_policy_triggered"] for r in ep
                            for w in r["window_audit"]) / max(attempts_w, 1)),
                    "selected_agent_fraction_over_attempted_window_agents": (
                        sum(w["clean_agent_count"] for w in scored)
                        / max(sum(w["total_agent_count"] for r in ep
                                  for w in r["window_audit"]), 1)),
                    "all_agents_in_score_event_window_fraction": (
                        sum(w["all_window_agents_in_score_event"] for w in scored)
                        / max(attempts_w, 1)),
                    "endpoint_implication_active_control_fraction": (
                        sum(r["selected_agent_endpoint_clearance_implication_labels"]
                            [L:].count(True) for r in ep) / max(active_steps, 1)),
                    "tracking_error_max_m": max(
                        (max(r["tracking_error_norm_labels"]) for r in ep), default=None),
                    "tracking_bound_violation_count": sum(
                        sum(not x for x in r["tracking_bound_labels"]) for r in ep),
                    "mean_path_length_m": float(np.mean(
                        [r["path_length_m"] for r in ep])) if ep else None,
                    "path_length_episode_bootstrap_ci95": bootstrap_mean_ci(
                        [r["path_length_m"] for r in ep], 1000 + len(cells)),
                    "mean_signed_waypoint_progress_m": float(np.mean(
                        [r["signed_waypoint_progress_m"] for r in ep])) if ep else None,
                    "mean_waypoint_completions": float(np.mean(
                        [r["waypoint_completion_count"] for r in ep])) if ep else None,
                    "fallback_reason_counts": {
                        reason: sum(1 for r in ep for row in r["fallback_trace"]
                                    if row["reason"] == reason)
                        for reason in sorted({row["reason"] for r in ep
                                              for row in r["fallback_trace"]})},
                }
    paired = {}
    for regime in REGIMES:
        for bound in TRACKING_BOUNDS:
            diffs = []
            for seed_index in range(N_SEEDS):
                pair = {x["attempt"]["method"]: x["record"] for x in records
                        if x["attempt"]["regime"] == regime
                        and x["attempt"]["tracking_error_bound_m"] == bound
                        and x["attempt"]["seed_index"] == seed_index}
                if set(pair) == set(METHODS):
                    diffs.append(pair["cs"]["path_length_m"]
                                 - pair["spending_cs"]["path_length_m"])
            key = f"{regime}|track={bound:.2f}m|mixture-minus-spending"
            paired[key] = {
                "n_paired_initial_conditions": len(diffs),
                "mean_path_length_difference_m": (
                    float(np.mean(diffs)) if diffs else None),
                "paired_seed_bootstrap_ci95": bootstrap_mean_ci(
                    diffs, 9000 + len(paired)),
                "caveat": "closed-loop score streams can differ after robot actions diverge",
            }
    return {
        "artifact_schema_version": 1,
        "study_id": STUDY_ID,
        "locked_spec_sha256": ledger["spec_sha256"],
        "generated_utc": now(),
        "n_predeclared_attempts": len(ledger["attempts"]),
        "n_completed_attempts": len(records),
        "n_failed_executions_retained": sum(
            e["status"] == "failed" for a in ledger["attempts"]
            for e in a["executions"]),
        "n_interrupted_executions_retained": sum(
            e["status"] == "interrupted" for a in ledger["attempts"]
            for e in a["executions"]),
        "cells": cells,
        "paired_initial_condition_comparisons": paired,
        "same_frozen_stream_sha256": canonical_sha(same_stream_payload),
        "interpretation": (
            "Episode-level simulated descriptive summaries; 40 attempts per cell "
            "share one frozen base pool/model. Windows are not treated as independent, "
            "zero observed collisions is not a safety proof, and there is no 320-run "
            "portfolio confidence guarantee."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="bounded checkpoint run; omission completes all pending attempts")
    args = parser.parse_args()
    if args.max_attempts is not None and args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")

    # This study is intentionally one worker.  Avoid hidden BLAS/Torch fan-out.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    OUT.mkdir(parents=True, exist_ok=True)
    ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    spec_sha = canonical_sha(spec)
    if SPEC_PATH.exists():
        old = json.loads(SPEC_PATH.read_text())
        if canonical_sha(old) != spec_sha:
            raise SystemExit("locked specification/input/source hash changed; refuse resume")
    else:
        atomic_json(SPEC_PATH, spec)

    if LEDGER_PATH.exists():
        ledger = json.loads(LEDGER_PATH.read_text())
        if ledger.get("spec_sha256") != spec_sha:
            raise SystemExit("ledger does not match locked specification")
        for item in ledger["attempts"]:
            if item["status"] == "started":
                item["executions"][-1]["status"] = "interrupted"
                item["executions"][-1]["interrupted_detected_utc"] = now()
                item["status"] = "pending"
        atomic_json(LEDGER_PATH, ledger)
    else:
        ledger = {
            "artifact_schema_version": 1,
            "study_id": STUDY_ID,
            "spec_sha256": spec_sha,
            "created_utc": now(),
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": package_version("numpy"),
                "scipy": package_version("scipy"),
                "torch": package_version("torch"),
            },
            "attempts": [{"attempt_id": row["attempt_id"], "status": "pending",
                          "executions": []} for row in spec["attempts"]],
        }
        atomic_json(LEDGER_PATH, ledger)

    predictors = {regime: Predictor(str(MODELS / f"gru_{regime}.pt"))
                  for regime in REGIMES}
    same_payload = same_frozen_stream(spec_sha, predictors)
    atomic_json(SAME_STREAM_PATH, same_payload)
    ledger["same_frozen_stream_path"] = str(SAME_STREAM_PATH.relative_to(OUT))
    ledger["same_frozen_stream_file_sha256"] = sha_file(SAME_STREAM_PATH)
    atomic_json(LEDGER_PATH, ledger)

    spec_by_id = {row["attempt_id"]: row for row in spec["attempts"]}
    completed_now = 0
    for item in ledger["attempts"]:
        if item["status"] == "completed":
            continue
        if args.max_attempts is not None and completed_now >= args.max_attempts:
            break
        a = spec_by_id[item["attempt_id"]]
        execution = {"execution_index": len(item["executions"]) + 1,
                     "status": "started", "started_utc": now()}
        item["executions"].append(execution)
        item["status"] = "started"
        atomic_json(LEDGER_PATH, ledger)
        try:
            warm_path = DATA / f"cal_{a['regime']}.npy"
            model_path = MODELS / f"gru_{a['regime']}.pt"
            warm = np.load(warm_path)
            calibrator = make_calibrator(a["method"], DELTA, warm)
            record = run_episode(
                a["regime"], calibrator, a["method"], predictors[a["regime"]],
                np.random.default_rng(a["pcg64_seed"]), n_windows=N_WINDOWS,
                patrol=True, vehicle="di", latency=0, act_noise=0.0,
                tracking_margin=a["tracking_margin_m"],
                tracking_error_bound=a["tracking_error_bound_m"],
                tracking_error_phase=a["tracking_error_phase_rad"],
                identity_policy=IDENTITY_POLICY,
                identity_agent_speed_bound=IDENTITY_SPEED_CAP_MPS[a["regime"]],
                record_traj=True, record_audit_trace=True)
            traj = record.pop("traj")
            trace_path = TRACE_DIR / f"{a['attempt_id']}.npz"
            atomic_npz(trace_path, **trace_arrays(traj))
            completed_utc = now()
            payload = {
                "artifact_schema_version": 1,
                "study_id": STUDY_ID,
                "locked_spec_sha256": spec_sha,
                "attempt": a,
                "provenance": {
                    "base_pool_path": str(warm_path.relative_to(ROOT)),
                    "base_pool_sha256": sha_file(warm_path),
                    "base_pool_size": int(len(warm)),
                    "predictor_path": str(model_path.relative_to(ROOT)),
                    "predictor_sha256": sha_file(model_path),
                    "score_contract_id": SCORE_CONTRACT_ID,
                    "deployment_epoch_id": a["deployment_epoch_id"],
                    "confidence_state": "fresh_at_epoch_start",
                    "within_epoch_updates": "append_scored_windows_without_reset",
                    "started_utc": execution["started_utc"],
                    "completed_utc": completed_utc,
                },
                "trace": {
                    "path": str(trace_path.relative_to(OUT)),
                    "sha256": sha_file(trace_path),
                    "bytes": trace_path.stat().st_size,
                },
                "record": record,
            }
            record_path = ATTEMPT_DIR / f"{a['attempt_id']}.json"
            atomic_json(record_path, payload)
            execution.update({
                "status": "completed", "completed_utc": completed_utc,
                "record_path": str(record_path.relative_to(OUT)),
                "record_sha256": sha_file(record_path),
                "trace_path": str(trace_path.relative_to(OUT)),
                "trace_sha256": sha_file(trace_path),
            })
            item.update({"status": "completed",
                         "record_path": str(record_path.relative_to(OUT)),
                         "record_sha256": sha_file(record_path),
                         "trace_path": str(trace_path.relative_to(OUT)),
                         "trace_sha256": sha_file(trace_path)})
            completed_now += 1
            atomic_json(LEDGER_PATH, ledger)
            print(f"COMPLETE {a['attempt_id']} ({completed_now} this run)", flush=True)
        except Exception as exc:
            execution.update({"status": "failed", "completed_utc": now(),
                              "exception_type": type(exc).__name__,
                              "exception": str(exc),
                              "traceback": traceback.format_exc()})
            item["status"] = "pending"
            atomic_json(LEDGER_PATH, ledger)
            raise

    summary = build_summary(ledger, same_payload)
    atomic_json(SUMMARY_PATH, summary)
    done = summary["n_completed_attempts"]
    print(f"CHECKPOINT {done}/{len(spec['attempts'])} complete; "
          f"summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
