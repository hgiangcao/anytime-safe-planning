"""Strict validator for ``results/window_anchor_v1_study``.

The checker recomputes hashes, score values, anchor suffixes, identity-policy
inputs, tracking/collision labels, and the literal same-stream comparator.  It
rejects stale/recentered constraints and partial matrices.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from csq import AlphaSpendingQuantileCS, QuantileCS
from predictor import H, L
from run_revised_anchor_study import (ATTEMPT_DIR, BETA, DATA, DELTA,
                                      LEDGER_PATH, METHODS, N_SEEDS,
                                      N_WINDOWS, OUT, REGIMES, SAME_STREAM_PATH,
                                      SPEC_PATH, SUMMARY_PATH, TRACKING_BOUNDS,
                                      build_spec, canonical_sha, sha_file)
from sims import COLL_DIST, SUBSTEPS


PASS = 0
FAIL: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    if condition:
        PASS += 1
        print(f"PASS {name}" + (f": {detail}" if detail else ""))
    else:
        FAIL.append(f"{name}: {detail}")
        print(f"FAIL {name}: {detail}")


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def validate_same_stream(spec_sha: str, ledger: dict) -> None:
    check("same-stream artifact exists", SAME_STREAM_PATH.exists())
    if not SAME_STREAM_PATH.exists():
        return
    payload = json.loads(SAME_STREAM_PATH.read_text())
    check("same-stream file hash",
          sha_file(SAME_STREAM_PATH) == ledger["same_frozen_stream_file_sha256"])
    check("same-stream spec binding", payload["locked_spec_sha256"] == spec_sha)
    for regime in REGIMES:
        block = payload["regimes"][regime]
        warm_path = OUT.parents[1] / block["base_pool_path"]
        warm = np.load(warm_path)
        rows = block["rows"]
        scores = np.asarray([r["score"] for r in rows], dtype="<f8")
        check(f"same-stream {regime} count", len(rows) == 100)
        check(f"same-stream {regime} score hash",
              hashlib.sha256(scores.tobytes()).hexdigest()
              == block["score_stream_float64_le_sha256"])
        cs = QuantileCS(BETA, DELTA, warm)
        sp = AlphaSpendingQuantileCS(BETA, DELTA, warm)
        ok = True
        for k, row in enumerate(rows, 1):
            r_cs, r_sp = float(cs.radius()), float(sp.radius())
            expect_cs = None if not np.isfinite(r_cs) else r_cs
            expect_sp = None if not np.isfinite(r_sp) else r_sp
            ok &= (row["query_index"] == k
                   and row["total_score_count"] == len(warm) + k - 1
                   and (row["mixture_cs_radius"] is None
                        if expect_cs is None else close(row["mixture_cs_radius"], expect_cs))
                   and (row["spending_cs_radius"] is None
                        if expect_sp is None else close(row["spending_cs_radius"], expect_sp))
                   and row["spending_cs_rank"] == sp.last_rank
                   and close(row["spending_alpha"], DELTA / (k * (k + 1.0))))
            cs.add(row["score"])
            sp.add(row["score"])
        check(f"same-stream {regime} exact replay", ok)


def validate_attempt(spec_row: dict, item: dict) -> tuple[bool, dict]:
    aid = spec_row["attempt_id"]
    record_path = OUT / item["record_path"]
    trace_path = OUT / item["trace_path"]
    if not record_path.exists() or not trace_path.exists():
        return False, {"error": "missing record or trace"}
    if sha_file(record_path) != item["record_sha256"]:
        return False, {"error": "record hash mismatch"}
    if sha_file(trace_path) != item["trace_sha256"]:
        return False, {"error": "trace hash mismatch"}
    payload = json.loads(record_path.read_text())
    r = payload["record"]
    if payload["attempt"] != spec_row:
        return False, {"error": "attempt specification mismatch"}
    if payload["trace"]["sha256"] != sha_file(trace_path):
        return False, {"error": "payload trace hash mismatch"}
    if not (r["artifact_schema_version"] == 3
            and r["execution_contract"] == "window_anchor_v1"
            and r["steps"] == L + N_WINDOWS * H == 69
            and r["n_attempted_windows"] == N_WINDOWS
            and len(r["control_audit_trace"]) == r["steps"]
            and r["fallback_steps"] == r["fallback_steps_warmup"]
            + r["fallback_steps_active_window"]
            and r["fallback_steps_warmup"] == L):
        return False, {"error": "schema/length/fallback invariant"}
    if payload["provenance"]["deployment_epoch_id"] != spec_row["deployment_epoch_id"]:
        return False, {"error": "epoch mismatch"}

    z = np.load(trace_path, allow_pickle=False)
    fresh = z["fresh_predictions"]
    constraint = z["constraint_predictions_padded"]
    scales = z["constraint_region_scales_padded"]
    horizons = z["constraint_horizon"]
    anchors = z["constraint_anchor_step"]
    agents_before = z["agents_before_control"]
    agents_after = z["agents_after_control"]
    subs = z["agent_substep_positions"]
    robot_before = z["robot_before_control"]
    robot_plan = z["robot_planned_endpoint"]
    robot_real = z["robot_realized_endpoint"]
    if not (len(fresh) == len(constraint) == len(anchors) == 69
            and subs.shape[1] == SUBSTEPS):
        return False, {"error": "NPZ shape invariant"}
    if not np.array_equal(agents_after[:-1], agents_before[1:]):
        return False, {"error": "agent trace is discontinuous"}

    uid_generation = np.zeros(agents_before.shape[1], dtype=int)
    anchor_ok = True
    identity_ok = True
    tracking_ok = True
    collision_ok = True
    for t, ctl in enumerate(r["control_audit_trace"]):
        h = int(horizons[t])
        expected_h = H if t < L else H - ((t - L) % H)
        if h != expected_h:
            anchor_ok = False
        if anchors[t] < 0:
            expect = fresh[t]
            expect_scales = np.arange(1, H + 1) / H
        elif ctl["identity_policy_fallback"]:
            expect = np.repeat(agents_before[t][None, :, :], h, axis=0)
            expect_scales = np.arange(1, h + 1) / h
            identity_ok &= ctl["identity_policy_input"].startswith(
                "current_observation")
        else:
            anchor = int(anchors[t])
            offset = t - anchor
            expect = fresh[anchor, offset:]
            expect_scales = np.arange(offset + 1, H + 1) / H
            anchor_ok &= anchor == L + ((t - L) // H) * H
        anchor_ok &= (np.array_equal(constraint[t, :h], expect)
                      and np.allclose(scales[t, :h], expect_scales)
                      and ctl["constraint_anchor_step"]
                      == (None if anchors[t] < 0 else int(anchors[t])))
        anchor_ok &= ctl["constraint_prediction_sha256"] == hashlib.sha256(
            np.asarray(constraint[t, :h], dtype="<f8").tobytes()).hexdigest()
        expect_uids = [f"{i}:{int(g)}" for i, g in enumerate(uid_generation)]
        identity_ok &= ctl["observed_agent_uids_at_plan"] == expect_uids
        for transition in ctl["identity_transitions_observed_after_control"]:
            slot = transition["slot"]
            identity_ok &= (transition["from_uid"]
                            == f"{slot}:{uid_generation[slot]}")
            uid_generation[slot] += 1
            identity_ok &= (transition["to_uid"]
                            == f"{slot}:{uid_generation[slot]}")

        err = float(np.linalg.norm(robot_real[t] - robot_plan[t]))
        tracking_ok &= (close(err, ctl["tracking_error_norm_m"], 2e-10)
                        and err <= spec_row["tracking_error_bound_m"] + 2e-10
                        and ctl["tracking_bound_satisfied"])
        fr = (np.arange(1, SUBSTEPS + 1) / SUBSTEPS)[:, None]
        robot_sub = robot_before[t][None, :] * (1 - fr) + robot_real[t][None, :] * fr
        md = float(np.linalg.norm(
            subs[t] - robot_sub[:, None, :], axis=2).min())
        transition_slots = [x["slot"] for x in
                            ctl["identity_transitions_observed_after_control"]]
        if transition_slots:
            md = min(md, float(np.linalg.norm(
                agents_after[t, transition_slots] - robot_real[t][None, :],
                axis=1).min()))
        collision_ok &= (close(md, ctl["observed_step_min_distance_m"], 2e-10)
                         and bool(md < COLL_DIST)
                         == ctl["observed_substep_collision"])
    if not anchor_ok:
        return False, {"error": "stale/recentered anchor or scale"}
    if not identity_ok:
        return False, {"error": "identity UID/current-observation policy"}
    if not tracking_ok:
        return False, {"error": "tracking bound/trace"}
    if not collision_ok:
        return False, {"error": "collision trace"}

    # Recompute every scored window from the full anchored prediction and
    # realized agent endpoints, including the retrospective selected set.
    score_ok = True
    for w in r["window_audit"]:
        t0 = int(w["prediction_issued_step"])
        pred = fresh[t0]
        score_ok &= (w["anchor_prediction_sha256"] == hashlib.sha256(
            np.asarray(pred, dtype="<f8").tobytes()).hexdigest())
        selected = w["score_selected_agent_indices"]
        if selected:
            actual = agents_after[t0:t0 + H, selected, :]
            err = np.linalg.norm(actual - pred[:, selected, :], axis=2)
            err = err / (np.arange(1, H + 1)[:, None] / H)
            score_ok &= close(float(err.max()), w["score"], 2e-10)
        else:
            score_ok &= w["score"] is None
        score_ok &= len(w["score_selected_agent_uids"]) == w["clean_agent_count"]
    if not score_ok:
        return False, {"error": "window score/hash/selection"}
    if not close(float(np.linalg.norm(
            robot_real - robot_before, axis=1).sum()), r["path_length_m"], 2e-9):
        return False, {"error": "path length"}
    return True, r


def main() -> None:
    check("locked spec exists", SPEC_PATH.exists())
    check("ledger exists", LEDGER_PATH.exists())
    if not SPEC_PATH.exists() or not LEDGER_PATH.exists():
        raise SystemExit(1)
    spec = json.loads(SPEC_PATH.read_text())
    ledger = json.loads(LEDGER_PATH.read_text())
    spec_sha = canonical_sha(spec)
    check("current sources and inputs match lock",
          canonical_sha(build_spec()) == spec_sha)
    check("ledger spec binding", ledger["spec_sha256"] == spec_sha)
    check("predeclared matrix size",
          len(spec["attempts"]) == len(ledger["attempts"]) == 320)
    check("all attempts completed",
          all(a["status"] == "completed" for a in ledger["attempts"]),
          str(Counter(a["status"] for a in ledger["attempts"])))
    validate_same_stream(spec_sha, ledger)

    spec_by_id = {a["attempt_id"]: a for a in spec["attempts"]}
    valid = 0
    records = []
    errors = []
    for item in ledger["attempts"]:
        if item["status"] != "completed":
            continue
        ok, result = validate_attempt(spec_by_id[item["attempt_id"]], item)
        if ok:
            valid += 1
            records.append((spec_by_id[item["attempt_id"]], result))
        else:
            errors.append(f"{item['attempt_id']}: {result['error']}")
    check("all completed traces validate", valid == 320,
          f"valid={valid}; first_errors={errors[:3]}")
    counts = Counter((a["regime"], a["method"], a["tracking_error_bound_m"])
                     for a, _ in records)
    check("40 episodes in every locked cell",
          len(counts) == 8 and set(counts.values()) == {N_SEEDS}, str(counts))
    check("accepted action example retained",
          any("certified" in r["control_audit_trace"][t]["plan_status"]
              and r["control_audit_trace"][t]["plan_status"] == "certified"
              for _, r in records for t in range(L, r["steps"])))
    check("fallback example retained",
          any(r["fallback_steps_active_window"] > 0 for _, r in records))
    check("identity/excluded-agent example retained",
          any(w["identity_policy_triggered"]
              or w["excluded_at_anchor_agent_uids"]
              for _, r in records for w in r["window_audit"]))
    check("tracking bounds all satisfied",
          all(all(r["tracking_bound_labels"]) for _, r in records))

    check("summary exists", SUMMARY_PATH.exists())
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
        check("summary bound to full matrix",
              summary["locked_spec_sha256"] == spec_sha
              and summary["n_completed_attempts"] == 320
              and len(summary["cells"]) == 8)
    print(f"\n{PASS} validation gates passed; {len(FAIL)} failed")
    for item in FAIL:
        print(" -", item)
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
