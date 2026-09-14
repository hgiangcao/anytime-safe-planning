"""Independent, generator-free audit of the locked 2026-09-08 corpus.

This checker intentionally does not import ``review_round2`` or simulator code.
It validates the released files, recomputes summaries from JSON/NPZ records,
and writes its report outside the hash-locked result directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_round2"
REPORT = ROOT / "results" / "checks" / "review_round2_independent_audit.json"
FAMILIES = ("attribution", "search_test", "long_horizon",
            "warehouse_resolution", "stale", "independent_planner")
EXPECTED = {"attribution": 30, "search_test": 6, "long_horizon": 27,
            "warehouse_resolution": 8, "stale": 9,
            "independent_planner": 9}


class Audit:
    def __init__(self) -> None:
        self.checks = 0
        self.state_entries = 0
        self.event_entries = 0

    def require(self, condition: bool, message: str, weight: int = 1) -> None:
        if not condition:
            raise AssertionError(message)
        self.checks += int(weight)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def free_cells(map_name: str) -> int:
    lines = (ROOT / "data" / "maps" / f"{map_name}.map").read_text().splitlines()[4:]
    return sum(ch not in "@TO" for line in lines for ch in line)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def audit() -> dict:
    a = Audit()
    protocol = read_json(OUT / "protocol.json")
    summary = read_json(OUT / "summary.json")
    manifest = read_json(OUT / "manifest.json")

    a.require(protocol["locked_at"] == "2026-09-08T15:30:00+07:00",
              "unexpected protocol lock time")
    a.require(protocol["common"] == {"eps_tie": 1.0,
                                      "negative_outcomes_retained": True,
                                      "swap": True, "workers": 1},
              "common protocol changed")
    a.require(sha256(ROOT / protocol["protocol_markdown"]) ==
              protocol["protocol_markdown_sha256"],
              "protocol markdown hash mismatch")
    a.require(sha256(OUT / "protocol.json") == manifest["protocol_sha256"],
              "protocol JSON hash mismatch")
    a.require(sha256(ROOT / "code" / "review_round2.py") ==
              manifest["source_sha256"], "driver source hash mismatch")
    a.require(manifest["status"] == "PASS", "manifest is not PASS")

    actual_files = {str(p.relative_to(ROOT)) for p in OUT.rglob("*")
                    if p.is_file() and not p.name.endswith(".tmp")
                    and p.name != "manifest.json"}
    listed_files = set(manifest["artifacts"])
    a.require(actual_files == listed_files,
              f"manifest inventory mismatch: extra={actual_files-listed_files}, "
              f"missing={listed_files-actual_files}")
    for rel, digest in sorted(manifest["artifacts"].items()):
        a.require(sha256(ROOT / rel) == digest, f"artifact hash mismatch: {rel}")

    rows: dict[str, list[tuple[Path, dict]]] = {}
    for family in FAMILIES:
        family_rows = []
        for path in sorted((OUT / family).glob("*.json")):
            if path.name not in {"summary.json", "sensitivity.json"}:
                family_rows.append((path, read_json(path)))
        rows[family] = family_rows
        a.require(len(family_rows) == EXPECTED[family],
                  f"wrong {family} count")
    a.require(sum(map(len, rows.values())) == manifest["n_trace_records"] == 89,
              "trace-record total mismatch")
    a.require(summary["observed_counts"] == EXPECTED ==
              summary["expected_counts"], "summary family counts mismatch")
    a.require(summary["complete"] is True, "summary is incomplete")

    seen_conditions = set()
    total_events = 0
    for family, family_rows in rows.items():
        n_free_by_map = {}
        for record_path, rec in family_rows:
            cond = rec["condition"]
            a.require(cond["family"] == family,
                      f"family mismatch in {record_path}")
            condition_key = json.dumps(cond, sort_keys=True,
                                       separators=(",", ":"))
            a.require(condition_key not in seen_conditions,
                      f"duplicate condition: {record_path}")
            seen_conditions.add(condition_key)
            a.require(close(rec["throughput_tasks_per_step"],
                            rec["completed"] / cond["steps"]),
                      f"throughput mismatch: {record_path}")
            trace_path = ROOT / rec["trace"]["path"]
            a.require(sha256(trace_path) == rec["trace"]["sha256"],
                      f"trace hash mismatch: {trace_path}")
            with np.load(trace_path, allow_pickle=False) as data:
                pos = data["position_trace"]
                goals = data["goal_trace"]
                events = data["goal_events"]
                a.require(pos.shape == tuple(rec["trace"]["position_shape"]),
                          f"position shape mismatch: {trace_path}")
                a.require(goals.shape == tuple(rec["trace"]["goal_shape"]),
                          f"goal shape mismatch: {trace_path}")
                a.require(pos.shape == goals.shape ==
                          (cond["steps"] + 1, cond["N"]),
                          f"trace contract mismatch: {trace_path}")
                a.require(events.shape == (rec["completed"], 7),
                          f"event shape mismatch: {trace_path}")
                a.require(json.loads(str(data["condition_json"])) == cond,
                          f"embedded condition mismatch: {trace_path}")
                n_free = n_free_by_map.setdefault(cond["map"],
                                                    free_cells(cond["map"]))
                a.require(bool(np.all((pos >= 0) & (pos < n_free))),
                          f"position index out of range: {trace_path}", pos.size)
                a.require(bool(np.all((goals >= 0) & (goals < n_free))),
                          f"goal index out of range: {trace_path}", goals.size)
                a.state_entries += pos.size + goals.size
                if len(events):
                    t, agent, assigned, delay, moves, origin, goal = events.T
                    a.require(bool(np.all((t >= 1) & (t <= cond["steps"]))),
                              f"event time out of range: {trace_path}", len(t))
                    a.require(bool(np.all((agent >= 0) & (agent < cond["N"]))),
                              f"event agent out of range: {trace_path}", len(t))
                    a.require(bool(np.all(t - assigned == delay)),
                              f"event delay mismatch: {trace_path}", len(t))
                    a.require(bool(np.all(moves <= delay)),
                              f"event moves exceed delay: {trace_path}", len(t))
                    a.require(bool(np.all(pos[t, agent] == goal)),
                              f"completion position mismatch: {trace_path}", len(t))
                    a.require(bool(np.all(pos[assigned, agent] == origin)),
                              f"task origin mismatch: {trace_path}", len(t))
                    a.require(bool(np.all(goals[assigned, agent] == goal)),
                              f"assigned goal mismatch: {trace_path}", len(t))
                    a.event_entries += len(events)
                total_events += len(events)
                tm = rec["trace_metrics"]
                counts = []
                for start in range(0, cond["steps"], 500):
                    counts.append(int(np.sum((events[:, 0] > start) &
                                             (events[:, 0] <=
                                              min(start + 500, cond["steps"])))))
                a.require(counts == tm["completed_per_window"],
                          f"window counts mismatch: {trace_path}", len(counts))
                a.require(sum(counts) == rec["completed"],
                          f"events do not sum to completions: {trace_path}")

    a.require(total_events == 2_878_327, "unexpected completed-event total")

    # Recompute every released group aggregate without using driver helpers.
    for family, family_rows in rows.items():
        by_label = defaultdict(list)
        by_map_label = defaultdict(list)
        for _, rec in family_rows:
            label = rec["condition"]["label"]
            by_label[label].append(rec)
            by_map_label[f"{rec['condition']['map']}::{label}"].append(rec)
        for view_name, groups in (("by_label", by_label),
                                  ("by_map_label", by_map_label)):
            released = summary["families"][family][view_name]
            a.require(set(released) == set(groups),
                      f"summary groups differ: {family}/{view_name}")
            for label, values in groups.items():
                out = released[label]
                throughput = [r["throughput_tasks_per_step"] for r in values]
                tails = [r["trace_metrics"]["last_half_over_whole"]
                         for r in values]
                a.require(out["n"] == len(values),
                          f"group n mismatch: {family}/{label}")
                a.require(close(out["throughput_mean"],
                                np.mean(throughput)),
                          f"group throughput mismatch: {family}/{label}")
                a.require(close(out["last_half_over_whole_mean"],
                                np.mean(tails)),
                          f"group tail mismatch: {family}/{label}")
                a.require(out["collapse_runs"] == sum(bool(
                          r["trace_metrics"]["collapse_window_indices"])
                          for r in values),
                          f"group collapse mismatch: {family}/{label}")

    search = summary["search"]
    train = np.asarray(search["training_runs"], float)
    validation = np.asarray(search["validation_runs"], float)
    a.require(train.shape == validation.shape == (49, 2),
              "search candidate ledger shape mismatch")
    a.require(int(np.argmax(train.mean(axis=1))) ==
              search["training_selected_index"] == 0,
              "training selection mismatch")
    a.require(int(np.argmax(validation.mean(axis=1))) ==
              search["validation_selected_index"] == 0,
              "validation selection mismatch")
    selected_path = OUT / "search" / "selected.npz"
    with np.load(selected_path, allow_pickle=False) as data:
        a.require(np.array_equal(data["initial_theta"],
                                 data["validation_selected_theta"]),
                  "validation-selected theta is not retained start",
                  data["initial_theta"].size)
        a.require(np.array_equal(data["initial_weights"],
                                 data["selected_weights"]),
                  "validation-selected weights are not retained start",
                  data["initial_weights"].size)
    initial_tests = [r for _, r in rows["search_test"]
                     if r["condition"]["label"] == "retained_initial"]
    selected_tests = [r for _, r in rows["search_test"]
                      if r["condition"]["label"] == "validation_selected"]
    a.require([r["completed"] for r in initial_tests] ==
              [r["completed"] for r in selected_tests],
              "test outputs differ for identical selected field")

    sensitivity = read_json(OUT / "warehouse_resolution" /
                            "sensitivity.json")
    fields = {}
    for k in (32, 64, 128, 256):
        path = next((OUT / "guidance").glob(
            f"warehouse-10-20-10-2-1__N800__g2__lane__K{k}.npz"))
        with np.load(path, allow_pickle=False) as data:
            fields[k] = np.asarray(data["weights"], float)
    ref = fields[256] / fields[256].mean()
    for k, field in fields.items():
        normalized = field / field.mean()
        rel = np.linalg.norm(normalized - ref) / max(np.linalg.norm(ref),
                                                     1e-12)
        corr = np.corrcoef(normalized, ref)[0, 1]
        a.require(close(rel, sensitivity[str(k)]["relative_l2_to_K256"]),
                  f"warehouse L2 mismatch: K{k}")
        a.require(close(corr, sensitivity[str(k)]["pearson_to_K256"]),
                  f"warehouse correlation mismatch: K{k}")

    for _, rec in rows["stale"]:
        mode = rec["condition"]["label"]
        for event in rec["update_events"]:
            if mode == "synchronous":
                a.require(event["applied_at_step"] ==
                          event["requested_at_step"],
                          "synchronous update was delayed")
            elif mode == "stale":
                expected_delay = math.ceil(event["compute_and_rebuild_s"] /
                                           event["control_period_s"])
                a.require(event["measured_delay_steps"] == expected_delay,
                          "stale measured delay mismatch")
                a.require(event["applied_at_step"] -
                          event["requested_at_step"] == expected_delay,
                          "stale application delay mismatch")

    long_cells = summary["families"]["long_horizon"]["by_map_label"]
    a.require(long_cells["empty-32-32::tolls"]["throughput_mean"] >
              long_cells["empty-32-32::unweighted"]["throughput_mean"],
              "expected open-map positive result missing")
    a.require(long_cells["maze-32-32-4::tolls"]["throughput_mean"] <
              long_cells["maze-32-32-4::unweighted"]["throughput_mean"],
              "maze negative result was lost")
    a.require(long_cells[
              "warehouse-10-20-10-2-1::tolls"]["throughput_mean"] <
              long_cells[
              "warehouse-10-20-10-2-1::unweighted"]["throughput_mean"],
              "warehouse negative result was lost")

    return {
        "status": "PASS",
        "checks": a.checks,
        "trace_records": manifest["n_trace_records"],
        "artifact_hashes": len(manifest["artifacts"]),
        "state_entries_checked": a.state_entries,
        "completed_events_checked": a.event_entries,
        "protocol_sha256": manifest["protocol_sha256"],
        "summary_sha256": sha256(OUT / "summary.json"),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "negative_outcomes_retained": True,
        "driver_imported": False,
    }


def main() -> None:
    report = audit()
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
