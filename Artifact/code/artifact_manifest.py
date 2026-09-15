"""Create a byte-level provenance manifest without rerunning experiments.

The manifest deliberately distinguishes statistical-only artifacts from planning
records produced under the legacy recentering controller.  Missing historical
metadata is reported as missing; it is never reconstructed from file names.
Files are hashed in 1 MiB chunks so this audit has constant memory use.

    python code/artifact_manifest.py          # write results/provenance_manifest.json
    python code/artifact_manifest.py --check  # verify the shipped bytes against it
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "provenance_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(
        path.stat().st_mtime, tz=dt.timezone.utc
    ).isoformat().replace("+00:00", "Z")


def role(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("code/"):
        return "source_code"
    if rel == "run_checks.sh":
        return "source_code"
    if rel == "requirements.txt":
        return "environment"
    if rel.startswith("results/figures/"):
        return "derived_figure"
    if rel.startswith("results/data/"):
        return "data_input"
    if rel.startswith("results/models/"):
        return "model_input"
    if rel.startswith("results/cache/"):
        return "numerical_cache"
    if rel.startswith("results/"):
        return "result_artifact"
    return "documentation"


def contract(path: Path) -> str | None:
    """Classify artifact semantics, without inferring unavailable run metadata."""
    rel = path.relative_to(ROOT).as_posix()
    legacy_prefixes = (
        "results/main/", "results/long/", "results/vehicle/",
    )
    legacy_files = {
        "results/exp6_ablations.json", "results/exp9_matched.json",
        "results/exp10_vehicle.json", "results/summary.json",
    }
    statistical_files = {
        "results/exp1_validity.json", "results/exp2_epsdelta.csv",
        "results/exp2_headline.json", "results/exp2_horizon.csv",
        "results/exp2_radius.csv", "results/exp7_budget.json",
        "results/exp8_robust.json", "results/exp12_rho_loso.json",
        "results/dependence_aware_long_real.json",
    }
    if rel == "results/window_anchor_smoke.json":
        return "window_anchor_v1"
    if rel.startswith("results/window_anchor_v1_study/"):
        return "window_anchor_v1"
    if rel in {"results/figures/fig_core.pdf", "results/figures/fig_core.png"}:
        return "statistical_only_no_control_contract"
    if rel in {"results/exp7_budget.json",
               "results/figures/fig_anytime.pdf",
               "results/figures/fig_anytime.png"}:
        return "mixed_statistical_and_legacy_recentered_sources"
    if rel.startswith("results/figures/"):
        return "legacy_fresh_prediction_recentered_v0"
    if rel.startswith(legacy_prefixes) or rel in legacy_files:
        return "legacy_fresh_prediction_recentered_v0"
    if rel in statistical_files:
        return "statistical_only_no_control_contract"
    return None


def iter_files():
    selected = [
        ROOT / "README.md", ROOT / "PROVENANCE.md", ROOT / "requirements.txt",
        ROOT / "run_checks.sh",
    ]
    selected.extend(sorted((ROOT / "code").glob("*.py")))
    selected.extend(sorted((ROOT / "code").glob("*.sh")))
    selected.extend(sorted((ROOT / "results").rglob("*")))
    seen: set[Path] = set()
    for path in selected:
        if path == OUT or path in seen or not path.is_file():
            continue
        if path.name == ".DS_Store":
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("results/cache/") or "/boundary_cache/" in rel:
            continue  # numerical caches, rebuilt on demand by csq.cs_boundary
        seen.add(path)
        yield path


def check() -> int:
    """Verify every file listed in the manifest against its recorded size and SHA-256."""
    if not OUT.exists():
        print(f"FAIL {OUT.relative_to(ROOT)} is missing")
        return 1
    entries = json.loads(OUT.read_text(encoding="utf-8"))["files"]
    bad = []
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file():
            bad.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            bad.append(f"changed: {entry['path']}")
    for item in bad[:20]:
        print("  -", item)
    if bad:
        print(f"FAIL provenance manifest: {len(bad)} of {len(entries)} files missing or changed")
        return 1
    print(f"PASS provenance manifest: {len(entries)} files match their recorded SHA-256")
    return 0


def main() -> None:
    entries = []
    contracts: dict[str, int] = {}
    for path in iter_files():
        c = contract(path)
        if c:
            contracts[c] = contracts.get(c, 0) + 1
        entries.append({
            "path": path.relative_to(ROOT).as_posix(),
            "role": role(path),
            "bytes": path.stat().st_size,
            "mtime_utc": iso_mtime(path),
            "sha256": sha256(path),
            "execution_contract": c,
        })

    payload = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "generator": "code/artifact_manifest.py",
        "repository_revision": None,
        "repository_revision_note": (
            "No project-local VCS revision was available; SHA-256 hashes bind "
            "the exact audited bytes."
        ),
        "current_execution_contract": "window_anchor_v1",
        "new_record_required_fields": [
            "predictor.model_id", "predictor_update_steps", "constraint_anchor_steps",
            "window_audit.prediction_issued_step",
            "window_audit.score_observed_step",
            "window_audit.score", "window_audit.radius",
            "window_audit.containment_violation",
            "window_audit.score_selected_agent_indices",
            "window_audit.score_selected_agent_uids",
            "window_audit.identity_transition_records",
            "window_audit.all_window_agents_in_score_event",
            "window_audit.finite_sample_radius_available_under_declared_assumptions",
            "tracking_contract", "statistical_contract.assumption_status",
            "fallback_labels", "collision_step_labels",
            "anchored_constraint_feasible_labels", "tracking_bound_labels",
            "containment_labels_by_control_step",
            "selected_agent_endpoint_clearance_implication_labels",
            "identity_policy_contract", "tracking_stress",
            "fallback_steps_warmup", "fallback_steps_active_window",
            "path_length_m", "signed_waypoint_progress_m",
            "control_audit_trace.constraint_prediction_sha256",
            "control_audit_trace.observed_agent_uids_at_plan",
        ],
        "contract_counts": contracts,
        "limitations": [
            "Legacy planning records lack anchor/update timestamps and per-step "
            "certificate/fallback labels, so those fields cannot be recovered.",
            "The manifest records current file mtimes, which are not necessarily "
            "the original experiment start/end timestamps.",
            "Planning artifacts outside results/window_anchor_v1_study use the "
            "legacy recentered contract and do not validate window_anchor_v1.",
            "The fresh study is simulation-only, shares one frozen pool/model "
            "per regime, and supplies no portfolio confidence guarantee.",
            "No dependency lockfile or original host image digest accompanied "
            "the legacy runs.",
            "Historical LOSO models were trained before the generator replaced "
            "Python's process-randomized hash seed with deterministic CRC32; "
            "their exact training seeds are unavailable, so model-file hashes "
            "are the only exact identifiers for those weights.",
        ],
        "external_data_source": {
            "description": "ETH/UCY text annotations via the Trajectron++ mirror",
            "base_url": (
                "https://raw.githubusercontent.com/StanfordASL/"
                "Trajectron-plus-plus/master/experiments/pedestrians/raw/raw/"
                "all_data"
            ),
            "raw_file_hashes_are_in_manifest": True,
            "original_download_timestamps_available": False,
        },
        "claim_sources": {
            "uniform_mixture_rate": ["code/csq.py", "code/test_core.py",
                                     "code/audit_new.py"],
            "exact_harmonic_cutoff": [
                "code/csq.py", "code/test_core.py", "results/exp2_horizon.csv"
            ],
            "dependence_aware_long_real": [
                "code/audit_dependence.py",
                "results/dependence_aware_long_real.json",
                "results/long/",
            ],
            "held_out_shift_diagnostic": [
                "code/exp12_rho_loso.py", "results/exp12_rho_loso.json"
            ],
            "window_anchor_contract": [
                "code/envloop.py", "code/planner.py", "code/test_core.py",
                "code/run_revised_anchor_study.py",
                "code/check_revised_anchor_study.py",
                "results/window_anchor_v1_study/"
            ],
            "same_g1_alpha_spending_comparator": [
                "code/csq.py", "code/test_core.py",
                "results/window_anchor_v1_study/same_frozen_stream.json"
            ],
        },
        "files": entries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(entries)} files)")


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        raise SystemExit(check())
    main()
