# Provenance of the shipped results

`results/provenance_manifest.json` binds every shipped source file, input, model weight,
result file and figure by SHA-256 and records each file's byte size, filesystem
modification time (UTC), role and execution-contract label. It lists exactly the files of
this package; numerical caches are not shipped and not listed. Regenerate or verify it with:

```sh
python code/artifact_manifest.py          # rewrite the manifest (run last)
python code/artifact_manifest.py --check  # verify the shipped bytes against it
```

The snapshot has no version-control revision identifier, so the manifest does not invent
one. The hashes are the reproducible identity of the available bytes. Historical
experiment start/end times, dependency lock data and host-image identifiers were not
recorded for the legacy runs and remain unavailable.

## External data

The five ETH/UCY annotation files in `results/data/` (`biwi_eth.txt`, `biwi_hotel.txt`,
`students003.txt`, `crowds_zara01.txt`, `crowds_zara02.txt`) came through the Trajectron++
raw-data mirror whose base URL is recorded in `code/exp0_data.py` and in the manifest.
Their exact bytes are hashed. Original download timestamps were not saved.

Historical leave-one-scene-out (LOSO) predictor training derived its seed from Python's
process-randomized built-in `hash`, so those numeric training seeds cannot be recovered;
the shipped model hashes are the exact identifiers of those weights. From-scratch
training now uses deterministic CRC32 seeds and records each fold seed.

## Current-contract evidence

`results/window_anchor_smoke.json` is a small current-contract schema record. It stores
exact seeds, a hash of the generated warm-score vector, source hashes, dependency
versions, start/end timestamps and all per-window/per-step labels. Its purpose is
contract checking, not performance evidence.

The authoritative evidence for the current controller is isolated under
`results/window_anchor_v1_study/`. `locked_spec.json` pre-enumerates all 320 attempts and
binds code, frozen pools, predictors, replay inputs, risk semantics, seed/phase rules and
the confidence-lifetime policy. `ledger.json` records an atomic started/terminal history
for every attempt. Each `attempts/*.json` file binds a full compressed `traces/*.npz`
companion; the arrays retain robot and agent states, all five simulator substeps, fresh
and constrained predictions, anchors/scales, planned/realized endpoints and collision
labels. The strict validator `code/check_revised_anchor_study.py` recomputes rather than
trusts these labels. `same_frozen_stream.json` is a separate literal identical-score-stream
comparison; no legacy robot-control outcome enters it.

## Contract classification

- `window_anchor_v1` is the current implementation in `code/envloop.py` and
  `code/planner.py`. Its schema records predictor identity, prediction-update steps,
  constraint-anchor steps, score observation steps, scores/radii, anchored-constraint
  feasibility, containment, tracking-bound status, score-selected agent identities and
  all-clean status, selected-agent endpoint-clearance implication, fallback reasons and
  observed substep-collision labels. `code/test_core.py` exercises this contract.
- `results/window_anchor_v1_study/` is the completed current-contract study. Its 40
  episodes per cell share a frozen pool/model and are descriptive; it supplies neither
  independent replications of the statistical event nor a portfolio confidence
  guarantee. All negative collision/fallback/containment outcomes remain in the ledger.
- `legacy_fresh_prediction_recentered_v0` labels the planning results in `results/main`,
  `results/long`, `results/vehicle`, `results/matched` and the derived planning summaries
  (`exp6_ablations.json`, `exp9_matched.json`, `exp10_vehicle.json`, `summary.json`).
  Those runs replaced prediction centers within a scored window and do not contain the
  timestamps or labels needed to audit the current contract. Their
  throughput/collision fields are not evidence for `window_anchor_v1`.
- `statistical_only_no_control_contract` labels score-only simulations and reanalyses
  (`exp1_validity.json`, `exp2_*`, `exp7_budget.json` synthetic rows, `exp8_robust.json`,
  `exp12_rho_loso.json`, `dependence_aware_long_real.json`). They support statistical and
  rank statements but no robot-execution statement.

Generated figures inherit the semantics and limitations of their source results; figures
that mix contracts are descriptive. No missing legacy field has been reconstructed from a
file name or from current code.
