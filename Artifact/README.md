# Anytime-valid calibration for crowd-navigation planning

## Overview

This package studies how to calibrate the radius by which a crowd-navigation planner
inflates predicted pedestrian positions, so that its statistical statement survives
replanning at data-dependent times. `code/csq.py` implements three separately stated
contracts: G1, a time-uniform (anytime-valid) beta-binomial-mixture confidence sequence for
a quantile of the prediction-error score (per-window risk); G2, an anytime bound on the
cumulative number of violated windows; and G3, episode-level "any violation" calibrators
(union-bound conformal prediction, AUC, AUC-B) whose finite-sample rank horizon is
computed exactly. These sit next to adaptive-conformal, Gaussian, uncalibrated and
exact-binomial alpha-spending comparators. Experiments use two compact 2-D crowd
simulators (social force and an ORCA-style model), recorded agent-only replays and five
ETH/UCY pedestrian scenes, a constant-velocity predictor with a small GRU residual head,
and a sampling-based receding-horizon planner for a double integrator (and a unicycle).
The current execution contract, `window_anchor_v1`, constrains every control inside a
window against the suffix of the window-start prediction; a predeclared 320-attempt study
reports its fallback, containment, tracking and observed-collision outcomes as simulated,
descriptive evidence. Statistical validity of the tube is not collision safety, and the
older planning results (`results/main`, `long`, `matched`, `vehicle`) come from a legacy
controller that recentered predictions within a window (see `PROVENANCE.md`).

## Contents

```
README.md                  this file
PROVENANCE.md              data lineage and execution-contract label of every result folder
requirements.txt           pinned Python packages
run_checks.sh              quick check
code/
  csq.py                   confidence sequences (G1), violation budget (G2), G3 calibrators,
                           exact rank horizons, baselines
  envloop.py, planner.py   episode runner with the window_anchor_v1 contract; sampling planner
  predictor.py, sims.py    constant-velocity + GRU predictor; social-force and ORCA-style simulators
  np2compat.py             lets NumPy 1.26 load the shipped pickles written under NumPy 2.x
  status_contract.py       exported planner-status labels
  test_core.py             unit tests
  exp0_data.py             simulator rollouts, GRU training, calibration scores, ETH/UCY scenes
  exp1_validity.py         synthetic time-uniform validity
  exp2_radius.py           radius and certifiable-horizon curves
  exp3_planning.py         8-window planning conditions (legacy controller)
  exp5_longrun.py          120-window deployments (legacy controller)
  exp6_ablations.py        predictor and planner ablations
  exp7_budget.py           G2 cumulative-violation budget
  exp8_robust.py           shift-robust confidence sequence
  exp9_matched.py          rank wall, lookahead wall, matched-calibration arm
  exp10_vehicle.py         unicycle and shift-robust stress tests
  exp12_rho_loso.py        held-out shift-budget diagnostic
  exp14_closedloop_shift.py  shift pricing on the window_anchor_v1 study records
  run_revised_anchor_study.py, check_revised_anchor_study.py
                           locked 320-attempt window_anchor_v1 study and its strict validator
  collision_attribution.py, contract_smoke.py, make_lifetime_example.py,
  make_regression_examples.py
                           analyses of the study records, smoke episode, standalone examples
  audit_new.py             audit of stored values in exp7-exp10 and episode totals
  audit_dependence.py      dependence-aware intervals for the long real-data runs
  make_summary.py, update_summary.py   results/summary.json
  exp4_figures.py, exp11_figures.py, exp13_core_fig.py, redraw_figures.py   figures
  check_regenerated_outputs.py   regenerate-and-compare step of run_checks.sh
  artifact_manifest.py     writes or verifies results/provenance_manifest.json
  run_all.sh               full pipeline
results/
  data/                    calibration pools, replays, ETH/UCY annotations and derived scenes (28 MB)
  models/                  seven GRU predictor weights (80 KB)
  window_anchor_v1_study/  locked_spec.json, ledger.json, summary.json, same_frozen_stream.json,
                           attempts/ (320 JSON records), traces/ (320 NPZ full traces)  (100 MB)
  main/ (144 files), long/ (48), matched/ (5), vehicle/ (20)   legacy planning conditions
  closedloop_shift/        exp14 protocol and summary
  checks/                  lifetime_example.json, regression_examples.json
  figures/                 PDF and PNG figures
  exp1_validity.json, exp2_*.csv, exp2_headline.json, exp6-exp10 and exp12 JSON, summary.json,
  collision_attribution.json, dependence_aware_long_real.json, window_anchor_smoke.json,
  provenance_manifest.json (SHA-256, size and contract label of every shipped file)
```

## Setup

Python 3.12 (tested with 3.12.12 on macOS arm64, CPU only):

```sh
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

No compiler, GPU or system library is needed; PyTorch runs on the CPU. Every input is
included, so the quick check needs no download. Only `code/exp0_data.py` uses the network,
and only when it has to rebuild the ETH/UCY scenes (see Data).

## Quick check

```sh
sh run_checks.sh                            # interpreter: $PYTHON, default python3
PYTHON=.venv/bin/python sh run_checks.sh    # works from any working directory
```

About 35 s on one core (it sets `OMP_NUM_THREADS=1` unless already set). Exit status is
non-zero if any step fails. Steps:

1. `code/artifact_manifest.py --check`: every file listed in
   `results/provenance_manifest.json` still has its recorded size and SHA-256.
2. `code/test_core.py`: 92 unit checks, e.g. exact e-value mean, the uniform-mixture
   `sqrt(log t / t)` rate, the exact harmonic cutoff (including `n=1, delta=0.5`),
   permanent budget exhaustion, the alpha-spending comparator, the anchored prediction
   suffix, the tracking margin, identity handling and rejection of stale predictions.
3. `code/check_revised_anchor_study.py`: 23 gates. It recomputes every record/trace hash,
   window score, anchor suffix, identity transition, tracking error, substep collision
   label and path length of the 320 attempts from the NPZ traces, replays the literal
   same-stream comparison, and checks that the seven study sources and six inputs still
   match the SHA-256 values in `locked_spec.json`.
4. `code/make_regression_examples.py --example anchor|identity|tracking|rank`: four
   standalone examples with predeclared expectations.
5. `code/check_regenerated_outputs.py`: in a temporary copy (the shipped files are never
   modified) it re-runs 13 steps (`audit_new.py`, `exp2_radius.py`,
   `exp7_budget.py --reuse-synthetic`, `exp12_rho_loso.py`, `audit_dependence.py`,
   `exp14_closedloop_shift.py run`, `collision_attribution.py`, `make_lifetime_example.py`,
   `make_regression_examples.py`, `contract_smoke.py`, `make_summary.py`,
   `update_summary.py`, `redraw_figures.py`) and compares 18 regenerated files with the
   shipped ones: JSON value by value (ignoring only generation timestamps and the
   documented exceptions in Notes), CSV byte for byte, PNG by decoded pixels. Set
   `SKIP_FIGURES=1` to skip the PNG comparison on a platform with a different matplotlib
   build.

## Full reproduction

`sh code/run_all.sh` (optionally `PYTHON=... WORKERS=4`) runs the steps below in order.
The long experiments are time-boxed passes that checkpoint each condition; the script
repeats a pass until it reports completion, and re-running it resumes after an
interruption. Started on the shipped results, `exp0_data.py` and the checkpointed runners
(steps 4, 5, 9, 10 and 13) skip everything already complete, while the other steps
recompute their outputs in place. To regenerate the result folders themselves:

- exact re-run on the shipped inputs: move everything in `results/` except `data/` and
  `models/` aside, then run the pipeline.

  ```sh
  mkdir -p shipped_results
  for d in results/*; do
    case "$d" in results/data|results/models) ;; *) mv "$d" shipped_results/ ;; esac
  done
  sh code/run_all.sh
  ```

- rebuild the inputs as well: `mv results shipped_results && sh code/run_all.sh`.
  `exp0_data.py` then downloads the five ETH/UCY files and retrains all seven GRU
  predictors. New predictor bytes produce a new study lock, and numbers are expected to
  differ somewhat from the shipped ones (the historical leave-one-scene-out training seeds
  are unknown).

| step | command (in `code/`) | writes (under `results/`) | cores | wall time |
|---|---|---|---|---|
| 0 | `exp0_data.py` | `data/`, `models/` | 1 | minutes (CPU GRU training, 7 predictors) |
| 1 | `test_core.py` | nothing | 1 | 8 s |
| 2 | `exp1_validity.py` | `exp1_validity.json` | 1 | about 1 min |
| 3 | `exp2_radius.py` | `exp2_radius.csv`, `exp2_horizon.csv`, `exp2_epsdelta.csv`, `exp2_headline.json` | 1 | 1 s |
| 4 | `exp3_planning.py --minutes 8.5 --workers W`, repeated | `main/` (144 condition files) | W | several 8.5-min passes |
| 5 | `exp5_longrun.py --minutes 10 --workers W`, repeated | `long/` (48 condition files) | W | several 10-min passes |
| 6 | `exp6_ablations.py` | `exp6_ablations.json` | 1 | minutes |
| 7 | `exp7_budget.py` | `exp7_budget.json` | 1 | minutes (4 x 20,000 synthetic 3,000-window runs) |
| 8 | `exp8_robust.py` | `exp8_robust.json` | 1 | minutes |
| 9 | `exp9_matched.py --workers W`, repeated | `matched/`, `exp9_matched.json`, `data/cal_sf_matched.npy` | W | 25-min passes |
| 10 | `exp10_vehicle.py --workers W`, repeated | `vehicle/`, `exp10_vehicle.json` | W | 20-min passes |
| 11 | `exp12_rho_loso.py` | `exp12_rho_loso.json` | 1 | 1 s |
| 12 | `audit_dependence.py` | `dependence_aware_long_real.json` | 1 | 1 s |
| 13 | `run_revised_anchor_study.py`, `check_revised_anchor_study.py` | `window_anchor_v1_study/` | 1 (serial by design) | 108 s for 320 attempts, then 3 s |
| 14 | `exp14_closedloop_shift.py protocol`, then `run` | `closedloop_shift/` | 1 | 1 s |
| 15 | `collision_attribution.py`, `contract_smoke.py`, `make_lifetime_example.py`, `make_regression_examples.py` | `collision_attribution.json`, `window_anchor_smoke.json`, `checks/` | 1 | 1-2 s each |
| 16 | `redraw_figures.py` | `figures/` | 1 | 5 s |
| 17 | `make_summary.py`, `update_summary.py`, `audit_new.py` | `summary.json` | 1 | 1 s, 1 s, 6 s |
| 18 | `artifact_manifest.py` | `provenance_manifest.json` | 1 | 2 s |

`W` is `$WORKERS` (default 3). Times given in seconds were measured for this package on
one core; the 108 s of step 13 come from the ledger timestamps of the shipped run. Steps 0,
2 and 4-10 recorded no start/end times and were not re-run here (compute budget); the
estimates in the table are rough and come from the project's earlier notes (about 2 min for
the simulator part of step 0, about 1 min for step 2, and about 15 min with 3 workers for a
first, smaller version of step 4) and from each runner's time box. The step-13 runner refuses to resume
if any locked source or input changed; delete `results/window_anchor_v1_study/` to start a
new lock.

## Data

- **Simulator data** (generated by `exp0_data.py` from `sims.py` with fixed seeds; no
  external source): `cal_sf.npy`, `cal_orca.npy` (2,094 calibration window scores each),
  `replay_sf.pkl`, `replay_orca.pkl` (100 agent-only replay rollouts each),
  `replay_long_sf.pkl`, `replay_long_orca.pkl` (40 long rollouts each, 12 MB together),
  `cal_sf_matched.npy` (exp9 matched pool). The first four, together with
  `models/gru_sf.pt` and `models/gru_orca.pt`, are the SHA-256-locked inputs of the
  window_anchor_v1 study (2.2 MB).
- **ETH/UCY pedestrian annotations** (third-party): `biwi_eth.txt`, `biwi_hotel.txt`
  (ETH/BIWI Walking Pedestrians), `students003.txt`, `crowds_zara01.txt`,
  `crowds_zara02.txt` (UCY Crowds); 1.6 MB of `frame ped x y` rows at 2.5 Hz, obtained
  from the Trajectron++ raw-data mirror. Licence: none is stated in the mirror; the
  annotations remain subject to the terms of their original ETH and UCY providers.
  Derived from them by `exp0_data.py`: `real_<scene>.pkl` (interpolated, rotated, chunked
  scenes, 12 MB), `cal_real_<scene>.npy` and `.npy.json` (leave-one-scene-out pools),
  `cal_eth.npy` and `replay_eth.pkl` (single-scene pilot), `meta.json`. To fetch the raw
  files yourself into `results/data/` (their SHA-256 values are in
  `results/provenance_manifest.json`):

  ```sh
  for f in biwi_eth.txt biwi_hotel.txt students003.txt crowds_zara01.txt crowds_zara02.txt; do
    curl -fsSL -o "results/data/$f" \
      "https://raw.githubusercontent.com/StanfordASL/Trajectron-plus-plus/master/experiments/pedestrians/raw/raw/all_data/$f"
  done
  ```

- **Models**: `results/models/gru_sf.pt`, `gru_orca.pt` and the five leave-one-scene-out
  predictors `gru_real_<scene>.pt` (about 11 KB each). Retraining does not reproduce these
  bytes, so they are shipped.
- **Not shipped**: `results/cache/` (31 MB of cached confidence-sequence boundary arrays)
  and `results/closedloop_shift/boundary_cache/` (0.2 MB). `csq.cs_boundary` rebuilds them
  on first use; for example `audit_new.py` rebuilds its 2,000,000-entry boundary in a few
  seconds.

## Expected results

- **window_anchor_v1 study** (`results/window_anchor_v1_study/summary.json`, `cells`):
  320/320 predeclared attempts completed, no failed or interrupted execution, 8 cells of
  40 episodes. Across cells: active-window fallback fraction 0.567-0.662; score-selected
  agent fraction 0.793-0.802; endpoint-clearance implication on 0.240-0.291 of active
  controls; finite-radius containment violations in 0.056-0.178 of windows;
  observed-collision episode fraction 0.025-0.175; zero tracking-bound violations (maximum
  tracking error 0.10 m in the 0.10 m cells).
- **Same frozen score stream** (`same_frozen_stream.json`, 100 scores per regime): mean
  radius of the mixture CS vs the alpha-spending CS 1.417 vs 1.461 m (social force) and
  1.585 vs 1.649 m (ORCA-style).
- **Collision attribution** (`results/collision_attribution.json`): 79 colliding controls
  among 20,480 active controls (five warm-up controls per attempt excluded); 75 under
  fallback (0.598% of 12,550 fallback controls) and 4 under anchored control (0.050% of
  7,930); first contact happened under fallback in 25 of the 29 collision episodes.
- **Closed-loop shift pricing** (`results/closedloop_shift/summary.json`): all 2,560
  windows have a finite radius at `rho_tot=0`, with 288 domination failures (11.25%);
  146 of 320 attempts have none; a budget on the grid removes them for 7 attempts and not
  for 167; the failure share is still 10.39% at `rho_tot=5`.
- **Rank horizons** (`results/exp2_headline.json`, n=2,094, delta=0.05; also
  `results/checks/regression_examples.json`): 104 certifiable windows for union CP with
  any allocation, 107 for the exact harmonic cutoff, 9 for AUC, 52 for AUC-B
  (lambda=0.05); unbounded for the CS, whose final radius is 1.721 m vs the plug-in
  quantile 1.616 m (+6.5%).
- **Synthetic validity** (`results/exp1_validity.json`): CS anytime undercoverage 0.0421
  (beta=0.9, delta=0.05) and 0.0883 (beta=0.9, delta=0.1); a plug-in quantile recomputed
  at every step fails in 0.951-0.966 of runs. Over 8-window episodes at delta=0.05:
  union CP 0.048, AUC 0.044, AUC-B 0.018, episode-targeted ACI 0.049, ACI 0.340.
- **G2 budget and robustness** (`results/exp7_budget.json`, `exp8_robust.json`, checked
  by `audit_new.py`): budget B_W = 2, 15, 77 at W = 8, 120, 1000 (eps=delta'=0.05);
  synthetic anytime exceedance 0.021, 0.073, 0.030, 0.060 for (eps, delta') = (0.05, 0.05),
  (0.05, 0.1), (0.1, 0.05), (0.1, 0.1). Under a persistent +0.01 shift the plain CS fails
  in 0.51 of runs and the shift-robust CS in 0.000.
- **Shift-budget transfer** (`results/exp12_rho_loso.json`): the held-out point transfer
  covers 9 of 10 cases and fails for `real_zara1` at delta=0.1 (8.71 supplied vs 12.0
  needed); the conservative source upper bounds cover 10 of 10.
- **Long real-data runs** (`results/dependence_aware_long_real.json`,
  `long_real_cs_0.05`): certified-window violation rate 0.025, episode-cluster 95% interval
  [0.005, 0.052], scene-cluster interval [0.001, 0.076], leave-one-scene-out range
  [0.014, 0.039].
- **Stored-value audit** (`audit_new.py`): 73 values in `exp7`-`exp10` plus totals of
  15,758 episodes in 217 condition files; e.g. matched pool n=2,597 with a 129-window rank
  wall (`exp9_matched.json`).

## Notes

- **Seeds and determinism.** Every experiment seeds NumPy `default_rng` (PCG64) with fixed
  integers or CRC32-derived values: study episode seeds come from a SHA-256 rule recorded
  in `locked_spec.json`, legacy episodes and bootstraps from CRC32 strings, and the
  smoke/regression examples from constants in their scripts. GRU training calls
  `torch.manual_seed`; the historical leave-one-scene-out seeds are unknown (see
  `PROVENANCE.md`). In the pinned environment the regenerated JSON, CSV and PNG
  files match the shipped ones exactly, apart from generation timestamps and the two
  documented exceptions below.
- **Locked study sources.** `check_revised_anchor_study.py` requires `csq.py`,
  `envloop.py`, `planner.py`, `predictor.py`, `sims.py`, `run_revised_anchor_study.py`,
  `check_revised_anchor_study.py` and the six study inputs to keep their exact bytes;
  editing any of them makes the gate "current sources and inputs match lock" fail by
  design. These files are shipped unmodified.
- **Documented comparison exceptions.** `results/summary.json` was assembled by earlier
  revisions of `make_summary.py` and `update_summary.py`: its free-form `caveats` list has
  older wording and three repeated appends, while every other key regenerates identically.
  `exp7_budget.py --reuse-synthetic` records the SHA-256 of the file it reused
  (`synthetic_rows_reused_from_prior_sha256`), which necessarily differs from the value in
  the shipped file.
- **Figures.** `exp4_figures.fig_longrun` and `exp4_figures.fig_traj` cannot be redrawn with
  the current code: `fig_longrun` draws a third panel on a two-panel grid, and `fig_traj`
  reads a per-step prediction field that the current episode runner no longer records.
  Their shipped files come from an earlier revision of the plotting code, and
  `python code/exp4_figures.py` stops at `fig_longrun`; use `code/redraw_figures.py`. The
  shipped `fig_radius` differs from a redraw only in the legend of its right panel (a label
  was edited after it was drawn), so it is rendered but not compared. PDF files embed a
  creation date and are never compared.
- **Legacy results.** `results/main`, `long`, `matched`, `vehicle`, `exp6_ablations.json`,
  `exp9_matched.json`, `exp10_vehicle.json` and `summary.json` come from the legacy
  recentered controller; they are provenance, not evidence for `window_anchor_v1`. The
  study outcomes are simulated and descriptive (40 episodes per cell, one frozen pool and
  model per regime, no portfolio confidence guarantee).
- **Pickles.** `results/data/*.pkl` were written under NumPy 2.x; `code/np2compat.py`
  installs module aliases so they load under the pinned NumPy 1.26.
- **Changes relative to the original project snapshot.** `collision_attribution.py --check`
  now compares only the stored JSON with the recomputed values; `artifact_manifest.py` lists
  this package's own files, skips numerical caches and gained `--check`; `run_all.sh` is
  portable and includes the resume loops; `check_regenerated_outputs.py` and
  `redraw_figures.py` are new; `results/provenance_manifest.json` was regenerated for this
  package. Two scripts whose only purpose was producing or checking the typeset write-up
  were left out, as were development notes. No experiment code, locked source or result
  file was changed.
