#!/bin/sh
# Full reproduction pipeline for p2-anytime-safe-planning.
# Run from the repo root (ICRA27/). Total runtime ~25 min on an M1 (3 workers).
# Every step is idempotent/checkpointed; re-run safely after interruption.
set -e
PY=.venv/bin/python
cd 20_anytime-safe-planning/code

# 0. unit tests for the statistical core (must pass; ~40 s)
$PY test_core.py

# 1. data: sim rollouts, GRU predictors, calibration scores, ETH fetch+fallback (~2 min)
$PY exp0_data.py

# 2. synthetic validity of the confidence-sequence machinery (~1 min)
$PY exp1_validity.py

# 3. radius-vs-horizon curves (~1 min)
$PY exp2_radius.py

# 4. main planning experiment: 48 conditions x 200 episodes (11 for eth).
#    Repeat this line until it prints ALL CONDITIONS COMPLETE (~2-3 runs, ~15 min)
$PY exp3_planning.py --minutes 8.5 --workers 3
$PY exp3_planning.py --minutes 8.5 --workers 3

# 5. figures (PDF+PNG in results/figures/) (~30 s)
$PY exp4_figures.py

# 6. aggregate summary
$PY make_summary.py
