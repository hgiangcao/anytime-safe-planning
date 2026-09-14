#!/bin/sh
# Full experiment chain (benchmark maps, PIBT+swap, tie tolerance eps=1.0).
# Every script is checkpointed per condition, so re-running resumes.
set -e
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
echo "=== exp1b headline ===";                 $PY code/exp1b_headline_bench.py 600
echo "=== exp3b transfer ===";                 $PY code/exp3b_transfer.py
echo "=== exp6b ablations ===";                $PY code/exp6b_ablations.py
echo "=== exp4b directional ===";              $PY code/exp4b_directional.py
echo "=== exp7 planner transfer ===";          $PY code/exp7_planner_transfer.py
echo "=== exp2c strengthened ES (cold) ===";   $PY code/exp2c_es_strong.py cold 25
echo "=== exp2c strengthened ES (warm) ===";   $PY code/exp2c_es_strong.py warm 25
echo "ALL DONE"
