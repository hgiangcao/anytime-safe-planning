#!/bin/sh
# exp7 and the grid-parameterisation CMA-ES were skipped when the main chain
# aborted at exp4b (path bug, since fixed). Run them, then regenerate.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
echo "=== exp7 planner transfer ==="; $PY code/exp7_planner_transfer.py
echo "=== exp2c grid CMA-ES cold ==="; $PY code/exp2c_es_strong.py cold 25
echo "=== exp2c grid CMA-ES warm ==="; $PY code/exp2c_es_strong.py warm 25
echo "MISSING DONE"
