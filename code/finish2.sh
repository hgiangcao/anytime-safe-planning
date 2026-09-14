#!/bin/sh
# Waits for run_all_v2.sh, then runs the tie-generality experiment and
# regenerates every paper artifact, ending with the provenance check.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
while pgrep -f "run_all_v2\.sh" > /dev/null 2>&1; do sleep 30; done
echo "=== main chain done; exp8 tie generality ==="
$PY code/exp8_tie_generality.py
echo "=== regenerating paper artifacts ==="
$PY code/aggregate_v2.py
$PY code/make_tables.py > /dev/null
$PY code/make_figures_v2.py
$PY code/check_provenance.py
cd paper && latexmk -pdf -interaction=nonstopmode root.tex > /dev/null 2>&1
grep -E "Output written|^! " root.log | tail -3
echo "FINISH2 DONE"
