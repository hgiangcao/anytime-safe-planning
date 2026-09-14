#!/bin/sh
# Wait for the experiment chain, then regenerate every paper artifact.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
while pgrep -f "run_all_v2.sh" >/dev/null 2>&1; do sleep 30; done
echo "=== chain finished, regenerating ==="
$PY code/aggregate_v2.py
$PY code/make_tables.py > /dev/null
$PY code/make_figures_v2.py
cd paper && latexmk -pdf -interaction=nonstopmode root.tex > /dev/null 2>&1
grep -c "Output written" root.log 2>/dev/null
echo "REGEN DONE"
