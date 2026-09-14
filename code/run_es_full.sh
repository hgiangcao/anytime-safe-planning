#!/bin/sh
# Fully-expressive per-edge search (separable CMA-ES over |E| log-weights),
# cold-started and toll-warm-started. Runs after the main chain.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
while pgrep -f "run_all_v2\.sh" > /dev/null 2>&1; do sleep 30; done
while pgrep -f "exp8_tie_generality" > /dev/null 2>&1; do sleep 30; done
echo "=== per-edge sep-CMA-ES, cold ==="; $PY code/exp2c_es_strong.py fullcold 25 full
echo "=== per-edge sep-CMA-ES, warm ==="; $PY code/exp2c_es_strong.py fullwarm 25 full
echo "=== final regeneration ==="
$PY code/aggregate_v2.py
$PY code/make_tables.py > /dev/null
$PY code/make_figures_v2.py
$PY code/check_provenance.py
cd paper && latexmk -pdf -interaction=nonstopmode root.tex > /dev/null 2>&1
grep -E "Output written|^! " root.log | tail -3
echo "ES FULL DONE"
