#!/bin/sh
# Full experiment pipeline. Every script is checkpointed: re-run any step to
# resume where it left off (safe under a 10-minute shell cap).
# From project root: sh code/run_all.sh
PY=/Users/phatttt/Documents/Claude/Projects/ICRA27/.venv/bin/python
cd "$(dirname "$0")" || exit 1

$PY -m pytest tests/test_core.py -q     # unit tests (optimization core etc.)
$PY exp1_headline.py                    # headline sweep (~30-45 min total)
$PY exp2_es.py                          # CMA-ES baseline search + eval (~8 min)
$PY exp3_transfer.py                    # transfer across N (~25 min)
$PY exp4_heatmaps.py                    # congestion heatmap runs (~1 min)
$PY exp5_fluid.py                       # fluid-model prediction table
$PY exp6_ablations.py                   # ablations (~15 min)
$PY make_figures.py                     # figures + tables + summary.json
