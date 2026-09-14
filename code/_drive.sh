#!/bin/sh
# Repeatedly invoke checkpointed experiment scripts until each reports
# complete. Emits one status line per script invocation (for Monitor).
PY=/Users/phatttt/Documents/Claude/Projects/ICRA27/.venv/bin/python
cd "$(dirname "$0")" || exit 1
LOG=/private/tmp/claude-503/-Users-phatttt-Documents-Claude-Projects-ICRA27/73a2dd3b-6291-4f81-836b-778f86da648d/scratchpad/drive.log
: > "$LOG"

run_until_done() {
  script=$1; tries=$2
  i=0
  while [ $i -lt "$tries" ]; do
    i=$((i+1))
    out=$($PY "$script" 2>&1); code=$?
    echo "== $script try $i ==" >> "$LOG"; echo "$out" >> "$LOG"
    last=$(echo "$out" | grep -E "complete|pending|budget|Error|Traceback|done" | tail -1)
    echo "$script try $i (exit $code): $last"
    if [ $code -ne 0 ]; then echo "$script FAILED (exit $code)"; return 1; fi
    if echo "$out" | grep -q "complete ("; then return 0; fi
    if echo "$out" | grep -q ", 0 pending"; then return 0; fi
    if echo "$out" | grep -q "ES search done"; then return 0; fi
  done
  echo "$script NOT COMPLETE after $tries tries"
  return 1
}

run_until_done exp1_headline.py 12
run_until_done exp2_es.py 6
run_until_done exp3_transfer.py 12
run_until_done exp4_heatmaps.py 3
run_until_done exp6_ablations.py 8
$PY exp5_fluid.py >> "$LOG" 2>&1 && echo "exp5 done"
$PY make_figures.py >> "$LOG" 2>&1 && echo "figures done"
echo "ALL DRIVER STEPS FINISHED"
