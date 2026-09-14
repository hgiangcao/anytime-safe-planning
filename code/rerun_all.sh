#!/bin/sh
# Re-run the closed-loop experiments after the planner-fallback change and the
# addition of the aucb / aci_ep arms, plus the new long-horizon experiment.
cd "$(dirname "$0")"
PY=../../.venv/bin/python
# (results/main is NOT wiped here; delete it manually when the planner changes)
i=0
while [ $i -lt 12 ]; do
  i=$((i+1))
  L=$($PY exp5_longrun.py --minutes 6 --workers 4 | tail -1)
  M=$($PY exp3_planning.py --minutes 6 --workers 4 | tail -1)
  echo "pass $i : long=[$L] main=[$M]"
  case "$L$M" in
    *"ALL LONG CONDITIONS COMPLETE"*"ALL CONDITIONS COMPLETE"*) echo "DONE"; break;;
  esac
done
