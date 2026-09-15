#!/bin/sh
# Full reproduction pipeline: regenerates every folder and file under results/.
#
#     sh code/run_all.sh                                   # from any working directory
#     PYTHON=.venv/bin/python WORKERS=4 sh code/run_all.sh
#
# The long experiments are time-boxed passes that checkpoint every condition; the loops
# below repeat a pass until it reports completion, and a re-run after an interruption
# resumes.  On top of the shipped results every condition is already complete and is
# skipped.  To rebuild from scratch, move results/ aside first (see README.md).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
WORKERS="${WORKERS:-3}"
MAX_PASSES="${MAX_PASSES:-40}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$ROOT"
mkdir -p results/main results/long results/matched results/vehicle results/checks results/figures
LOG="$(mktemp "${TMPDIR:-/tmp}/run_all.XXXXXX")"
trap 'rm -f "$LOG"' EXIT

# repeat_until MARKER SCRIPT ARGS...: repeat until the last output line contains MARKER
repeat_until() {
    marker="$1"
    shift
    pass=0
    while [ "$pass" -lt "$MAX_PASSES" ]; do
        pass=$((pass + 1))
        "$PY" "$@" > "$LOG" 2>&1 || { cat "$LOG"; return 1; }
        cat "$LOG"
        if tail -n 1 "$LOG" | grep -q "$marker"; then
            return 0
        fi
    done
    echo "still incomplete after $pass passes: $*" >&2
    return 1
}

# repeat_until_counts SCRIPT ARGS...: repeat until no "tag/method: done/target" line is short
repeat_until_counts() {
    pass=0
    while [ "$pass" -lt "$MAX_PASSES" ]; do
        pass=$((pass + 1))
        "$PY" "$@" > "$LOG" 2>&1 || { cat "$LOG"; return 1; }
        cat "$LOG"
        if awk '/^ *[^ :]+\/[^ :]+: [0-9]+\/[0-9]+$/ { split($NF, a, "/"); if (a[1] + 0 < a[2] + 0) short = 1 }
                END { exit short }' "$LOG"; then
            return 0
        fi
    done
    echo "still incomplete after $pass passes: $*" >&2
    return 1
}

echo "== 0. data: simulator rollouts, GRU predictors, calibration scores, ETH/UCY scenes"
"$PY" code/exp0_data.py
echo "== 1. unit tests"
"$PY" code/test_core.py
echo "== 2. exp1: synthetic time-uniform validity"
"$PY" code/exp1_validity.py
echo "== 3. exp2: radius and certifiable-horizon curves"
"$PY" code/exp2_radius.py
echo "== 4. exp3: 8-window planning conditions -> results/main"
repeat_until "ALL CONDITIONS COMPLETE" code/exp3_planning.py --minutes 8.5 --workers "$WORKERS"
echo "== 5. exp5: 120-window long-horizon deployments -> results/long"
repeat_until "ALL LONG CONDITIONS COMPLETE" code/exp5_longrun.py --minutes 10 --workers "$WORKERS"
echo "== 6. exp6: predictor and planner ablations"
"$PY" code/exp6_ablations.py
echo "== 7. exp7: G2 cumulative-violation budget"
"$PY" code/exp7_budget.py
echo "== 8. exp8: shift-robust confidence sequence"
"$PY" code/exp8_robust.py
echo "== 9. exp9: rank wall, lookahead wall, matched-calibration arm -> results/matched"
repeat_until_counts code/exp9_matched.py --workers "$WORKERS"
echo "== 10. exp10: unicycle vehicle and shift-robust stress tests -> results/vehicle"
repeat_until_counts code/exp10_vehicle.py --workers "$WORKERS"
echo "== 11. exp12: held-out shift-budget diagnostic"
"$PY" code/exp12_rho_loso.py
echo "== 12. dependence-aware intervals for the long real-data runs"
"$PY" code/audit_dependence.py
echo "== 13. window_anchor_v1 study (320 serial attempts) and its strict validator"
"$PY" code/run_revised_anchor_study.py
"$PY" code/check_revised_anchor_study.py
echo "== 14. exp14: closed-loop shift pricing on the study records"
"$PY" code/exp14_closedloop_shift.py protocol
"$PY" code/exp14_closedloop_shift.py run
echo "== 15. study-record analyses, smoke episode and regression examples"
"$PY" code/collision_attribution.py
"$PY" code/contract_smoke.py
"$PY" code/make_lifetime_example.py
"$PY" code/make_regression_examples.py
echo "== 16. figures"
"$PY" code/redraw_figures.py
echo "== 17. aggregate summary and stored-value audit"
"$PY" code/make_summary.py
"$PY" code/update_summary.py
"$PY" code/audit_new.py
echo "== 18. provenance manifest (last)"
"$PY" code/artifact_manifest.py
echo "run_all.sh complete"
