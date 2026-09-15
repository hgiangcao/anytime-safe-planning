#!/bin/sh
# Quick check: unit tests plus fast checks that recompute reported values from the
# shipped frozen results.  Works from any working directory:
#
#     sh run_checks.sh                        # interpreter: $PYTHON, default python3
#     PYTHON=.venv/bin/python sh run_checks.sh
#
# Steps: provenance manifest (shipped bytes), unit tests, the strict validator of the
# 320-attempt window_anchor_v1 study, the four standalone regression examples, and a
# regenerate-and-compare pass over the derived result files and figures (run in a
# scratch copy, so nothing shipped is modified).  Exit status is non-zero on any failure.
ROOT="$(cd "$(dirname "$0")" && pwd)" || exit 1
PY="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$ROOT" || exit 1

failed=0
start_all=$(date +%s)

step() {
    name="$1"
    shift
    t0=$(date +%s)
    echo "== $name"
    if "$PY" "$@"; then
        echo "-- PASS $name ($(( $(date +%s) - t0 )) s)"
    else
        echo "-- FAIL $name ($(( $(date +%s) - t0 )) s)"
        failed=$((failed + 1))
    fi
}

step "shipped files match results/provenance_manifest.json" code/artifact_manifest.py --check
step "unit tests (code/test_core.py)" code/test_core.py
step "window_anchor_v1 study validator (23 gates, 320 full traces)" code/check_revised_anchor_study.py
for example in anchor identity tracking rank; do
    step "regression example: $example" code/make_regression_examples.py --example "$example"
done
step "regenerated results and figures match the shipped copies" code/check_regenerated_outputs.py

echo
if [ "$failed" -eq 0 ]; then
    echo "run_checks.sh: all steps passed in $(( $(date +%s) - start_all )) s"
    exit 0
fi
echo "run_checks.sh: $failed step(s) FAILED"
exit 1
