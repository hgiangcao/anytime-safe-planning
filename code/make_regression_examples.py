"""Round-8 20.P2.1: the anchor, identity, tracking and rank-cutoff regressions as
four *independently runnable* examples, each with its own expected output.

Reviewer 4 asked that the release expose small entry points that can be run one at a
time, rather than only as gates inside the 23-gate study validator.  Each example here
is self-contained: ``--example rank`` and ``--example tracking`` need no stored study at
all (they recompute from ``csq``/``planner``), while ``--example anchor`` and
``--example identity`` replay a single stored attempt.  Every example declares the
expectation it checks BEFORE reading any observation, so a silent regression cannot be
absorbed by rewriting the expectation.

The artifact also records the exported planner-status contract (Round-8 20.P0.2/20.P1.3):
which assumption each status invokes, what error allocation it spends, and which
confidence lifetime it belongs to -- so an unsupported statistical premise is never
reported under the same label as deterministic anchored feasibility.

Run::

    python code/make_regression_examples.py                  # all four + write JSON
    python code/make_regression_examples.py --example rank   # just one, no JSON write

Writes ``results/checks/regression_examples.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csq import cert_horizon_fixed_T, cert_horizon_minimal, cert_horizon_schedule  # noqa: E402
from planner import plan_step  # noqa: E402
from status_contract import CONTRACT as STATUS_CONTRACT, self_check  # noqa: E402
from predictor import H as PRED_H  # noqa: E402
from sims import DT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
STUDY = os.path.join(R, "window_anchor_v1_study")

# The manuscript's pool/budget for every horizon number it quotes.
N_POOL, DELTA = 2094, 0.05
A_MAX = 2.0


# ------------------------------------------------------------------ example 1
def example_rank() -> dict:
    """Prop. 2/3 rank cutoffs: the four horizons quoted in Sec. IV-C and Sec. V-D.

    Pure arithmetic on the empirical-rank / additive-allocation family; needs no
    stored artifact, so a reader can run it against a fresh checkout.
    """
    expected = {"fixed_pool_known_T": 104, "optimal_horizon_free": 107,
                "auc": 9, "aucb_lambda_0.05": 52}

    def aucb_levels(lam):
        state = {"rem": DELTA}

        def eps_of_w(w):
            n_w = N_POOL + w - 1
            floor = 1.0 / (n_w + 1)
            e = max(floor, lam * state["rem"])
            state["rem"] -= e
            return e
        return eps_of_w

    observed = {
        "fixed_pool_known_T": cert_horizon_fixed_T(N_POOL, DELTA),
        "optimal_horizon_free": cert_horizon_minimal(N_POOL, DELTA),
        "auc": cert_horizon_schedule(lambda w: DELTA / (w * (w + 1)), N_POOL, DELTA),
        "aucb_lambda_0.05": cert_horizon_schedule(aucb_levels(0.05), N_POOL, DELTA),
    }
    return {
        "name": "rank_cutoff",
        "runs_standalone": True,
        "checks": ("the four certifiable horizons quoted for n=2094, delta=0.05 inside "
                   "the empirical-rank per-window-allocation family"),
        "scope": ("family-specific: split-CP empirical order statistics combined by "
                  "sum_w eps_w <= delta. Not a limit on conformal prediction generally."),
        "expected": expected,
        "observed": observed,
        "pass": observed == expected,
    }


# ------------------------------------------------------------------ example 2
def example_tracking() -> dict:
    """The tracking contract: the injected perturbation never exceeds m_tr.

    Reruns the manuscript's ``projected_acceleration_endpoint_bound_v1`` stress on a
    fresh planner call, so the bound is re-derived rather than re-read.
    """
    m_tr = 0.10
    rng = np.random.default_rng(20260908)
    worst = 0.0
    n = 0
    for trial in range(200):
        p = rng.uniform(-4.0, 4.0, 2)
        v = rng.uniform(-1.0, 1.0, 2)
        goal = rng.uniform(-5.0, 5.0, 2)
        preds = rng.uniform(-5.0, 5.0, (PRED_H, 5, 2))
        a_nom, _ = plan_step(p, v, goal, preds, 0.6, PRED_H,
                             tracking_margin=m_tr, return_status=True)
        angle = trial * (np.sqrt(5.0) - 1.0) * np.pi
        da = (m_tr / (DT * DT)) * np.array([np.cos(angle), np.sin(angle)])
        a_real = a_nom + da
        norm = float(np.linalg.norm(a_real))
        if norm > A_MAX:                       # nonexpansive projection to ||a||<=A_MAX
            a_real = a_real * (A_MAX / norm)
        err = float(np.linalg.norm(a_real - a_nom) * DT * DT)
        worst = max(worst, err)
        n += 1
    return {
        "name": "tracking_bound",
        "runs_standalone": True,
        "checks": ("a deterministic acceleration perturbation of magnitude m_tr/dt^2, "
                   "projected onto ||a||<=2, moves the one-step endpoint by at most m_tr"),
        "scope": ("robot-side endpoint bound at discrete control times; supplied "
                  "separately from pedestrian containment and not a swept-volume bound"),
        "expected": {"tracking_margin_m": m_tr, "max_endpoint_error_m_at_most": m_tr,
                     "trials": 200},
        "observed": {"trials": n, "max_endpoint_error_m": worst},
        "pass": bool(n == 200 and worst <= m_tr + 1e-12),
    }


# ------------------------------------------------------------ stored-attempt IO
def _one_attempt() -> tuple[dict, str] | tuple[None, None]:
    files = sorted(glob.glob(os.path.join(STUDY, "attempts", "*.json")))
    if not files:
        return None, None
    payload = json.loads(open(files[0]).read())
    return payload, os.path.relpath(files[0], ROOT)


# ------------------------------------------------------------------ example 3
def example_anchor() -> dict:
    """The execution contract: every within-window replan keeps the window-start tube.

    For one stored attempt, each control's enforced prediction must be the *suffix* of
    the prediction issued at its window start -- never the fresh prediction of that
    step.  A recentered constraint would silently change the object that was scored.
    """
    payload, src = _one_attempt()
    if payload is None:
        return {"name": "anchor_suffix", "runs_standalone": False,
                "pass": None, "skipped": "no stored attempt found"}
    trace = os.path.join(STUDY, payload["trace"]["path"])
    z = np.load(trace, allow_pickle=False)
    fresh = z["fresh_predictions"]
    con = z["constraint_predictions_padded"]
    hor = z["constraint_horizon"]
    anc = z["constraint_anchor_step"]
    audit = payload["record"]["control_audit_trace"]
    H = fresh.shape[1]
    anchored = recentered = identity_policy = 0
    ok = True
    for t in range(len(anc)):
        if anc[t] < 0:                          # warm-up: no active window
            continue
        if audit[t]["identity_policy_input"] != "anchored_prediction_suffix":
            # the explicitly labelled current-observation speed envelope: a simulator
            # policy with no score-based claim, so it is exempt by construction.
            identity_policy += 1
            continue
        h = int(hor[t])
        ok &= bool(np.array_equal(con[t][:h], fresh[int(anc[t])][H - h:]))
        anchored += 1
        if t != int(anc[t]) and np.array_equal(con[t][:h], fresh[t][:h]):
            recentered += 1
    return {
        "name": "anchor_suffix",
        "runs_standalone": True,
        "checks": ("constraint_predictions[t] equals the remaining suffix of the "
                   "window-start prediction, for every active control of one attempt"),
        "scope": ("one stored attempt; the 23-gate validator runs the same test on all "
                  "320. Controls under the current-observation identity policy are "
                  "counted separately and carry no score-based implication."),
        "source": src,
        "expected": {"all_anchored_controls_match_suffix": True, "recentered_controls": 0},
        "observed": {"anchored_controls": anchored, "suffix_matches": bool(ok),
                     "recentered_controls": recentered,
                     "identity_policy_controls_exempt": identity_policy},
        "pass": bool(ok and recentered == 0 and anchored > 0),
    }


# ------------------------------------------------------------------ example 4
def example_identity() -> dict:
    """The identity contract: an excluded/respawned UID carries no score-based claim.

    Windows whose agent set is not fully clean must not report a selected-agent
    endpoint implication for the excluded UIDs; the current-observation speed envelope
    is an explicit simulator policy with no statistical or continuous-time guarantee.
    """
    payload, src = _one_attempt()
    if payload is None:
        return {"name": "identity_exclusion", "runs_standalone": False,
                "pass": None, "skipped": "no stored attempt found"}
    rec = payload["record"]
    windows = rec["window_audit"]
    touched = [w for w in windows
               if w["identity_policy_triggered"] or w["excluded_at_anchor_agent_uids"]]
    leaked = 0
    for w in touched:
        selected = set(w["score_selected_agent_uids"])
        excluded = set(w["excluded_at_anchor_agent_uids"])
        leaked += len(selected & excluded)
    return {
        "name": "identity_exclusion",
        "runs_standalone": True,
        "checks": ("no UID excluded at the anchor is also carried in that window's "
                   "score-selected set, so no certificate is reused across identities"),
        "scope": ("one stored attempt; the speed envelope itself is a simulator policy, "
                  "not a statistical or continuous-time guarantee"),
        "source": src,
        "expected": {"identity_windows_present": True,
                     "excluded_uids_in_selected_set": 0},
        "observed": {"windows_with_identity_events": len(touched),
                     "excluded_uids_in_selected_set": leaked,
                     "contract": rec["identity_policy_contract"]},
        "pass": bool(len(touched) > 0 and leaked == 0),
    }


EXAMPLES = {"rank": example_rank, "tracking": example_tracking,
            "anchor": example_anchor, "identity": example_identity}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", choices=sorted(EXAMPLES),
                    help="run one example standalone and skip the JSON write")
    args = ap.parse_args()

    names = [args.example] if args.example else ["anchor", "identity", "tracking", "rank"]
    results = [EXAMPLES[n]() for n in names]
    for r in results:
        verdict = "SKIP" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL")
        print(f"{verdict} {r['name']}: {r.get('checks', r.get('skipped', ''))}")
    failed = [r["name"] for r in results if r["pass"] is False]

    if args.example is None:
        out = {
            "purpose": "round-8 20.P2.1 independently runnable anchor/identity/"
                       "tracking/rank-cutoff regressions",
            "how_to_run_one": "python code/make_regression_examples.py --example <name>",
            "examples": results,
            "exported_status_contract": STATUS_CONTRACT,
            "exported_status_self_check": self_check(),
            "note": ("These are proof-of-implementation regressions, not a deployment "
                     "safety argument: they check the executable semantics of the "
                     "anchor, identity, tracking and rank contracts, and say nothing "
                     "about collision rates."),
        }
        p = os.path.join(R, "checks", "regression_examples.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(out, open(p, "w"), indent=1)
        print("wrote", p)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
