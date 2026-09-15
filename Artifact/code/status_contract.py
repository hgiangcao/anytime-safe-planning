"""Round-8 20.P1.3: the consumer-facing status export.

``planner.plan_step`` answers a purely geometric question -- is there an action that
satisfies the anchored constraint? -- and returns ``certified``,
``fallback_no_certificate`` or ``fallback_no_feasible_action``.  Whether the
*statistical* premise of Theorems 1-3 holds is a separate fact that the planner cannot
observe: it depends on the score population and the declared confidence lifetime.

Reporting both under the single label ``certified`` would let a consumer keep the same
confidence wording after the premise has been withdrawn.  This module is the export
layer that keeps them apart.  It deliberately lives outside the study's locked source
set (``results/window_anchor_v1_study/locked_spec.json``): it re-labels an already
recorded run and never changes one.

    >>> export_status("certified", premise_supported=False)
    'anchored_feasible_premise_unsupported'

Each exported status carries the assumption it invokes, the error allocation it spends
(the G2 count event costs the TOTAL delta+delta', not delta' alone) and its confidence
lifetime, answering the reviewer's request that a status say what it is claiming.
"""
from __future__ import annotations

CERTIFIED = "certified"
ANCHORED_UNSUPPORTED = "anchored_feasible_premise_unsupported"

#: Status -> what it claims.  ``evidence`` is the load-bearing field: only
#: ``statistical+geometric`` invokes G1/G2/G3.
CONTRACT: dict[str, dict[str, str]] = {
    CERTIFIED: {
        "assumes": ("i.i.d. windows for the conditional G1/G2 reading; for G3, "
                    "exchangeability with the pool and a level allocation that does "
                    "not depend on score values"),
        "error_allocation": ("delta for G1 and G3; delta+delta' in total for the G2 "
                             "count event (the count boundary's own delta' is only "
                             "one component)"),
        "evidence": "statistical+geometric",
        "lifetime": ("the current base-pool bytes, predictor bytes, score/selection "
                     "contract and deployment epoch"),
    },
    ANCHORED_UNSUPPORTED: {
        "assumes": "nothing statistical",
        "error_allocation": "none spent; the radius is empirical, not certified",
        "evidence": "geometric",
        "lifetime": "withdrawn certificate; the radius is a monitoring diagnostic",
    },
    "fallback_no_certificate": {
        "assumes": "nothing statistical",
        "error_allocation": "none; the radius is +inf, so no finite region exists",
        "evidence": "none",
        "lifetime": "n/a",
    },
    "fallback_no_feasible_action": {
        "assumes": "nothing statistical",
        "error_allocation": "none",
        "evidence": "none",
        "lifetime": "n/a",
    },
    "identity_speed_envelope_feasible_uncertified": {
        "assumes": "nothing statistical",
        "error_allocation": "none; an explicit simulator policy, not a theorem",
        "evidence": "geometric",
        "lifetime": "n/a",
    },
    "fallback_identity_current_observation_evasive": {
        "assumes": "nothing statistical",
        "error_allocation": "none",
        "evidence": "none",
        "lifetime": "n/a",
    },
    "fallback_no_active_window": {
        "assumes": "nothing statistical",
        "error_allocation": "none",
        "evidence": "none",
        "lifetime": "n/a",
    },
}


def export_status(plan_status: str, *, premise_supported: bool) -> str:
    """Map a recorded planner status to the status a consumer should be shown.

    ``premise_supported`` is the statistical premise of Theorems 1-3.  When it is
    False, a feasible anchored action is still deterministically feasible -- the
    triangle-inequality argument at discrete control endpoints does not depend on any
    distributional assumption -- but it carries no G1/G2/G3 statement, so it is
    exported as ``anchored_feasible_premise_unsupported`` rather than ``certified``.
    Statuses that never invoked a statistical claim are returned unchanged.
    """
    if plan_status not in CONTRACT:
        raise KeyError(f"unknown planner status: {plan_status!r}")
    if plan_status == CERTIFIED and not premise_supported:
        return ANCHORED_UNSUPPORTED
    return plan_status


def self_check() -> dict:
    """Every status is declared, and only `certified` invokes a statistical claim."""
    statistical = [k for k, v in CONTRACT.items() if v["evidence"] == "statistical+geometric"]
    geometric = [k for k, v in CONTRACT.items() if v["evidence"] == "geometric"]
    return {
        "n_statuses": len(CONTRACT),
        "statuses_invoking_a_statistical_claim": statistical,
        "statuses_with_deterministic_feasibility_only": geometric,
        "certified_downgrades_to": export_status(CERTIFIED, premise_supported=False),
        "certified_is_unique_statistical_status": statistical == [CERTIFIED],
        "premise_unsupported_is_not_certified":
            export_status(CERTIFIED, premise_supported=False) != CERTIFIED,
        "geometric_statuses_unchanged_by_premise": all(
            export_status(s, premise_supported=False) == s for s in geometric),
    }
