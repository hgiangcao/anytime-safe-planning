"""Price the closed-loop distribution shift with the shift-robust process (item 20.P1.1).

WHY THIS EXISTS.  Prop. 4 (`RobustQuantileCS`) restores time-uniform validity under a
declared total shift budget rho_tot, at an explicit and constant cost of rho_tot/beta
nats -- equivalently, delta shrinks to delta*exp(-rho_tot/beta).  Until now that process
had only ever been exercised on LEGACY score streams, which are exactly the streams
that do not exhibit the closed-loop shift it exists to absorb.  The round-4 review asks
for it to be developed or tested on the CLOSED-LOOP distribution, and calls this the
main theoretical gap to deployment.

This script does that, using only data already on disk: the fresh 320-attempt anchored
study logs, per window, the deployed radius and the realised score
(`record.window_audit[*].radius` / `.score`), which together ARE the closed-loop score
stream the deployed calibrator saw.  For each attempt we replay its own confidence
sequence -- the locked spec resets confidence state per attempt -- and ask the only
operationally meaningful question:

    what total shift budget rho_tot would this attempt have needed for its radius to
    have dominated every closed-loop score it actually met, and what does that cost?

Reporting is by attempt, retaining every one, including the attempts that need nothing.
No score is recomputed and no episode is re-simulated: this is an analysis of frozen
records, and it cannot change any published containment number.

    python code/exp14_closedloop_shift.py protocol
    python code/exp14_closedloop_shift.py run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402

from csq import QuantileCS, RobustQuantileCS  # noqa: E402

STUDY = ROOT / "results" / "window_anchor_v1_study"
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "closedloop_shift"
OUT.mkdir(parents=True, exist_ok=True)

BETA, DELTA = 0.9, 0.1
# Grid of total shift budgets, in nats of tolerated conditional over-exceedance.
RHO_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

PROTOCOL = {
    "status": "locked_before_any_outcome_of_this_analysis",
    "item": "round-4 20.P1.1 -- test the shift-robust score process on the closed-loop distribution",
    "inputs": "frozen results/window_anchor_v1_study attempt records only; no re-simulation",
    "unit": "one attempt (the locked spec resets confidence state per attempt)",
    "stream": "the attempt's own window_audit scores, in window order",
    "warm_pool": "results/data/cal_<regime>.npy, exactly as the deployed calibrator used",
    "beta": BETA, "delta": DELTA,
    "rho_grid": RHO_GRID,
    "endpoint_1": ("fraction of attempts whose deployed radius already dominates every "
                   "closed-loop score it met (rho_tot = 0 suffices)"),
    "endpoint_2": ("the smallest rho_tot on the grid that restores domination for each "
                   "attempt that needs one, and the implied effective delta "
                   "delta*exp(-rho_tot/beta)"),
    "endpoint_3": "the same, split by regime and by injected tracking bound",
    "retention": "every attempt retained, including those needing no budget and those the grid cannot fix",
    "not_claimed": ("this prices an OBSERVED shift on one simulated study; it does not "
                    "establish a shift budget that transfers to deployment, and rho_tot "
                    "chosen after seeing the stream is a diagnostic, not a certificate"),
}
PROTOCOL_SHA = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def stage_protocol() -> None:
    (OUT / "protocol.json").write_text(json.dumps(
        {"protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    (OUT / "PROTOCOL_SHA256.txt").write_text(PROTOCOL_SHA + "\n")
    print("locked", PROTOCOL_SHA)


def _require_locked() -> None:
    p = OUT / "protocol.json"
    if not p.exists():
        raise SystemExit("run the 'protocol' stage first")
    if json.loads(p.read_text())["protocol_sha256"] != PROTOCOL_SHA:
        raise SystemExit("protocol digest moved; refusing to run")


CACHE = OUT / "boundary_cache"
_CS_CACHE: dict = {}


def _calibrator(rho: float, warm, t_max: int):
    """A calibrator for this rho, with the boundary built once per (rho, t_max) and cached."""
    key = (rho, t_max)
    if key not in _CS_CACHE:
        CACHE.mkdir(parents=True, exist_ok=True)
        _CS_CACHE[key] = True
    if rho == 0.0:
        return QuantileCS(BETA, DELTA, warm, t_max=t_max, cache_dir=str(CACHE))
    return RobustQuantileCS(BETA, DELTA, warm, rho_tot=rho, t_max=t_max, cache_dir=str(CACHE))


def _scan(scores, warm) -> dict:
    """Per-rho behaviour of the shift-robust process on one closed-loop stream.

    Three outcomes per window, kept separate because conflating them is exactly the
    mistake this analysis exists to avoid:
      * certified and dominated : finite radius r with s <= r
      * certified and VIOLATED  : finite radius r with s >  r  (a real domination failure)
      * uncertified             : no finite radius yet (u_t > available samples).  The
        deployed controller falls back here; it is NOT a domination failure.
    Raising rho_tot inflates the evidence threshold, so it makes radii LARGER and, once
    the required order statistic exceeds the pool, makes them INFINITE.  The price of
    shift robustness is therefore paid in certificate availability, not in coverage.
    """
    out = []
    for rho in RHO_GRID:
        cs = _calibrator(rho, warm, t_max=len(warm) + len(scores) + 8)
        cert = viol = 0
        for s_ in scores:
            r = cs.radius()
            if np.isfinite(r):
                cert += 1
                if s_ > r:
                    viol += 1
            cs.add(s_)
        out.append({"rho": rho, "n_certified": cert, "n_violating": viol})
    return {"n_windows": len(scores), "grid": out}


def stage_run() -> None:
    _require_locked()
    warm = {r: np.load(DATA / f"cal_{r}.npy") for r in ("sf", "orca")}
    rows = []
    for path in sorted((STUDY / "attempts").glob("*.json")):
        d = json.loads(path.read_text())
        a, rec = d["attempt"], d["record"]
        audit = rec.get("window_audit") or []
        scores = [w["score"] for w in audit if w.get("score") is not None]
        if not scores:
            rows.append({"attempt_id": a["attempt_id"], "regime": a["regime"],
                         "method": a["method"], "track_m": a["tracking_error_bound_m"],
                         "n_windows": 0, "rho": None, "n_viol_rho0": None,
                         "note": "no scored window"})
            continue
        sc = _scan(scores, warm[a["regime"]])
        g0 = sc["grid"][0]
        # smallest grid rho with zero domination failures among CERTIFIED windows
        fix = next((g for g in sc["grid"] if g["n_violating"] == 0), None)
        rows.append({"attempt_id": a["attempt_id"], "regime": a["regime"],
                     "method": a["method"], "track_m": a["tracking_error_bound_m"],
                     "n_windows": sc["n_windows"],
                     "n_certified_rho0": g0["n_certified"],
                     "n_violating_rho0": g0["n_violating"],
                     "rho_star": (fix["rho"] if fix else float("inf")),
                     "n_certified_at_rho_star": (fix["n_certified"] if fix else 0),
                     "eff_delta_at_rho_star": (DELTA * float(np.exp(-fix["rho"] / BETA))
                                               if fix else 0.0),
                     "grid": sc["grid"]})
    scored = [r for r in rows if r["n_windows"]]
    clean0 = [r for r in scored if r["n_violating_rho0"] == 0]
    dirty0 = [r for r in scored if r["n_violating_rho0"] > 0]
    fixed = [r for r in dirty0 if np.isfinite(r["rho_star"])]
    unfixed = [r for r in dirty0 if not np.isfinite(r["rho_star"])]

    tw = sum(r["n_windows"] for r in scored)
    tc0 = sum(r["n_certified_rho0"] for r in scored)
    tv0 = sum(r["n_violating_rho0"] for r in scored)

    def grid_totals():
        agg = []
        for i, rho in enumerate(RHO_GRID):
            c = sum(r["grid"][i]["n_certified"] for r in scored)
            v = sum(r["grid"][i]["n_violating"] for r in scored)
            agg.append({"rho": rho, "n_certified": c, "n_violating": v,
                        "certified_frac": c / max(1, tw),
                        "violation_frac_of_certified": v / max(1, c),
                        "eff_delta": DELTA * float(np.exp(-rho / BETA))})
        return agg

    def by(key):
        out = {}
        for r in scored:
            g = out.setdefault(str(r[key]), {"n": 0, "n_clean_rho0": 0, "n_unfixable": 0})
            g["n"] += 1
            g["n_clean_rho0"] += int(r["n_violating_rho0"] == 0)
            g["n_unfixable"] += int(r["n_violating_rho0"] > 0 and not np.isfinite(r["rho_star"]))
        return out

    summary = {
        "protocol_sha256": PROTOCOL_SHA,
        "n_attempts": len(rows), "n_scored_attempts": len(scored),
        "n_clean_at_rho0": len(clean0), "n_with_violation_at_rho0": len(dirty0),
        "n_fixable_on_grid": len(fixed), "n_unfixable_on_grid": len(unfixed),
        "total_windows": tw, "total_certified_at_rho0": tc0,
        "total_violating_at_rho0": tv0,
        "violation_frac_of_certified_at_rho0": tv0 / max(1, tc0),
        "certified_frac_at_rho0": tc0 / max(1, tw),
        "grid_totals": grid_totals(),
        "by_regime": by("regime"), "by_track_m": by("track_m"), "by_method": by("method"),
        "attempts": [{k: v for k, v in r.items() if k != "grid"} for r in rows],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"scored attempts                 : {len(scored)}")
    print(f"windows                         : {tw}; certified at rho=0: {tc0} "
          f"({100*tc0/max(1,tw):.1f}%)")
    print(f"domination failures at rho=0    : {tv0} of {tc0} certified "
          f"({100*tv0/max(1,tc0):.1f}%)")
    print(f"attempts clean at rho=0         : {len(clean0)}/{len(scored)}")
    print(f"attempts fixed by a grid budget : {len(fixed)}; not fixed: {len(unfixed)}")
    print("\nrho_tot   certified%   violations/certified   eff_delta")
    for g in summary["grid_totals"]:
        print(f"{g['rho']:>7.3f}   {100*g['certified_frac']:9.1f}   "
              f"{100*g['violation_frac_of_certified']:20.1f}   {g['eff_delta']:.3g}")
    for k, g in summary["by_regime"].items():
        print(f"  regime {k}: n={g['n']} clean@0={g['n_clean_rho0']} unfixable={g['n_unfixable']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("protocol", "run"))
    {"protocol": stage_protocol, "run": stage_run}[ap.parse_args().stage]()
