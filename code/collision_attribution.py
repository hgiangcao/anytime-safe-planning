"""Attribute observed substep collisions to fallback vs. anchored controls.

Round-6 reviewers all asked the same question of the fresh anchored study:
"What fraction of collisions occurs during fallback versus certified controls?"
Nothing new has to be run to answer it.  Every one of the 320 frozen attempt
records already carries a per-control ``control_audit_trace`` whose entries hold
both ``fallback`` and ``observed_substep_collision``, so the breakdown is a
regrouping of the same bytes the main table is built from.

Conventions match Table I(b) exactly: the first five controls of each attempt are
warm-up and are excluded, so the denominator here is the same "active controls"
denominator the fallback fraction uses.

This is an association, not a causal attribution.  Fallback is entered precisely
when no active window certifies a constraint, which is also when the robot is
most likely to be in a difficult configuration; the split says where the residual
risk sits, not what produced it.

    python code/collision_attribution.py           # recompute and write JSON
    python code/collision_attribution.py --check    # fail if JSON or paper drifted
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ROOT / "results" / "window_anchor_v1_study" / "attempts"
OUT = ROOT / "results" / "collision_attribution.json"
WARMUP = 5  # matches Table I(b): "five warm-up controls excluded"


def compute() -> dict:
    files = sorted(ATTEMPTS.glob("*.json"))
    if not files:
        raise SystemExit(f"no attempt records under {ATTEMPTS}")
    n_ctl = {"fallback": 0, "anchored": 0}
    n_coll_ctl = {"fallback": 0, "anchored": 0}
    n_first = {"fallback": 0, "anchored": 0}
    n_episodes = 0
    n_coll_episodes = 0
    n_coll_steps_field = 0
    cells: dict[str, dict] = {}
    for f in files:
        record = json.loads(f.read_text())["record"]
        trace = record["control_audit_trace"]
        # the per-control trace must agree with the record's own summary fields,
        # otherwise this regrouping is not the same evidence as the main table
        if [c["observed_substep_collision"] for c in trace] != record["collision_step_labels"]:
            raise SystemExit(f"trace/collision_step_labels disagree in {f.name}")
        if [c["fallback"] for c in trace] != record["fallback_labels"]:
            raise SystemExit(f"trace/fallback_labels disagree in {f.name}")
        n_coll_steps_field += int(record["n_coll_steps"])
        n_episodes += 1
        cell = f.name.split("__seed")[0]
        c_agg = cells.setdefault(cell, {"fallback_controls": 0, "fallback_collision_controls": 0,
                                        "anchored_controls": 0, "anchored_collision_controls": 0,
                                        "episodes": 0, "collision_episodes": 0})
        c_agg["episodes"] += 1
        first = None
        for i, c in enumerate(trace[WARMUP:], start=WARMUP):
            kind = "fallback" if c["fallback"] else "anchored"
            n_ctl[kind] += 1
            c_agg[f"{kind}_controls"] += 1
            if c["observed_substep_collision"]:
                n_coll_ctl[kind] += 1
                c_agg[f"{kind}_collision_controls"] += 1
                if first is None:
                    first = kind
        if record["collision"]:
            n_coll_episodes += 1
            c_agg["collision_episodes"] += 1
            if first is None:
                raise SystemExit(f"collision episode with no active colliding control: {f.name}")
            n_first[first] += 1

    total_ctl = n_ctl["fallback"] + n_ctl["anchored"]
    total_coll_ctl = n_coll_ctl["fallback"] + n_coll_ctl["anchored"]
    return {
        "source": "results/window_anchor_v1_study/attempts (320 frozen records)",
        "warmup_controls_excluded_per_attempt": WARMUP,
        "episodes": n_episodes,
        "collision_episodes": n_coll_episodes,
        "active_controls": total_ctl,
        "colliding_controls": total_coll_ctl,
        "colliding_controls_all_including_warmup": n_coll_steps_field,
        "fallback_controls": n_ctl["fallback"],
        "anchored_controls": n_ctl["anchored"],
        "fallback_collision_controls": n_coll_ctl["fallback"],
        "anchored_collision_controls": n_coll_ctl["anchored"],
        "fallback_collision_rate_pct": 100.0 * n_coll_ctl["fallback"] / n_ctl["fallback"],
        "anchored_collision_rate_pct": 100.0 * n_coll_ctl["anchored"] / n_ctl["anchored"],
        "first_contact_under_fallback_episodes": n_first["fallback"],
        "first_contact_under_anchored_episodes": n_first["anchored"],
        "by_cell": cells,
        "interpretation": ("Association only. Fallback is entered exactly when no active window "
                           "certifies a constraint, so the split localises residual risk and does "
                           "not attribute cause."),
    }


def paper_numbers(d: dict) -> list[tuple[str, str]]:
    """The exact strings the manuscript prints, so --check catches paper drift."""
    return [
        ("colliding controls under fallback", f"${d['fallback_collision_controls']}$ of "
                                              f"${d['colliding_controls']}$"),
        ("first contact under fallback", f"${d['first_contact_under_fallback_episodes']}$ of "
                                         f"${d['collision_episodes']}$"),
        ("fallback rate", f"{d['fallback_collision_rate_pct']:.3f}"),
        ("anchored rate", f"{d['anchored_collision_rate_pct']:.3f}"),
        ("fallback controls", f"{d['fallback_controls']:,}".replace(",", "{,}")),
        ("anchored controls", f"{d['anchored_controls']:,}".replace(",", "{,}")),
    ]


def main() -> int:
    check = "--check" in sys.argv
    d = compute()
    if check:
        bad = []
        if not OUT.exists():
            bad.append(f"missing {OUT.relative_to(ROOT)}")
        else:
            stored = json.loads(OUT.read_text())
            for k, v in d.items():
                if k == "by_cell":
                    continue
                if stored.get(k) != v:
                    bad.append(f"{k}: stored={stored.get(k)!r} recomputed={v!r}")
        tex = (ROOT / "paper" / "root.tex").read_text()
        for name, needle in paper_numbers(d):
            if needle not in tex:
                bad.append(f"root.tex missing {name}: {needle!r}")
        if bad:
            print("FAIL collision_attribution:")
            for b in bad:
                print("  -", b)
            return 1
        print("PASS collision_attribution: JSON and root.tex match the frozen records")
        return 0
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for k, v in d.items():
        if k not in ("by_cell", "interpretation", "source"):
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
