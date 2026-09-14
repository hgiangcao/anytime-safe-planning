"""Round-7 17.P2.1: a small self-contained fidelity-sweep / native-check artifact.

Collects, per benchmark cell, the epsilon_tie fidelity sweep and the native-PIBT check
into one file a reader can inspect without rerunning anything.  Writes
results/checks/screen_artifact.json; nothing existing is modified.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def load(*parts):
    p = os.path.join(R, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    tie = load("checks", "tie_resolution.json")
    nat = load("native_check", "export_manifest.json")
    out = {
        "purpose": "round-7 17.P2.1 fidelity-sweep and native-check artifact",
        "what_this_is": ("the two measurements the validity screen rests on: the epsilon_tie "
                         "fidelity sweep (does following the field more faithfully help or hurt?) "
                         "and the compiled third-party PIBT native check (does the ordering "
                         "transfer off our simulator?)."),
        "fidelity_sweep": tie,
        "native_check_manifest": nat,
        "work_budget": {
            "units": "training-evaluation equivalents (two seeds x 500 steps each)",
            "rollout_free_toll_solve": 21,
            "map_only_corridor_screen": 0,
            "fidelity_diagnostic": "planner rollouts at two tie widths per cell",
            "cold_black_box_search_for_reference": 3070,
            "note": ("the three costs differ in kind, not just in size: the solve needs no rollouts, "
                     "the map-only screen needs neither rollouts nor the planner, and only the "
                     "fidelity diagnostic requires running the planner."),
        },
        "reading": ("A cell is inside the boundary when raising fidelity (lowering epsilon_tie) "
                    "raises throughput. The screen predicts that sign from the map alone via the "
                    "corridor share rho_2; it is a registered prediction on five maps, not a "
                    "validated classifier."),
    }
    p = os.path.join(R, "checks", "screen_artifact.json")
    json.dump(out, open(p, "w"), indent=1)
    n = len(tie) if isinstance(tie, (list, dict)) else 0
    print(f"fidelity sweep entries: {n}; native manifest: {'yes' if nat else 'missing'}")
    print("wrote", p)


if __name__ == "__main__":
    main()
