"""Round-7 20.P2.1: a valid-to-diagnostic-only transition example, with an explicit
confidence-lifetime identifier.

Shows one concrete case where the certified statement stops applying and the reported
quantity becomes diagnostic only: the CS-R replay on legacy score streams sits under a
DIFFERENT confidence lifetime than the anchored closed-loop study, because the score
population changed.  Writes results/checks/lifetime_example.json.
"""
import glob, hashlib, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def sha8(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:8]


def main():
    anchored = os.path.join(R, "exp1_validity.json")
    legacy = os.path.join(R, "closedloop_shift")
    legacy_files = sorted(glob.glob(os.path.join(legacy, "*.json")))
    out = {
        "purpose": "round-7 20.P2.1 valid -> diagnostic-only transition example",
        "lifetimes": [
            {"lifetime_id": f"anchored:{sha8(anchored)}",
             "status": "CERTIFIED",
             "scope": "G1/G2/G3 on scores exchangeable with the frozen warm pool, anchored controller",
             "source": os.path.relpath(anchored, ROOT)},
            {"lifetime_id": f"legacy-csr:{sha8(legacy_files[0]) if legacy_files else None}",
             "status": "DIAGNOSTIC ONLY",
             "why": ("the score population is a legacy stream, not the anchored closed-loop one; "
                     "a changed score/selection contract starts a new confidence lifetime, so the "
                     "certified event does not carry over"),
             "not_claimed": ["a certified shift budget", "anchored-controller performance"],
             "source": os.path.relpath(legacy_files[0], ROOT) if legacy_files else None},
        ],
        "rule": ("A change to base-pool bytes, predictor bytes, score/selection contract, or "
                 "deployment epoch starts a new confidence lifetime. Appending scores inside one "
                 "unchanged epoch does not. Anything reported under a new lifetime without "
                 "recalibration is diagnostic, never certified."),
    }
    p = os.path.join(R, "checks", "lifetime_example.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    for lt in out["lifetimes"]:
        print(f"  {lt['lifetime_id']:28s} {lt['status']}")
    print("wrote", p)


if __name__ == "__main__":
    main()
