"""Regenerate derived result files and figures from the shipped inputs and compare them
with the shipped copies.

Each step runs one shipped script, unchanged, inside a scratch copy of the package, so the
shipped results are never modified: code/ and the small result files are copied, and the
large read-only input folders (results/data, models, main, long, matched, vehicle and
window_anchor_v1_study) are symbolic links to the shipped ones.

Comparison rules
* JSON: value by value (NaN equals NaN).  Top-level keys that only record when or where a
  file was written are ignored: generated_utc, started_utc, completed_utc, environment.
  Two further per-file exceptions are declared (and explained) in STEPS below.
* CSV: byte-identical.
* PNG: byte-identical or identical decoded pixels.  PDF files embed a creation date and
  are not compared.  Set SKIP_FIGURES=1 to skip the figure step, e.g. with a matplotlib
  build other than the pinned one.

    python code/check_regenerated_outputs.py          # exit status 1 on any mismatch
    python code/check_regenerated_outputs.py --keep   # keep the scratch copy for inspection
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINKED = ("data", "models", "main", "long", "matched", "vehicle", "window_anchor_v1_study")
COPIED_SUBDIRS = ("checks", "closedloop_shift")
VOLATILE = {"generated_utc", "started_utc", "completed_utc", "environment"}


def out(path, compare=True, ignore=(), keep_prior=False):
    return {"path": path, "compare": compare, "ignore": set(ignore), "keep_prior": keep_prior}


FIGS = [out(f"results/figures/{name}.png")
        for name in ("fig_core", "fig_anytime", "fig_validity", "fig_calibration", "fig_pareto")]
# The shipped fig_radius.png predates a legend-label edit in exp4_figures.py (panel b), so
# it is only required to render.
FIGS.append(out("results/figures/fig_radius.png", compare=False))

STEPS = [
    ("stored-value audit of exp7-exp10 and episode totals", ["code/audit_new.py"], []),
    ("radius and certifiable-horizon curves (exp2)", ["code/exp2_radius.py"],
     [out("results/exp2_radius.csv"), out("results/exp2_horizon.csv"),
      out("results/exp2_epsdelta.csv"), out("results/exp2_headline.json")]),
    # --reuse-synthetic keeps the stored synthetic Monte-Carlo rows and recomputes the
    # deployment block from results/long.  The key below stores the SHA-256 of the file the
    # rows were read from, so it necessarily differs from the value in the shipped file.
    ("G2 violation budget on the recorded long runs (exp7)",
     ["code/exp7_budget.py", "--reuse-synthetic"],
     [out("results/exp7_budget.json", ignore={"synthetic_rows_reused_from_prior_sha256"},
          keep_prior=True)]),
    ("held-out shift-budget diagnostic (exp12)", ["code/exp12_rho_loso.py"],
     [out("results/exp12_rho_loso.json")]),
    ("dependence-aware intervals for the long real-data runs", ["code/audit_dependence.py"],
     [out("results/dependence_aware_long_real.json")]),
    ("closed-loop shift pricing on the 320 study records (exp14)",
     ["code/exp14_closedloop_shift.py", "run"], [out("results/closedloop_shift/summary.json")]),
    ("collision attribution on the 320 study records", ["code/collision_attribution.py"],
     [out("results/collision_attribution.json")]),
    ("confidence-lifetime example", ["code/make_lifetime_example.py"],
     [out("results/checks/lifetime_example.json")]),
    ("anchor/identity/tracking/rank regression examples", ["code/make_regression_examples.py"],
     [out("results/checks/regression_examples.json")]),
    ("window_anchor_v1 contract smoke episode", ["code/contract_smoke.py"],
     [out("results/window_anchor_smoke.json")]),
    ("aggregate summary, first stage (make_summary)", ["code/make_summary.py"],
     [out("results/summary.json", compare=False)]),
    # The shipped summary.json was assembled by earlier revisions of make_summary.py and
    # update_summary.py: its free-text "caveats" list has older wording and three repeated
    # appends.  Every other key regenerates identically.
    ("aggregate summary, second stage (update_summary)", ["code/update_summary.py"],
     [out("results/summary.json", ignore={"caveats"}, keep_prior=True)]),
    ("figures", ["code/redraw_figures.py"], FIGS),
]


def build_shadow(tmp: Path) -> Path:
    shadow = tmp / "package"
    shutil.copytree(ROOT / "code", shadow / "code",
                    ignore=shutil.ignore_patterns("__pycache__"))
    res = shadow / "results"
    res.mkdir(parents=True)
    for name in LINKED:
        os.symlink(ROOT / "results" / name, res / name, target_is_directory=True)
    for f in (ROOT / "results").iterdir():
        if f.is_file():
            shutil.copy2(f, res / f.name)
    for sub in COPIED_SUBDIRS:
        (res / sub).mkdir()
        for f in (ROOT / "results" / sub).iterdir():
            if f.is_file():
                shutil.copy2(f, res / sub / f.name)
    (res / "figures").mkdir()
    return shadow


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def json_diffs(a, b, path, found, limit=5):
    if len(found) >= limit:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                found.append(f"{path}/{k}: present only in the "
                             f"{'regenerated' if k in a else 'shipped'} file")
            else:
                json_diffs(a[k], b[k], f"{path}/{k}", found, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            found.append(f"{path}: {len(a)} regenerated vs {len(b)} shipped items")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            json_diffs(x, y, f"{path}[{i}]", found, limit)
    elif is_number(a) and is_number(b):
        if not (a == b or (isinstance(a, float) and isinstance(b, float)
                           and math.isnan(a) and math.isnan(b))):
            found.append(f"{path}: regenerated {a!r} vs shipped {b!r}")
    elif a != b or type(a) is not type(b):
        found.append(f"{path}: regenerated {str(a)[:80]!r} vs shipped {str(b)[:80]!r}")


def compare(shadow: Path, spec: dict) -> list[str]:
    new, old = shadow / spec["path"], ROOT / spec["path"]
    if not old.exists():
        return [f"{spec['path']}: no shipped copy to compare with"]
    suffix = new.suffix.lower()
    if suffix == ".json":
        x, y = json.loads(new.read_text()), json.loads(old.read_text())
        for key in VOLATILE | spec["ignore"]:
            x.pop(key, None)
            y.pop(key, None)
        found: list[str] = []
        json_diffs(x, y, spec["path"], found)
        return found
    if new.read_bytes() == old.read_bytes():
        return []
    if suffix == ".png":
        import matplotlib.image as mpimg
        import numpy as np
        a, b = mpimg.imread(new), mpimg.imread(old)
        if a.shape == b.shape and np.array_equal(a, b):
            return []
        n_diff = int((a != b).any(axis=-1).sum()) if a.shape == b.shape else -1
        return [f"{spec['path']}: pixels differ (shape {a.shape} vs {b.shape}, "
                f"{n_diff} differing pixels)"]
    return [f"{spec['path']}: bytes differ"]


def run_step(shadow: Path, label: str, cmd: list[str], outputs: list[dict]) -> bool:
    prior = {}
    for spec in outputs:
        p = shadow / spec["path"]
        if spec["keep_prior"]:
            prior[spec["path"]] = p.stat().st_mtime_ns if p.exists() else None
        elif p.exists():
            p.unlink()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    t0 = time.time()
    proc = subprocess.run([sys.executable, *cmd], cwd=shadow, env=env,
                          capture_output=True, text=True)
    seconds = time.time() - t0
    problems: list[str] = []
    n_compared = 0
    if proc.returncode != 0:
        problems.append(f"{' '.join(cmd)} exited with status {proc.returncode}")
    else:
        for spec in outputs:
            p = shadow / spec["path"]
            if not p.exists():
                problems.append(f"{spec['path']} was not written")
                continue
            if spec["keep_prior"] and p.stat().st_mtime_ns == prior.get(spec["path"]):
                problems.append(f"{spec['path']} was not rewritten")
                continue
            if spec["compare"]:
                problems.extend(compare(shadow, spec))
                n_compared += 1
    last = (proc.stdout.strip().splitlines() or [""])[-1][:100]
    if problems:
        print(f"FAIL {label} ({seconds:.0f} s)")
        for item in problems:
            print("   -", item)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
        for line in tail:
            print("   |", line)
        return False
    detail = f"{n_compared} file(s) identical" if outputs else last
    print(f"PASS {label}: {detail} ({seconds:.0f} s)")
    return True


def main() -> int:
    keep = "--keep" in sys.argv[1:]
    tmp = Path(tempfile.mkdtemp(prefix="regen_check_"))
    try:
        shadow = build_shadow(tmp)
        n_pass = n_fail = n_skip = 0
        for label, cmd, outputs in STEPS:
            if label == "figures" and os.environ.get("SKIP_FIGURES") == "1":
                print(f"SKIP {label} (SKIP_FIGURES=1)")
                n_skip += 1
                continue
            if run_step(shadow, label, cmd, outputs):
                n_pass += 1
            else:
                n_fail += 1
        print(f"\nregenerated outputs: {n_pass} steps passed, {n_fail} failed, {n_skip} skipped")
        if keep:
            print("scratch copy kept at", shadow)
        return 1 if n_fail else 0
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
