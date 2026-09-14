"""Guard against shipping stale artifacts.

A stale figure left in paper/figures from a previous run is silently picked up
by LaTeX and looks identical to a fresh one. This script fails loudly if any
figure or generated table referenced by the paper is missing, or is older than
the results it is supposed to depict.

Run before every compile:  ../.venv/bin/python code/check_provenance.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
PAPER = os.path.join(PROJ, "paper")
SUMMARY = os.path.join(PROJ, "results", "summary_v2.json")


def main():
    if not os.path.exists(SUMMARY):
        print("FAIL: results/summary_v2.json missing (run code/aggregate_v2.py)")
        return 1
    t_sum = os.path.getmtime(SUMMARY)
    tex = open(os.path.join(PAPER, "root.tex")).read()
    problems, checked = [], 0

    for rel in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        path = os.path.join(PAPER, rel if rel.endswith(".pdf") else rel + ".pdf")
        checked += 1
        if not os.path.exists(path):
            problems.append(f"MISSING figure: {rel}")
        elif os.path.getmtime(path) < t_sum - 1:
            problems.append(f"STALE figure (older than summary_v2.json): {rel}")

    for gen in ("tables.tex", "numbers.tex"):
        path = os.path.join(PAPER, gen)
        checked += 1
        if not os.path.exists(path):
            problems.append(f"MISSING generated file: {gen}")
        elif os.path.getmtime(path) < t_sum - 1:
            problems.append(f"STALE generated file: {gen}")

    # a figure left in the directory that the paper does not reference is a
    # leftover from an earlier run: report it, but it cannot be shipped, so it
    # is a warning rather than a failure.
    used = {os.path.basename(r)[:-4] if r.endswith(".pdf") else os.path.basename(r)
            for r in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)}
    figdir = os.path.join(PAPER, "figures")
    orphans = []
    if os.path.isdir(figdir):
        for fn in sorted(os.listdir(figdir)):
            if fn.endswith(".pdf") and fn[:-4] not in used:
                orphans.append(fn)

    if orphans:
        print("note: generated but not referenced by the paper: "
              + ", ".join(orphans))
    if problems:
        print(f"PROVENANCE FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print("  -", p)
        return 1
    print(f"provenance OK ({checked} artifacts, all newer than summary_v2.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
