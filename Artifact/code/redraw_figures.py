"""Redraw the figures in results/figures from the stored results (no experiment is re-run).

    python code/redraw_figures.py

Draws, as PDF and PNG:
  fig_core     exp13_core_fig.py  (time-uniform validity + certifiable horizons)
  fig_anytime  exp11_figures.py   (G2 violation budget, shift-robust CS, matched calibration)
  fig_validity, fig_radius, fig_calibration, fig_pareto   from exp4_figures.py

Not redrawn: exp4_figures.fig_longrun and exp4_figures.fig_traj.  With the current code
both raise (fig_longrun draws a third panel on a two-panel grid; fig_traj reads the
per-step "preds" field of legacy trajectory records, which the window_anchor_v1 episode
runner no longer emits).  Their shipped files were drawn by an earlier revision of the
plotting code.  For the same reason `python code/exp4_figures.py` stops at fig_longrun.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    for script in ("exp13_core_fig.py", "exp11_figures.py"):
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
    sys.path.insert(0, HERE)
    import matplotlib
    matplotlib.use("Agg")
    import exp4_figures
    for name in ("fig_validity", "fig_radius", "fig_calibration", "fig_pareto"):
        getattr(exp4_figures, name)()


if __name__ == "__main__":
    main()
