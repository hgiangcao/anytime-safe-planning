"""Exp 1 — headline throughput table.

4 maps x 3 agent counts x {unweighted, lanes, tolls (ours-offline),
tolls-online (ours-online)} x 10 seeds x 1000 steps.
(The ES baseline is evaluated at its training condition in exp2.)

Checkpointed: re-run to resume. ~480 runs.
"""
import common

MAPS = ["empty-32x32", "random-32x32", "room-32x32", "warehouse-33x36"]
NS = [100, 250, 400]
METHODS = ["unweighted", "lanes", "tolls", "tolls-online"]


def conditions():
    out = []
    for mname in MAPS:
        for N in NS:
            for method in METHODS:
                for seed in common.SEEDS:
                    out.append(dict(map=mname, N=N, method=method, seed=seed))
    return out


if __name__ == "__main__":
    common.precompute_tolls([(mn, N) for mn in MAPS for N in NS])
    common.run_pending("exp1", conditions())
