"""Exp 6 — ablations on empty-32x32 and random-32x32 at N=400, 5 seeds:
  (a) bidirectional coupling gamma in {0, 1, 2, 4}   (2 = default)
  (b) BPR exponent beta in {1, 2, 4}                 (2 = default)
  (c) online re-solve period in {100, 200, 400}      (200 = default)
"""
import common

MAPS = ["empty-32x32", "random-32x32"]
N = 400
SEEDS5 = list(range(5))


def conditions():
    out = []
    for mname in MAPS:
        for g in [0.0, 1.0, 2.0, 4.0]:
            for seed in SEEDS5:
                out.append(dict(map=mname, N=N, method="tolls", seed=seed,
                                alpha=1.0, beta=2, gamma=g, label=f"gamma={g}"))
        for b in [1, 2, 4]:
            for seed in SEEDS5:
                out.append(dict(map=mname, N=N, method="tolls", seed=seed,
                                alpha=1.0, beta=b, gamma=2.0, label=f"beta={b}"))
        for period in [100, 200, 400]:
            for seed in SEEDS5:
                out.append(dict(map=mname, N=N, method="tolls-online",
                                seed=seed, period=period,
                                label=f"period={period}"))
        # FW initialization: symmetric start (no lane seeding) vs default
        for seed in SEEDS5:
            out.append(dict(map=mname, N=N, method="tolls", seed=seed,
                            alpha=1.0, beta=2, gamma=2.0, seed_lanes=False,
                            label="sym-init"))
    return out


if __name__ == "__main__":
    pre = []
    for mname in MAPS:
        for g in [0.0, 1.0, 2.0, 4.0]:
            pre.append((mname, N, dict(alpha=1.0, beta=2, gamma=g)))
        for b in [1, 2, 4]:
            pre.append((mname, N, dict(alpha=1.0, beta=b, gamma=2.0)))
        pre.append((mname, N, dict(alpha=1.0, beta=2, gamma=2.0,
                                   seed_lanes=False)))
    for mname, NN, kw in pre:
        common.precompute_tolls([(mname, NN)], **kw)
    common.run_pending("exp6", conditions())
