"""Emit one tiny, deterministic window_anchor_v1 schema audit artifact.

This is a controller-contract smoke test, not a performance experiment.  It is
kept deliberately small and records exact seeds, dependency versions, source
hashes, and start/end timestamps alongside every execution label.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from envloop import make_calibrator, run_episode
from predictor import Predictor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "window_anchor_smoke.json"
WARM_SEED = 20260907
EPISODE_SEED = 20260908


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    started = now()
    warm = np.abs(np.random.default_rng(WARM_SEED).normal(size=500))
    warm_hash = hashlib.sha256(np.asarray(warm, dtype="<f8").tobytes()).hexdigest()
    predictor = Predictor(cv_only=True)
    record = run_episode(
        "sf", make_calibrator("cs", 0.10, warm), "cs", predictor,
        np.random.default_rng(EPISODE_SEED), n_windows=3, vehicle="di",
        latency=0, act_noise=0.0, tracking_margin=0.0,
    )
    payload = {
        "artifact_schema_version": 1,
        "purpose": "contract_smoke_not_performance_evidence",
        "started_utc": started,
        "completed_utc": now(),
        "invocation": "../.venv/bin/python code/contract_smoke.py",
        "randomness": {
            "warm_score_seed": WARM_SEED,
            "episode_seed": EPISODE_SEED,
            "warm_score_generator": "abs(PCG64.standard_normal(500))",
            "warm_score_float64_le_sha256": warm_hash,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": platform.system(),
            "machine": platform.machine(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "torch": version("torch"),
        },
        "source_sha256": {
            name: digest(ROOT / "code" / name)
            for name in ("contract_smoke.py", "envloop.py", "planner.py",
                         "csq.py", "predictor.py", "sims.py")
        },
        "input_artifacts": [],
        "record": record,
    }
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {OUT.relative_to(ROOT)}; steps={record['steps']}; "
          f"windows={record['n_windows']}")


if __name__ == "__main__":
    main()
