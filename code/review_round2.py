"""Checkpointed prospective study for the 8 September 2026 Paper 17 review.

The protocol is frozen in REVIEW_ROUND2_PROTOCOL.md.  This driver uses one
condition at a time, writes every trace before advancing, and keeps negative
outcomes.  It is intentionally separate from the historical result corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "review_round2"
PROTOCOL_MD = ROOT / "REVIEW_ROUND2_PROTOCOL.md"
sys.path.insert(0, str(HERE))

from exp2c_es_strong import SepCMAES  # noqa: E402
from mapf import flow, guidance, maps  # noqa: E402
from mapf.pibt import dist_table  # noqa: E402
from mapf.sim import TaskModel, run_lifelong  # noqa: E402

MU = 1.0
GAMMA = 2.0
EPS_TIE = 1.0
CONTROL_PERIOD_S = 0.1
ATTRIBUTION_SEEDS = (20, 21, 22)
HORIZON_SEEDS = (30, 31, 32)
WAREHOUSE_SEEDS = (60, 61)
STALE_SEEDS = (70, 71, 72)
PLANNER_SEEDS = (80, 81, 82)


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):  # pragma: no cover
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def protocol_object() -> dict:
    return {
        "locked_at": "2026-09-08T15:30:00+07:00",
        "protocol_markdown": PROTOCOL_MD.name,
        "protocol_markdown_sha256": sha256_file(PROTOCOL_MD),
        "common": {"workers": 1, "eps_tie": EPS_TIE, "swap": True,
                   "negative_outcomes_retained": True},
        "attribution": {"map": "empty-32-32", "N": 400, "steps": 3000,
                        "seeds": list(ATTRIBUTION_SEEDS),
                        "random_initializations": 3},
        "search": {"map": "empty-32-32", "N": 400,
                   "parameterization": "one log-weight per directed edge",
                   "generations": 4, "population": 12,
                   "max_training_candidates_including_initial": 49,
                   "training_seeds": [10017, 10019], "training_steps": 500,
                   "validation_seeds": [50, 51], "validation_steps": 1000,
                   "test_seeds": [52, 53, 54], "test_steps": 3000},
        "long_horizon": {
            "cells": [["empty-32-32", 400], ["maze-32-32-4", 310],
                      ["warehouse-10-20-10-2-1", 800]],
            "methods": ["unweighted", "lanes", "tolls"],
            "seeds": list(HORIZON_SEEDS), "steps": 5000,
            "window_steps": 500,
            "collapse_definition": "zero completed goals in a nonoverlapping 500-step window"},
        "warehouse_resolution": {"map": "warehouse-10-20-10-2-1", "N": 800,
                                 "K_pick": [32, 64, 128, 256],
                                 "seeds": list(WAREHOUSE_SEEDS), "steps": 3000},
        "stale": {"map": "empty-32-32", "N": 400, "hotspot": 0.8,
                  "period_steps": 500, "control_period_s": CONTROL_PERIOD_S,
                  "seeds": list(STALE_SEEDS), "steps": 3000,
                  "modes": ["frozen", "synchronous", "stale"]},
        "independent_planner": {"implementation": "repository WindowedPP, not native PIBT",
                                "map": "empty-32-32", "N": 100,
                                "seeds": list(PLANNER_SEEDS), "steps": 3000,
                                "window": 8, "period": 4},
    }


def write_protocol() -> Path:
    path = OUT / "protocol.json"
    obj = protocol_object()
    if path.exists() and json.loads(path.read_text()) != obj:
        raise RuntimeError("protocol.json differs from the locked protocol")
    if not path.exists():
        atomic_json(path, obj)
    return path


def map_fingerprint(m) -> str:
    parts = [np.asarray(m["free"], np.uint8)]
    for key in ("pickups", "stations"):
        if key in m.get("meta", {}):
            parts.append(np.asarray(m["meta"][key], np.int64))
    return hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes()
                                   for x in parts)).hexdigest()


def guidance_path(map_name: str, N: int, gamma: float, init: str,
                  K_pick: int | None) -> Path:
    k = "default" if K_pick is None else str(K_pick)
    return OUT / "guidance" / f"{map_name}__N{N}__g{gamma:g}__{init}__K{k}.npz"


def solve_guidance(map_name: str, N: int, *, gamma=GAMMA, init="lane",
                   K_pick=None) -> tuple[np.ndarray, dict]:
    """Solve and checkpoint one flow initialization/resolution condition."""
    path = guidance_path(map_name, N, gamma, init, K_pick)
    if path.exists():
        with np.load(path, allow_pickle=True) as z:
            return z["weights"].copy(), json.loads(str(z["metadata_json"]))
    m = maps.get_map(map_name)
    net = flow.FlowNet(m, gamma=gamma, mu=MU)
    if init == "lane":
        bias = maps.crisscross_weights(m, guidance.SEED_BIAS_PENALTY)
    elif init == "symmetric":
        bias = None
    elif init.startswith("random"):
        seed = int(init.removeprefix("random"))
        rng = np.random.default_rng(2026090800 + seed)
        bias = np.exp(rng.uniform(-np.log(1.5), np.log(1.5), net.E))
    else:
        raise ValueError(init)
    t0 = time.perf_counter()
    sol = flow.calibrate_and_solve(net, N=N, K=K_pick, seed=0,
                                   init_bias=bias, tol=1e-3)
    elapsed = time.perf_counter() - t0
    meta = {k: sol[k] for k in ("lam", "obj", "gap", "linear_cost",
                                        "linear_optimum", "linear_residual",
                                        "linear_factor_bound", "iters",
                                        "in_transit", "mean_latency")}
    meta.update({"map": map_name, "map_sha256": map_fingerprint(m), "N": N,
                 "gamma": gamma, "mu": MU, "initialization": init,
                 "K_pick": K_pick, "solve_s": elapsed,
                 "weights_sha256": sha256_array(sol["tolls"]),
                 "flow_sha256": sha256_array(sol["f"])})
    atomic_npz(path, weights=np.asarray(sol["tolls"], np.float64),
               flow=np.asarray(sol["f"], np.float64),
               metadata_json=json.dumps(meta, sort_keys=True, default=json_default))
    return np.asarray(sol["tolls"], float), meta


_DIST_CACHE: dict[tuple, tuple[tuple[np.ndarray, np.ndarray], float]] = {}


def distance_table(m, weights):
    goal_cells = TaskModel(m, seed=0).goal_cells()
    whash = "unweighted" if weights is None else sha256_array(np.asarray(weights))
    ghash = "all" if goal_cells is None else sha256_array(np.asarray(goal_cells))
    key = (m["name"], whash, ghash)
    if key not in _DIST_CACHE:
        t0 = time.perf_counter()
        D = dist_table(m, weights, goal_cells=goal_cells)
        _DIST_CACHE[key] = (D, time.perf_counter() - t0)
    return _DIST_CACHE[key]


def quantiles(x) -> dict:
    x = np.asarray(x, float)
    if not len(x):
        return {"n": 0, "mean": None, "p50": None, "p90": None,
                "p99": None, "max": None}
    return {"n": int(len(x)), "mean": float(np.mean(x)),
            "p50": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
            "p99": float(np.quantile(x, 0.99)), "max": float(np.max(x))}


def trace_metrics(result: dict) -> dict:
    events = np.asarray(result["goal_events"], np.int64)
    steps = int(result["steps"])
    window = 500
    t = events[:, 0] if len(events) else np.empty(0, int)
    counts = [int(np.sum((t > a) & (t <= min(a + window, steps))))
              for a in range(0, steps, window)]
    collapse = [i for i, n in enumerate(counts) if n == 0]
    recovery = None
    if collapse:
        recovery = next((j for j in range(collapse[0] + 1, len(counts))
                         if counts[j] > 0), None)
    first_half = int(np.sum(t <= steps // 2))
    last_half = int(np.sum(t > steps // 2))
    return {
        "window_steps": window, "completed_per_window": counts,
        "collapse_window_indices": collapse,
        "first_recovery_window_index": recovery,
        "first_half_throughput": first_half / max(steps // 2, 1),
        "last_half_throughput": last_half / max(steps - steps // 2, 1),
        "last_half_over_whole": ((last_half / (steps - steps // 2)) /
                                 max(result["throughput"], 1e-12)),
        "task_delay_steps": quantiles(events[:, 3] if len(events) else []),
        "moves_per_completed_task": quantiles(events[:, 4] if len(events) else []),
    }


def condition_key(condition: dict) -> str:
    raw = json.dumps(condition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def run_and_save(family: str, label: str, map_name: str, N: int, steps: int,
                 seed: int, weights=None, *, guidance_meta=None, online=None,
                 hotspot=0.0, planner="pibt", planner_kw=None) -> dict:
    condition = {"family": family, "label": label, "map": map_name, "N": N,
                 "steps": steps, "seed": seed, "hotspot": hotspot,
                 "planner": planner, "planner_kw": planner_kw,
                 "eps_tie": EPS_TIE, "swap": True,
                 "weights_sha256": (None if weights is None
                                     else sha256_array(np.asarray(weights))),
                 "online_mode": None if online is None else online.get("mode")}
    key = f"{label}__{map_name}__N{N}__s{seed}__{condition_key(condition)}"
    result_path = OUT / family / f"{key}.json"
    trace_path = OUT / family / "traces" / f"{key}.npz"
    if result_path.exists() and trace_path.exists():
        return json.loads(result_path.read_text())
    m = maps.get_map(map_name)
    D, build_s = distance_table(m, weights)
    result = run_lifelong(m, N, steps, seed=seed, D=D, edge_weights=weights,
                          online=online, hotspot=hotspot, record_full=True,
                          eps_tie=EPS_TIE, swap=True, planner=planner,
                          planner_kw=planner_kw)
    atomic_npz(trace_path,
               position_trace=np.asarray(result["position_trace"], np.int32),
               goal_trace=np.asarray(result["goal_trace"], np.int32),
               goal_events=np.asarray(result["goal_events"], np.int64),
               condition_json=json.dumps(condition, sort_keys=True))
    metrics = trace_metrics(result)
    deployment_s = steps * CONTROL_PERIOD_S
    if online is not None and online.get("mode") == "synchronous":
        deployment_s += result["resolve_time"]
    rec = {
        "condition": condition,
        "throughput_tasks_per_step": result["throughput"],
        "completed": result["completed"], "wall_time_s": result["wall_time"],
        "initial_distance_table_build_s": build_s,
        "resolve_and_rebuild_s": result["resolve_time"],
        "n_resolves": result["n_resolves"],
        "update_events": result["update_events"],
        "deployment_timing_model": {
            "control_period_s": CONTROL_PERIOD_S,
            "elapsed_s": deployment_s,
            "completed_tasks_per_wall_s": result["completed"] / deployment_s,
            "assumption": ("synchronous pauses charged" if online is not None and
                           online.get("mode") == "synchronous" else
                           "offline setup excluded; stale compute assumed on one side worker")},
        "trace_metrics": metrics,
        "trace": {"path": str(trace_path.relative_to(ROOT)),
                  "sha256": sha256_file(trace_path),
                  "position_shape": list(np.asarray(result["position_trace"]).shape),
                  "goal_shape": list(np.asarray(result["goal_trace"]).shape),
                  "n_goal_events": int(len(result["goal_events"]))},
        "guidance": guidance_meta,
        "process_peak_rss_bytes_so_far": peak_rss_bytes(),
    }
    atomic_json(result_path, rec)
    print(f"{family}/{result_path.name}: throughput={result['throughput']:.4f} "
          f"tasks={result['completed']}", flush=True)
    return rec


def run_attribution() -> None:
    write_protocol()
    map_name, N, steps = "empty-32-32", 400, 3000
    m = maps.get_map(map_name)
    fields: list[tuple[str, np.ndarray | None, dict | None]] = [
        ("unweighted", None, None),
        ("seed_mild_1p5", maps.crisscross_weights(m, 1.5),
         {"construction": "mild lane initialization used directly; no optimization"}),
        ("lanes_5", maps.crisscross_weights(m, 5.0),
         {"construction": "strong handcrafted lane baseline; no optimization"}),
    ]
    for init in ("lane", "symmetric", "random0", "random1", "random2"):
        w, meta = solve_guidance(map_name, N, gamma=GAMMA, init=init)
        fields.append((f"fluid_g2_{init}", w, meta))
    for init in ("lane", "symmetric"):
        w, meta = solve_guidance(map_name, N, gamma=0.0, init=init)
        fields.append((f"fluid_g0_{init}", w, meta))
    for label, weights, meta in fields:
        for seed in ATTRIBUTION_SEEDS:
            run_and_save("attribution", label, map_name, N, steps, seed,
                         weights, guidance_meta=meta)
    _DIST_CACHE.clear()


def _search_eval(theta, seeds, steps) -> list[float]:
    m = maps.get_map("empty-32-32")
    weights = np.exp(np.clip(np.asarray(theta, float), -2.0, 2.0))
    D = dist_table(m, weights)
    return [float(run_lifelong(m, 400, steps, seed=s, D=D,
                               eps_tie=EPS_TIE)["throughput"])
            for s in seeds]


def run_search() -> None:
    write_protocol()
    search_dir = OUT / "search"
    state_path = search_dir / "state.npz"
    result_path = search_dir / "summary.json"
    initial_w, initial_meta = solve_guidance("empty-32-32", 400,
                                            gamma=GAMMA, init="lane")
    theta0 = np.log(initial_w)
    es = SepCMAES(theta0, 0.25, 12, seed=1701)
    if state_path.exists():
        with np.load(state_path, allow_pickle=True) as z:
            es.mean = z["mean"].copy(); es.sigma = float(z["sigma"])
            es.C = z["C"].copy(); es.pc = z["pc"].copy()
            es.ps = z["ps"].copy(); es.gen = int(z["gen"])
            archive = [x.copy() for x in z["archive_theta"]]
            train_runs = json.loads(str(z["training_runs_json"]))
            validation_runs = json.loads(str(z["validation_runs_json"]))
    else:
        archive = [theta0.copy()]
        runs = _search_eval(theta0, (10017, 10019), 500)
        train_runs = [runs]
        validation_runs = []
        atomic_npz(state_path, mean=es.mean, sigma=es.sigma, C=es.C,
                   pc=es.pc, ps=es.ps, gen=es.gen,
                   archive_theta=np.asarray(archive),
                   training_runs_json=json.dumps(train_runs),
                   validation_runs_json=json.dumps(validation_runs))
        print(f"search initial retained: {np.mean(runs):.4f}", flush=True)
    while es.gen < 4:
        X, Y = es.ask()
        fits = []
        for x in X:
            runs = _search_eval(x, (10017, 10019), 500)
            archive.append(x.copy()); train_runs.append(runs)
            fits.append(-float(np.mean(runs)))
        es.tell(Y, np.asarray(fits))
        atomic_npz(state_path, mean=es.mean, sigma=es.sigma, C=es.C,
                   pc=es.pc, ps=es.ps, gen=es.gen,
                   archive_theta=np.asarray(archive),
                   training_runs_json=json.dumps(train_runs),
                   validation_runs_json=json.dumps(validation_runs))
        print(f"search generation {es.gen}/4: gen-best={-min(fits):.4f} "
              f"retained-best={max(map(np.mean, train_runs)):.4f}", flush=True)
    while len(validation_runs) < len(archive):
        i = len(validation_runs)
        validation_runs.append(_search_eval(archive[i], (50, 51), 1000))
        atomic_npz(state_path, mean=es.mean, sigma=es.sigma, C=es.C,
                   pc=es.pc, ps=es.ps, gen=es.gen,
                   archive_theta=np.asarray(archive),
                   training_runs_json=json.dumps(train_runs),
                   validation_runs_json=json.dumps(validation_runs))
        print(f"search validation {i + 1}/{len(archive)}", flush=True)
    train_means = np.asarray([np.mean(x) for x in train_runs])
    validation_means = np.asarray([np.mean(x) for x in validation_runs])
    train_selected = int(np.argmax(train_means))
    validation_selected = int(np.argmax(validation_means))
    terminal_validation = _search_eval(es.mean, (50, 51), 1000)
    selected_theta = archive[validation_selected]
    atomic_npz(search_dir / "selected.npz", initial_theta=theta0,
               validation_selected_theta=selected_theta,
               training_selected_theta=archive[train_selected],
               terminal_mean=es.mean,
               initial_weights=initial_w,
               selected_weights=np.exp(np.clip(selected_theta, -2.0, 2.0)))
    for label, theta in (("retained_initial", theta0),
                         ("validation_selected", selected_theta)):
        weights = np.exp(np.clip(theta, -2.0, 2.0))
        for seed in (52, 53, 54):
            run_and_save("search_test", label, "empty-32-32", 400, 3000,
                         seed, weights,
                         guidance_meta={"source": "incumbent_search",
                                        "candidate_index": (0 if label == "retained_initial"
                                                            else validation_selected)})
    rec = {
        "initial_candidate_evaluated_and_retained": True,
        "n_training_candidates": len(archive), "generations": es.gen,
        "population": 12, "training_runs": train_runs,
        "validation_runs": validation_runs,
        "training_selected_index": train_selected,
        "validation_selected_index": validation_selected,
        "initial_training_mean": float(train_means[0]),
        "initial_validation_mean": float(validation_means[0]),
        "training_selected_mean": float(train_means[train_selected]),
        "validation_selected_mean": float(validation_means[validation_selected]),
        "terminal_mean_validation_runs": terminal_validation,
        "terminal_mean_was_selection_eligible": False,
        "initial_guidance": initial_meta,
        "state_sha256": sha256_file(state_path),
        "selected_sha256": sha256_file(search_dir / "selected.npz"),
    }
    atomic_json(result_path, rec)
    _DIST_CACHE.clear()


def run_horizon() -> None:
    write_protocol()
    for map_name, N in (("empty-32-32", 400), ("maze-32-32-4", 310),
                        ("warehouse-10-20-10-2-1", 800)):
        m = maps.get_map(map_name)
        tolls, meta = solve_guidance(map_name, N, gamma=GAMMA, init="lane")
        for label, weights, gmeta in (
                ("unweighted", None, None),
                ("lanes", maps.crisscross_weights(m, 5.0),
                 {"construction": "handcrafted crisscross penalty 5"}),
                ("tolls", tolls, meta)):
            for seed in HORIZON_SEEDS:
                run_and_save("long_horizon", label, map_name, N, 5000, seed,
                             weights, guidance_meta=gmeta)
        _DIST_CACHE.clear()


def run_warehouse() -> None:
    write_protocol()
    fields = {}
    metas = {}
    for K in (32, 64, 128, 256):
        fields[K], metas[K] = solve_guidance("warehouse-10-20-10-2-1", 800,
                                            gamma=GAMMA, init="lane", K_pick=K)
    ref = fields[256] / np.mean(fields[256])
    sensitivity = {}
    for K, w in fields.items():
        wn = w / np.mean(w)
        sensitivity[str(K)] = {
            "relative_l2_to_K256": float(np.linalg.norm(wn - ref) /
                                          max(np.linalg.norm(ref), 1e-12)),
            "pearson_to_K256": float(np.corrcoef(wn, ref)[0, 1]),
            "guidance": metas[K],
        }
        for seed in WAREHOUSE_SEEDS:
            run_and_save("warehouse_resolution", f"K{K}",
                         "warehouse-10-20-10-2-1", 800, 3000, seed, w,
                         guidance_meta=metas[K])
        _DIST_CACHE.clear()
    atomic_json(OUT / "warehouse_resolution" / "sensitivity.json", sensitivity)


def online_config(map_name: str, mode: str) -> dict:
    m = maps.get_map(map_name)
    return {"period": 500, "window": 6000, "K": 100000, "fw_iters": 30,
            "net": flow.FlowNet(m, gamma=GAMMA, mu=MU), "mode": mode,
            "control_period_s": CONTROL_PERIOD_S,
            "init_bias": maps.crisscross_weights(m,
                                                  guidance.SEED_BIAS_PENALTY)}


def run_stale() -> None:
    write_protocol()
    tolls, meta = solve_guidance("empty-32-32", 400, gamma=GAMMA, init="lane")
    for seed in STALE_SEEDS:
        run_and_save("stale", "frozen", "empty-32-32", 400, 3000, seed,
                     tolls, guidance_meta=meta, hotspot=0.8)
        for mode in ("synchronous", "stale"):
            run_and_save("stale", mode, "empty-32-32", 400, 3000, seed,
                         tolls, guidance_meta=meta,
                         online=online_config("empty-32-32", mode), hotspot=0.8)
    _DIST_CACHE.clear()


def run_planner() -> None:
    write_protocol()
    m = maps.get_map("empty-32-32")
    tolls, meta = solve_guidance("empty-32-32", 100, gamma=GAMMA, init="lane")
    for label, weights, gmeta in (
            ("unweighted", None, None),
            ("lanes", maps.crisscross_weights(m, 5.0),
             {"construction": "handcrafted crisscross penalty 5"}),
            ("tolls", tolls, meta)):
        for seed in PLANNER_SEEDS:
            run_and_save("independent_planner", label, "empty-32-32", 100,
                         3000, seed, weights, guidance_meta=gmeta,
                         planner="wpp", planner_kw={"window": 8, "period": 4})
    _DIST_CACHE.clear()


def load_family(family: str) -> list[dict]:
    d = OUT / family
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))
            if p.name not in ("sensitivity.json", "summary.json")]


def summarize_groups(groups: dict[str, list[dict]]) -> dict:
    out = {}
    for label, vals in sorted(groups.items()):
        t = np.asarray([x["throughput_tasks_per_step"] for x in vals])
        tail = np.asarray([x["trace_metrics"]["last_half_over_whole"] for x in vals])
        out[label] = {"n": len(vals), "throughput_mean": float(t.mean()),
                      "throughput_values": t.tolist(),
                      "last_half_over_whole_mean": float(tail.mean()),
                      "collapse_runs": int(sum(bool(x["trace_metrics"]["collapse_window_indices"])
                                               for x in vals))}
    return out


def grouped_summary(rows: list[dict]) -> dict:
    """Summarize both treatment labels and map-specific treatment cells.

    The long-horizon family spans three maps, so a label-only aggregate hides
    important negative map-level outcomes.  Keep both views in the released
    summary rather than relying on a manuscript-only reconstruction.
    """
    by_label: dict[str, list[dict]] = {}
    by_map_label: dict[str, list[dict]] = {}
    for r in rows:
        condition = r["condition"]
        label = condition["label"]
        map_name = condition["map"]
        by_label.setdefault(label, []).append(r)
        by_map_label.setdefault(f"{map_name}::{label}", []).append(r)
    return {"by_label": summarize_groups(by_label),
            "by_map_label": summarize_groups(by_map_label)}


def make_summary() -> dict:
    families = {name: load_family(name) for name in
                ("attribution", "search_test", "long_horizon",
                 "warehouse_resolution", "stale", "independent_planner")}
    expected = {"attribution": 30, "search_test": 6, "long_horizon": 27,
                "warehouse_resolution": 8, "stale": 9,
                "independent_planner": 9}
    counts = {k: len(v) for k, v in families.items()}
    out = {"complete": counts == expected and (OUT / "search" / "summary.json").exists(),
           "expected_counts": expected, "observed_counts": counts,
           "families": {k: grouped_summary(v) for k, v in families.items()},
           "search": (json.loads((OUT / "search" / "summary.json").read_text())
                      if (OUT / "search" / "summary.json").exists() else None),
           "protocol_sha256": sha256_file(OUT / "protocol.json")}
    atomic_json(OUT / "summary.json", out)
    return out


def audit_results() -> dict:
    summary = make_summary()
    if not summary["complete"]:
        raise AssertionError(f"incomplete results: {summary['observed_counts']}")
    trace_records = []
    for family in summary["expected_counts"]:
        trace_records.extend(load_family(family))
    for rec in trace_records:
        path = ROOT / rec["trace"]["path"]
        if sha256_file(path) != rec["trace"]["sha256"]:
            raise AssertionError(f"trace hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            pos, goals, events = z["position_trace"], z["goal_trace"], z["goal_events"]
            if list(pos.shape) != rec["trace"]["position_shape"]:
                raise AssertionError(f"position shape mismatch: {path}")
            if pos.shape != goals.shape or len(events) != rec["completed"]:
                raise AssertionError(f"trace/event mismatch: {path}")
            if pos.shape[0] != rec["condition"]["steps"] + 1:
                raise AssertionError(f"horizon mismatch: {path}")
    files = [p for p in OUT.rglob("*") if p.is_file() and
             not p.name.endswith(".tmp") and p.name != "manifest.json"]
    manifest = {"status": "PASS", "n_trace_records": len(trace_records),
                "protocol_sha256": sha256_file(OUT / "protocol.json"),
                "source_sha256": sha256_file(Path(__file__)),
                "artifacts": {str(p.relative_to(ROOT)): sha256_file(p)
                              for p in sorted(files)}}
    atomic_json(OUT / "manifest.json", manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("protocol", "attribution", "search",
                                      "horizon", "warehouse", "stale",
                                      "planner", "summary", "audit", "all"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("protocol", "all"):
        p = write_protocol(); print(f"protocol {p} sha256={sha256_file(p)}")
    for name, fn in (("attribution", run_attribution), ("search", run_search),
                     ("horizon", run_horizon), ("warehouse", run_warehouse),
                     ("stale", run_stale), ("planner", run_planner)):
        if a.stage in (name, "all"):
            fn()
    if a.stage in ("summary", "all"):
        print(json.dumps(make_summary(), indent=2, default=json_default))
    if a.stage in ("audit", "all"):
        print(json.dumps(audit_results(), indent=2, default=json_default))


if __name__ == "__main__":
    main()
