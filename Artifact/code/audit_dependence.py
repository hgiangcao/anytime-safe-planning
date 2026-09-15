"""Dependence-aware uncertainty for the 14 long ETH/UCY episodes.

Windows from one pedestrian trajectory are a single cluster.  This audit never
uses a window-level binomial/Wilson model: it resamples complete episodes and
also reports leave-one-scene-out sensitivity.  It only re-analyses stored JSON;
no simulator or predictor is run.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import pickle
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = os.path.join(ROOT, "results", "long")
DATA = os.path.join(ROOT, "results", "data")
OUT = os.path.join(ROOT, "results", "dependence_aware_long_real.json")
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
B = 10000


def scene_labels():
    labels = []
    for scene in SCENES:
        with open(os.path.join(DATA, f"real_{scene}.pkl"), "rb") as f:
            labels.extend([scene] * len(pickle.load(f)["long"]))
    return labels


def pairs(record, metric):
    windows = record["windows"]
    if metric == "certified_window_fraction":
        return sum(rad is not None for _, rad in windows), len(windows)
    if metric == "certified_window_violation_rate":
        finite = [(s, rad) for s, rad in windows if rad is not None]
        return sum(s > rad for s, rad in finite), len(finite)
    if metric == "collision_steps_per_1000":
        return 1000.0 * record["n_coll_steps"], record["steps"]
    if metric == "fallback_step_fraction":
        return record["fallback_steps"], record["steps"]
    if metric == "goals_per_episode":
        return record["goals"], 1
    raise ValueError(metric)


def ratio(nums, dens, idx=None):
    if idx is not None:
        nums, dens = nums[idx], dens[idx]
    den = float(np.sum(dens))
    return None if den == 0.0 else float(np.sum(nums) / den)


def summarize(path, labels):
    records = json.load(open(path))["records"]
    if len(records) != len(labels):
        raise ValueError(f"{path}: expected {len(labels)} scene-labelled records")
    out = {}
    for metric in ("certified_window_fraction", "certified_window_violation_rate",
                   "collision_steps_per_1000", "fallback_step_fraction",
                   "goals_per_episode"):
        nd = np.asarray([pairs(r, metric) for r in records], dtype=float)
        nums, dens = nd[:, 0], nd[:, 1]
        point = ratio(nums, dens)
        if point is None:
            out[metric] = {"estimate": None, "episode_cluster_bootstrap_ci95": None,
                           "leave_one_scene_out_range": None}
            continue
        rng = np.random.default_rng(zlib.crc32(f"dep|{os.path.basename(path)}|{metric}".encode()))
        draw = rng.integers(0, len(records), size=(B, len(records)))
        boot = nums[draw].sum(axis=1) / dens[draw].sum(axis=1)
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
        # Also make each source scene one higher-level cluster. With only five
        # clusters this percentile interval is deliberately coarse.
        lab = np.asarray(labels)
        scene_num = np.asarray([nums[lab == s].sum() for s in SCENES])
        scene_den = np.asarray([dens[lab == s].sum() for s in SCENES])
        sdraw = rng.integers(0, len(SCENES), size=(B, len(SCENES)))
        sboot = scene_num[sdraw].sum(axis=1) / scene_den[sdraw].sum(axis=1)
        sci = np.quantile(sboot, [0.025, 0.975]).tolist()
        loso = []
        for scene in SCENES:
            loso.append(ratio(nums, dens, np.where(lab != scene)[0]))
        out[metric] = {
            "estimate": point,
            "episode_cluster_bootstrap_ci95": [float(ci[0]), float(ci[1])],
            "scene_cluster_bootstrap_ci95": [float(sci[0]), float(sci[1])],
            "leave_one_scene_out_range": [float(min(loso)), float(max(loso))],
        }
    return {"n_episode_clusters": len(records), "n_scene_clusters": len(set(labels)),
            "metrics": out}


def main():
    labels = scene_labels()
    conditions = {}
    for path in sorted(glob.glob(os.path.join(LONG, "long_real_*.json"))):
        conditions[os.path.basename(path).removesuffix(".json")] = summarize(path, labels)
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "execution_contract": "legacy_fresh_prediction_recentered_v0",
        "warning": (
            "Intervals describe archived runs only. They preserve within-episode "
            "dependence but five scenes are too few for precise cross-scene inference."),
        "bootstrap": {
            "units": ["complete episode/trajectory", "source scene"],
            "replicates": B, "interval": "percentile 95%",
            "seed_rule": "crc32('dep|artifact_basename|metric')",
            "warning": "The scene-cluster interval has only five source clusters."
        },
        "scene_order": labels,
        "conditions": conditions,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    key = "long_real_cs_0.05"
    print(json.dumps({key: conditions[key]}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
