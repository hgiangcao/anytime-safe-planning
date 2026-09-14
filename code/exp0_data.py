"""exp0: data generation.

1. Seeded agent-only rollouts for each simulator: GRU-train / calibration / replay
   splits (disjoint seeds).
2. Train the GRU residual predictor per simulator (minutes, CPU).
3. Compute calibration window scores (identical protocol to deployment).
4. Attempt to fetch the tiny public ETH pedestrian txt (BIWI 'eth' scene, ~117 KB,
   Trajectron++ mirror of the standard crowds/ewap annotation, frame ped x y at
   2.5 Hz).  On success: interpolate to the 0.25 s control grid, rotate so the
   dominant walking direction is the x-axis (matching the sims' crossing flow),
   split into 17.5 s chunks -> even chunks = calibration, odd = replay pool.
   AUTOMATIC FALLBACK: if the fetch fails, the 'eth' regime is skipped and the
   replay regime (recorded sim rollouts) stands alone; a note is written to meta.

Run:  .venv/bin/python code/exp0_data.py
Idempotent: skips finished stages (checks output files).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import urllib.request
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2compat  # noqa: F401  (numpy>=2 pickles under numpy 1.x)
from envloop import T_MAX, window_scores_from_rollout
from predictor import H, L, Predictor, train
from sims import DT, OrcaLite, SocialForce, record_rollout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
MODELS = os.path.join(ROOT, "results", "models")
os.makedirs(DATA, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

N_TRAIN, N_CAL, N_REPLAY = 50, 150, 100
T_ROLL = 120  # calibration/train rollout length (14 scored windows each)
N_LONG, W_LONG = 40, 120      # long-horizon replay rollouts: 120 scored windows
T_LONG = 5 + W_LONG * 8 + 1   # = L + W_LONG*H + 1 control steps (~4 min at 4 Hz)

RAW_BASE = ("https://raw.githubusercontent.com/StanfordASL/Trajectron-plus-plus/"
            "master/experiments/pedestrians/raw/raw/all_data")
ETH_URL = f"{RAW_BASE}/biwi_eth.txt"

# The five standard ETH/UCY scenes, used for a leave-one-scene-out real-data
# study: for each held-out scene the predictor is trained and the calibration
# pool is built on the OTHER four scenes only (no leakage), and deployment
# replays the held-out scene.  All annotations are 'frame ped x y' at 2.5 Hz
# (frame step 10 of 25 fps), the same format as the biwi_eth pilot.
SCENES = {"eth": "biwi_eth.txt", "hotel": "biwi_hotel.txt",
          "univ": "students003.txt", "zara1": "crowds_zara01.txt",
          "zara2": "crowds_zara02.txt"}
W_REAL_LONG = 60                   # long real episodes: 60 windows ~ 2 min
T_REAL_LONG = 5 + W_REAL_LONG * 8 + 1


def make_sim(kind, rng):
    n = int(rng.integers(5, 12))
    return SocialForce(n, rng) if kind == "sf" else OrcaLite(n, rng)


def gen_rollouts(kind, seed0, count, T):
    out = []
    for i in range(count):
        rng = np.random.default_rng(seed0 + i)
        sim = make_sim(kind, rng)
        out.append(record_rollout(sim, T))
    return out


def stage_sims():
    meta = {}
    for kind, base in [("sf", 10000), ("orca", 20000)]:
        mpath = os.path.join(MODELS, f"gru_{kind}.pt")
        cpath = os.path.join(DATA, f"cal_{kind}.npy")
        rpath = os.path.join(DATA, f"replay_{kind}.pkl")
        if os.path.exists(mpath) and os.path.exists(cpath) and os.path.exists(rpath):
            print(f"[skip] {kind} already generated")
            continue
        print(f"[gen] {kind}: rollouts ...")
        train_r = gen_rollouts(kind, base, N_TRAIN, T_ROLL)
        cal_r = gen_rollouts(kind, base + 1000, N_CAL, T_ROLL)
        replay_r = gen_rollouts(kind, base + 2000, N_REPLAY, T_MAX + 1)
        print(f"[train] GRU {kind} ...")
        info = train(train_r, mpath, epochs=4, seed=base)
        print(f"        {info}")
        pred = Predictor(mpath)
        scores = []
        for P, ev in cal_r:
            scores.extend(window_scores_from_rollout(P, ev, pred))
        np.save(cpath, np.array(scores))
        with open(rpath, "wb") as f:
            pickle.dump(replay_r, f)
        meta[kind] = {"gru": info, "n_cal_scores": len(scores),
                      "cal_score_mean": float(np.mean(scores)),
                      "cal_score_q90": float(np.quantile(scores, 0.9))}
        print(f"[done] {kind}: {len(scores)} calibration scores")
    return meta


def stage_long():
    """Long-horizon replay rollouts (agent-only, robot invisible) for the
    continuous-deployment experiment.  Independent of the model/calibration
    stage so it can be added without retraining anything."""
    info = {}
    for kind, base in [("sf", 10000), ("orca", 20000)]:
        lpath = os.path.join(DATA, f"replay_long_{kind}.pkl")
        if os.path.exists(lpath):
            print(f"[skip] long rollouts {kind}")
            continue
        print(f"[gen] long rollouts {kind}: {N_LONG} x {T_LONG} steps ...")
        long_r = gen_rollouts(kind, base + 3000, N_LONG, T_LONG)
        with open(lpath, "wb") as f:
            pickle.dump(long_r, f)
        info[kind] = {"n_long_rollouts": len(long_r), "long_rollout_steps": T_LONG}
    return info


def _fetch(fname):
    raw = os.path.join(DATA, fname)
    if not os.path.exists(raw):
        req = urllib.request.Request(f"{RAW_BASE}/{fname}",
                                     headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=25) as r:
            blob = r.read()
        if len(blob) < 10000:
            raise RuntimeError(f"{fname}: fetch too small")
        with open(raw, "wb") as f:
            f.write(blob)
    return raw


def _interp_scene(raw):
    """'frame ped x y' at 2.5 Hz -> (tracks [T,n,2] on the DT grid, present
    [T,n] bool), centred and rotated so the dominant walking direction is +x."""
    A = np.loadtxt(raw)
    tsec = A[:, 0] / 25.0
    xy = A[:, 2:4] - A[:, 2:4].mean(axis=0)
    disp = []
    for pid in np.unique(A[:, 1]):
        m = A[:, 1] == pid
        if m.sum() >= 2:
            disp.append(xy[m][-1] - xy[m][0])
    disp = np.array(disp)
    w, V = np.linalg.eigh(disp.T @ disp)
    xy = xy @ V[:, ::-1]                      # principal direction -> x
    grid = np.arange(tsec.min(), tsec.max(), DT)
    PARK = np.array([60.0, 60.0])
    ids = np.unique(A[:, 1])
    tracks = np.full((len(grid), len(ids), 2), PARK, dtype=float)
    present = np.zeros((len(grid), len(ids)), dtype=bool)
    for j, pid in enumerate(ids):
        m = A[:, 1] == pid
        tt, pp = tsec[m], xy[m]
        o = np.argsort(tt)
        tt, pp = tt[o], pp[o]
        gm = (grid >= tt[0]) & (grid <= tt[-1])
        if gm.sum() < 2:
            continue
        tracks[gm, j, 0] = np.interp(grid[gm], tt, pp[:, 0])
        tracks[gm, j, 1] = np.interp(grid[gm], tt, pp[:, 1])
        present[gm, j] = True
    return tracks, present


def _chunk(tracks, present, CH, min_agents=3, min_mean=2.0):
    """Cut a scene into replayable chunks of CH control steps.  Pedestrian
    entries/exits are recorded as respawn events so the scoring protocol excludes
    those agents' windows, exactly as in the simulated regimes."""
    out = []
    for c in range(len(tracks) // CH):
        sl = slice(c * CH, (c + 1) * CH)
        pres = present[sl]
        keep = pres.any(axis=0)
        if keep.sum() < min_agents or pres[:, keep].sum(axis=1).mean() < min_mean:
            continue
        P = tracks[sl][:, keep].copy()
        pr = pres[:, keep]
        ev = [np.where(pr[t + 1] != pr[t])[0].astype(int) for t in range(CH - 1)]
        out.append((P, ev))
    return out


def stage_real():
    """Five-scene ETH/UCY corpus + leave-one-scene-out predictors and pools."""
    info = {}
    scenes = {}
    for name, fname in SCENES.items():
        path = os.path.join(DATA, f"real_{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                scenes[name] = pickle.load(f)
            print(f"[skip] scene {name}")
            continue
        tracks, present = _interp_scene(_fetch(fname))
        d = {"short": _chunk(tracks, present, T_MAX + 1),
             "long": _chunk(tracks, present, T_REAL_LONG)}
        with open(path, "wb") as f:
            pickle.dump(d, f)
        scenes[name] = d
        print(f"[done] scene {name}: {len(d['short'])} short + {len(d['long'])} long chunks")
    for name in SCENES:
        info[name] = {"n_short_chunks": len(scenes[name]["short"]),
                      "n_long_chunks": len(scenes[name]["long"])}
    # leave-one-scene-out predictor + calibration pool
    for held in SCENES:
        mpath = os.path.join(MODELS, f"gru_real_{held}.pt")
        cpath = os.path.join(DATA, f"cal_real_{held}.npy")
        if os.path.exists(mpath) and os.path.exists(cpath):
            print(f"[skip] fold {held}")
            info[held].update(json.load(open(cpath + ".json")))
            continue
        tr = [ch for k in SCENES if k != held for ch in scenes[k]["short"]]
        print(f"[train] fold {held}: GRU on {len(tr)} chunks from 4 scenes ...")
        # Python's built-in hash is process-randomized.  CRC32 makes future
        # from-scratch LOSO training reproducible across interpreter launches.
        fold_seed = zlib.crc32(f"loso|{held}".encode())
        ginfo = train(tr, mpath, epochs=6, seed=fold_seed)
        pred = Predictor(mpath)
        scores = []
        for P, ev in tr:
            scores.extend(window_scores_from_rollout(P, ev, pred))
        np.save(cpath, np.array(scores))
        rec = {"gru": ginfo, "training_seed": fold_seed,
               "n_cal_scores": len(scores),
               "cal_score_q90": float(np.quantile(scores, 0.9))}
        with open(cpath + ".json", "w") as f:
            json.dump(rec, f)
        info[held].update(rec)
        print(f"[done] fold {held}: {len(scores)} calibration scores")
    return info


def process_eth():
    """Returns (cal_scores list, replay pool list) or raises on any failure."""
    raw = os.path.join(DATA, "biwi_eth.txt")
    if not os.path.exists(raw):
        req = urllib.request.Request(ETH_URL, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=25) as r:
            blob = r.read()
        if len(blob) < 10000:
            raise RuntimeError("eth fetch too small")
        with open(raw, "wb") as f:
            f.write(blob)
    A = np.loadtxt(raw)  # frame, ped, x, y
    tsec = A[:, 0] / 25.0
    # rotate so dominant walking direction = x axis; center the scene
    xy = A[:, 2:4] - A[:, 2:4].mean(axis=0)
    disp = []
    for pid in np.unique(A[:, 1]):
        m = A[:, 1] == pid
        if m.sum() >= 2:
            disp.append(xy[m][-1] - xy[m][0])
    disp = np.array(disp)
    cov = disp.T @ disp
    w, V = np.linalg.eigh(cov)
    R = V[:, ::-1].T  # principal direction -> x
    xy = xy @ R.T
    t0, t1 = tsec.min(), tsec.max()
    grid = np.arange(t0, t1, DT)
    PARK = np.array([60.0, 60.0])
    ids = np.unique(A[:, 1])
    tracks = np.full((len(grid), len(ids), 2), PARK, dtype=float)
    present = np.zeros((len(grid), len(ids)), dtype=bool)
    for j, pid in enumerate(ids):
        m = A[:, 1] == pid
        tt, pp = tsec[m], xy[m]
        o = np.argsort(tt)
        tt, pp = tt[o], pp[o]
        gm = (grid >= tt[0]) & (grid <= tt[-1])
        if gm.sum() < 2:
            continue
        tracks[gm, j, 0] = np.interp(grid[gm], tt, pp[:, 0])
        tracks[gm, j, 1] = np.interp(grid[gm], tt, pp[:, 1])
        present[gm, j] = True
    # chunk into episodes of T_MAX+1 control steps
    CH = T_MAX + 1
    cal_scores, pool = [], []
    pred = Predictor(os.path.join(MODELS, "gru_sf.pt"))
    nchunks = len(grid) // CH
    for c in range(nchunks):
        sl = slice(c * CH, (c + 1) * CH)
        pres = present[sl]
        keep = pres.any(axis=0)
        if keep.sum() < 3 or pres[:, keep].sum(axis=1).mean() < 2.0:
            continue
        P = tracks[sl][:, keep].copy()
        pr = pres[:, keep]
        ev = []
        for t in range(CH - 1):
            ch = np.where(pr[t + 1] != pr[t])[0]  # entry or exit = teleport
            ev.append(ch.astype(int))
        if c % 2 == 0:
            cal_scores.extend(window_scores_from_rollout(P, ev, pred))
        else:
            pool.append((P, ev))
    if len(pool) < 8 or len(cal_scores) < 50:
        raise RuntimeError(f"eth too sparse: pool={len(pool)} cal={len(cal_scores)}")
    return cal_scores, pool


def stage_eth():
    cpath = os.path.join(DATA, "cal_eth.npy")
    rpath = os.path.join(DATA, "replay_eth.pkl")
    if os.path.exists(cpath) and os.path.exists(rpath):
        print("[skip] eth already processed")
        return {"status": "ok(cached)"}
    try:
        cal, pool = process_eth()
        np.save(cpath, np.array(cal))
        with open(rpath, "wb") as f:
            pickle.dump(pool, f)
        print(f"[done] eth: {len(cal)} cal scores, {len(pool)} replay chunks")
        return {"status": "ok", "n_cal_scores": len(cal), "n_replay_chunks": len(pool),
                "cal_score_q90": float(np.quantile(cal, 0.9))}
    except Exception as e:  # AUTOMATIC FALLBACK
        print(f"[fallback] eth unavailable ({e}); using recorded sim rollouts only")
        return {"status": f"fallback: {e}"}


if __name__ == "__main__":
    meta_path = os.path.join(DATA, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update(stage_sims())
    meta["long"] = stage_long()
    meta["eth"] = stage_eth()
    try:
        meta["real"] = stage_real()
    except Exception as e:  # AUTOMATIC FALLBACK: the 5-scene study is optional
        print(f"[fallback] real ETH/UCY corpus unavailable ({e})")
        meta["real"] = {"status": f"fallback: {e}"}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print("exp0 complete")
