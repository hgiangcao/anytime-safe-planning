"""Agent-motion predictor: constant velocity + small GRU residual head.

Prediction of agent position j control-steps ahead (j = 1..H):
    xhat(t+j) = x(t) + v(t) * j * DT + c(t) * (j/H)^2
where v(t) is the finite-difference velocity and c(t) is a learned 2-D correction
predicted by a GRU over the last L=5 velocity vectors, trained to regress the
H-step CV residual: c ~ x(t+H) - (x(t) + v(t) H DT).

The same (frozen) predictor is shared by ALL methods within a regime; each
closed-loop regime gets its own predictor trained on that regime's agent-only
calibration rollouts.  Training takes ~1 minute on CPU.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from sims import DT

H = 8   # prediction/planning horizon (control steps) = 2.0 s
L = 5   # history length (velocity vectors)
VSCALE = 1.5


class ResidualGRU(nn.Module):
    def __init__(self, hidden=24):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, vhist):  # vhist: [B, L, 2]
        out, _ = self.gru(vhist / VSCALE)
        return self.head(out[:, -1])  # [B, 2] correction (meters)


class Predictor:
    """cv_only=True gives the plain constant-velocity predictor."""

    def __init__(self, model_path=None, cv_only=False):
        self.cv_only = cv_only or model_path is None
        self.model_id = "constant_velocity" if self.cv_only else os.path.basename(model_path)
        if not self.cv_only:
            self.net = ResidualGRU()
            self.net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.net.eval()

    def predict(self, pos, vhist):
        """pos: [n,2] current positions; vhist: [n,L,2] recent velocities.
        Returns predicted positions [H, n, 2] for j = 1..H."""
        n = len(pos)
        v = vhist[:, -1, :]
        j = np.arange(1, H + 1, dtype=float)[:, None, None]
        pred = pos[None] + v[None] * j * DT
        if not self.cv_only:
            with torch.no_grad():
                c = self.net(torch.as_tensor(vhist, dtype=torch.float32)).numpy()
            pred = pred + c[None] * (j / H) ** 2
        return pred


def build_dataset(rollouts):
    """rollouts: list of position arrays [T+1, n, 2] (+ respawn events list) from
    record_rollout.  Returns (vhist [N,L,2], target [N,2]) with respawned agents'
    contaminated windows removed."""
    X, Y = [], []
    for P, ev in rollouts:
        T1, n, _ = P.shape
        V = np.diff(P, axis=0) / DT  # [T, n, 2]
        bad = np.zeros((T1, n), dtype=bool)  # step at which agent teleported
        for t, ids in enumerate(ev):
            if len(ids):
                bad[t + 1, ids] = True
        for t in range(L, T1 - 1 - H):
            # window uses V[t-L:t] (velocities into steps t-L+1..t) and target at t+H
            w_bad = bad[t - L + 1 : t + H + 1].any(axis=0)
            ok = ~w_bad
            if not ok.any():
                continue
            vh = V[t - L : t].transpose(1, 0, 2)[ok]  # [n_ok, L, 2]
            cv = P[t][ok] + V[t - 1][ok] * H * DT
            Y.append(P[t + H][ok] - cv)
            X.append(vh)
    return np.concatenate(X), np.concatenate(Y)


def train(rollouts, out_path, epochs=4, seed=0):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    X, Y = build_dataset(rollouts)
    net = ResidualGRU()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    n = len(Xt)
    idx = torch.randperm(n)
    losses = []
    for ep in range(epochs):
        perm = idx[torch.randperm(n)]
        tot = 0.0
        for i in range(0, n, 512):
            b = perm[i : i + 512]
            opt.zero_grad()
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        losses.append(tot / n)
    # report CV vs CV+GRU H-step RMSE on the training distribution
    with torch.no_grad():
        pred = net(Xt).numpy()
    rmse_cv = float(np.sqrt((Y**2).sum(1).mean()))
    rmse_gru = float(np.sqrt(((Y - pred) ** 2).sum(1).mean()))
    torch.save(net.state_dict(), out_path)
    return {"n_samples": int(n), "epoch_losses": losses, "rmse_cv": rmse_cv, "rmse_gru": rmse_gru}
