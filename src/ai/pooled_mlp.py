"""
Pooled MLP over many triangles (Sections 3.1 and 4.3, comment 5 of the reviewer).

Same parameterisation as ``pytorch_mlp``: the network predicts the log
development factor ``log(C_{i,j+1}/C_{i,j})`` and the reserve is obtained by
multiplicative roll-forward, so the only thing that changes between the
single-triangle MLP and this one is **how much information the network sees**.
That isolates the variable the thesis' hypothesis is about.

What links the triangles is stated explicitly in the features:

- ``development_year`` and ``accident_year_idx`` (scaled to [0, 1]), as in the
  single-triangle model;
- a one-hot **company** indicator, so each insurer keeps its own level of
  development speed (Kuo, 2019, uses an embedding for the same purpose);
- a one-hot **line** indicator when several lines are pooled, so the network
  is not forced to give workers' compensation and private auto the same
  pattern — it is allowed to share only what the data say is shared.

Training weights: within each triangle cells are weighted by their volume
(the chain-ladder weighting of equation 1), normalised so that every triangle
carries the same total weight; otherwise the three or four largest insurers
would define the pattern for everyone.

Cells with a negative increment (``C_{i,j+1} < C_{i,j}``, a net recovery over
the period) are **excluded from this network's training set**. The target is
``log(C_{i,j+1}/C_{i,j})`` and the roll-forward clamps the predicted log factor
at zero, so the model has no way to represent a decrease in the first place.
The recoveries themselves are kept everywhere else: they remain in the
triangle, in the observed cumulative each projection starts from, and in the
traditional estimators, whose volume-weighted factors absorb them (see
``src.data.schedule_p.usable``, which deliberately retains them as part of real
auto experience). Only the network's supervision drops those cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _Net(nn.Module):
    def __init__(self, n_in: int, hidden=(64, 32), p_drop: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_pooled_mlp(
    triangles: dict[tuple[str, str], pd.DataFrame],
    use_line: bool,
    epochs: int = 600,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    mc_samples: int = 50,
    seed: int = 42,
) -> dict[tuple[str, str], pd.Series]:
    """
    Fit one network on every upper triangle in *triangles* (keys = (line,
    company), values = pivots with NaN below the diagonal) and return the
    projected ultimate per accident year for each of them (MC-dropout mean).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    keys = list(triangles)
    comp_ix = {k: i for i, k in enumerate(keys)}
    lines = sorted({k[0] for k in keys})
    line_ix = {l: i for i, l in enumerate(lines)}
    n_comp, n_line = len(keys), len(lines) if use_line else 0
    any_piv = next(iter(triangles.values()))
    n_ay, n_dev = any_piv.shape

    def feat(k, i, j):  # i: accident-year index, j: target development period (2..n_dev)
        x = np.zeros(2 + n_comp + n_line, dtype=np.float32)
        x[0] = j / n_dev
        x[1] = i / (n_ay - 1)
        x[2 + comp_ix[k]] = 1.0
        if use_line:
            x[2 + n_comp + line_ix[k[0]]] = 1.0
        return x

    X, y, w = [], [], []
    for k, piv in triangles.items():
        C = piv.values.astype(float)
        rows, ws = [], []
        for i in range(n_ay):
            for j in range(1, n_dev):
                a, b = C[i, j - 1], C[i, j]
                if np.isfinite(a) and np.isfinite(b) and a > 0 and b >= a:
                    X.append(feat(k, i, j + 1))
                    y.append(np.log(b / a))
                    ws.append(a)
        ws = np.array(ws)
        w.extend(ws / ws.sum())

    X = torch.tensor(np.array(X))
    y = torch.tensor(np.array(y, dtype=np.float32))
    w = torch.tensor(np.array(w, dtype=np.float32))
    y_mu, y_sd = float(y.mean()), float(y.std())
    yt = (y - y_mu) / y_sd

    model = _Net(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.sum(w * (model(X) - yt) ** 2) / w.sum()
        loss.backward()
        opt.step()

    # Roll-forward, all MC passes vectorised per triangle.
    out = {}
    model.train()  # MC dropout
    with torch.no_grad():
        for k, piv in triangles.items():
            C = piv.values.astype(float)
            draws = np.zeros((mc_samples, n_ay))
            for i in range(n_ay):
                j0 = max(j for j in range(n_dev) if np.isfinite(C[i, j]))
                if j0 == n_dev - 1:
                    draws[:, i] = C[i, j0]
                    continue
                fx = torch.tensor(np.array([feat(k, i, j + 2) for j in range(j0, n_dev - 1)]))
                fx = fx.unsqueeze(0).repeat(mc_samples, 1, 1)
                lf = model(fx.reshape(-1, fx.shape[-1])).reshape(mc_samples, -1).numpy()
                lf = np.maximum(lf * y_sd + y_mu, 0.0)
                draws[:, i] = C[i, j0] * np.exp(lf.sum(axis=1))
            out[k] = pd.Series(draws.mean(axis=0), index=piv.index, name="pooled_mlp")
    return out
