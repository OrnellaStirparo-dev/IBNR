"""
GRU over the cumulative development path (sequence of paid-to-date by lag).
Trained as one-step-ahead prediction; ultimates at the horizon via roll-forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class _GRUStep(nn.Module):
    def __init__(self, hidden: int = 48, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(h[-1]).squeeze(-1)


def _pivot_cumulative(triangle_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], int]:
    t = triangle_df.sort_values(["accident_year", "development_year"])
    p = t.pivot(index="accident_year", columns="development_year", values="cumulative_claims")
    p = p.sort_index()
    years = [int(y) for y in p.index]
    max_dev = int(p.columns.max())
    return p, years, max_dev


def run_gru_triangle(
    triangle_df: pd.DataFrame,
    epochs: int = 400,
    lr: float = 1e-3,
    random_state: int = 42,
) -> tuple[_GRUStep, StandardScaler, StandardScaler, pd.Series]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    pivot, years, max_dev = _pivot_cumulative(triangle_df)
    mat = pivot.reindex(columns=range(1, max_dev + 1)).values.astype(float)

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for i in range(mat.shape[0]):
        row = mat[i]
        for j in range(1, max_dev):
            if np.isfinite(row[j]) and np.isfinite(row[j - 1]):
                prefix = row[:j].copy()
                prefix[~np.isfinite(prefix)] = 0.0
                xs.append(prefix)
                ys.append(float(row[j]))

    if len(xs) < 8:
        raise ValueError("Not enough sequential observations for GRU training (need at least 8 one-step pairs).")

    y_scaler = StandardScaler()
    y_fit = y_scaler.fit_transform(np.array(ys).reshape(-1, 1)).ravel()

    max_len = max(len(p) for p in xs)
    X_arr = np.zeros((len(xs), max_len, 1), dtype=np.float32)
    lengths = []
    x_scaler = StandardScaler()
    flat = np.concatenate([p[p > 0] for p in xs if len(p)])
    if len(flat) == 0:
        flat = np.array([0.0])
    x_scaler.fit(flat.reshape(-1, 1))

    for i, p in enumerate(xs):
        pl = len(p)
        lengths.append(pl)
        scaled = x_scaler.transform(p.reshape(-1, 1)).astype(np.float32)
        X_arr[i, :pl, 0] = scaled.flatten()

    Xt = torch.tensor(X_arr, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.float32)
    lens = torch.tensor(lengths, dtype=torch.long)

    model = _GRUStep()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt, lens)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    model.eval()

    def _roll_forward(row: np.ndarray) -> float:
        """Extrapolate to column max_dev using one-step GRU from last observed."""
        last_j = None
        for j in range(max_dev - 1, -1, -1):
            if np.isfinite(row[j]) and row[j] > 0:
                last_j = j
                break
        if last_j is None:
            return float("nan")
        cur = row.copy()
        cur[~np.isfinite(cur)] = 0.0
        for j in range(last_j, max_dev - 1):
            prefix = cur[: j + 1]
            pl = len(prefix)
            pad = np.zeros((max_len, 1), dtype=np.float32)
            pad[:pl, 0] = x_scaler.transform(prefix.reshape(-1, 1)).flatten()
            with torch.no_grad():
                ln = torch.tensor([pl], dtype=torch.long)
                nxt = model(torch.tensor(pad[None, ...], dtype=torch.float32), ln).numpy().item()
            nxt = float(y_scaler.inverse_transform([[nxt]])[0, 0])
            if j + 1 < max_dev:
                cur[j + 1] = nxt
        return float(cur[max_dev - 1])

    out = []
    for i in range(mat.shape[0]):
        out.append(_roll_forward(mat[i]))
    ult = pd.Series(out, index=years, name="gru_ultimate").sort_index()
    return model, x_scaler, y_scaler, ult
