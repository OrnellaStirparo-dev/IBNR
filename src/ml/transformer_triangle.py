"""
Compact Transformer encoder over development tokens (cumulative + normalized lag).
One-step-ahead training; ultimates via autoregressive roll-forward from last diagonal.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MiniDevTransformer(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, p_drop: float = 0.1) -> None:
        super().__init__()
        self.value_emb = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=p_drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pos = _PositionalEncoding(d_model, max_len=32)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.value_emb(x)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=pad_mask)
        return self.head(h).squeeze(-1)


def _pivot(triangle_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], int]:
    t = triangle_df.sort_values(["accident_year", "development_year"])
    p = t.pivot_table(index="accident_year", columns="development_year", values="cumulative_claims", aggfunc="first")
    p = p.sort_index()
    years = [int(y) for y in p.index]
    max_dev = int(p.columns.max())
    return p, years, max_dev


def run_transformer_triangle(
    triangle_df: pd.DataFrame,
    epochs: int = 350,
    lr: float = 6e-4,
    random_state: int = 42,
) -> tuple[MiniDevTransformer, StandardScaler, StandardScaler, int, pd.Series]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    pivot, years, max_dev = _pivot(triangle_df)
    mat = pivot.reindex(columns=range(1, max_dev + 1)).values.astype(float)

    x_cum = StandardScaler()
    obs = triangle_df.loc[triangle_df["cumulative_claims"].notna(), "cumulative_claims"].values.reshape(-1, 1)
    x_cum.fit(obs)

    xs: list[list[tuple[float, float]]] = []
    ys: list[float] = []
    for i in range(mat.shape[0]):
        row = mat[i]
        for j in range(2, max_dev + 1):
            if not np.isfinite(row[j - 1]):
                continue
            seq: list[tuple[float, float]] = []
            for k in range(1, j):
                if not np.isfinite(row[k - 1]) or row[k - 1] <= 0:
                    continue
                csn = float(x_cum.transform([[float(row[k - 1])]])[0, 0])
                seq.append((csn, k / max_dev))
            if not seq:
                continue
            xs.append(seq)
            ys.append(float(row[j - 1]))

    if len(xs) < 8:
        raise ValueError("Not enough sequence samples for the development Transformer.")

    y_scaler = StandardScaler()
    yn = y_scaler.fit_transform(np.array(ys).reshape(-1, 1)).ravel()

    max_len = max(len(s) for s in xs)
    B = len(xs)
    Xb = np.zeros((B, max_len, 2), dtype=np.float32)
    pad = np.ones((B, max_len), dtype=bool)
    for i, s in enumerate(xs):
        L = len(s)
        Xb[i, :L] = np.array(s, dtype=np.float32)
        pad[i, :L] = False

    Xt = torch.tensor(Xb, dtype=torch.float32)
    pm = torch.tensor(pad, dtype=torch.bool)
    yt = torch.tensor(yn, dtype=torch.float32)

    model = MiniDevTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xt, pad_mask=pm)
        Lidx = torch.tensor([len(xs[i]) - 1 for i in range(B)], dtype=torch.long, device=out.device)
        pred_last = out[torch.arange(B, device=out.device), Lidx]
        loss = loss_fn(pred_last, yt)
        loss.backward()
        opt.step()

    model.eval()

    def _one_step(prefix_vals: np.ndarray) -> float:
        """prefix_vals: cumulative values for dev 1..L (length L). Predict dev L+1."""
        seq = []
        for k in range(len(prefix_vals)):
            ck = prefix_vals[k]
            if not np.isfinite(ck) or ck <= 0:
                continue
            csn = float(x_cum.transform([[float(ck)]])[0, 0])
            seq.append([csn, (k + 1) / max_dev])
        if not seq:
            return float("nan")
        L = len(seq)
        Xs = np.zeros((1, max_len, 2), dtype=np.float32)
        pm1 = np.ones((1, max_len), dtype=bool)
        Xs[0, :L] = np.array(seq, dtype=np.float32)
        pm1[0, :L] = False
        with torch.no_grad():
            o = model(torch.tensor(Xs, dtype=torch.float32), pad_mask=torch.tensor(pm1, dtype=torch.bool))
            z = o[0, L - 1].numpy().item()
        return float(y_scaler.inverse_transform([[z]])[0, 0])

    out_ult: list[float] = []
    for i in range(mat.shape[0]):
        row = mat[i].copy()
        row[~np.isfinite(row)] = np.nan
        last_j = None
        for j in range(max_dev - 1, -1, -1):
            if np.isfinite(row[j]) and row[j] > 0:
                last_j = j
                break
        if last_j is None:
            out_ult.append(float("nan"))
            continue
        cur = row.copy()
        for j in range(last_j, max_dev - 1):
            prefix = cur[: j + 1]
            prefix = np.nan_to_num(prefix, nan=0.0)
            nxt = _one_step(prefix)
            cur[j + 1] = nxt
        out_ult.append(float(cur[max_dev - 1]))

    ult = pd.Series(out_ult, index=years, name="transformer_ultimate").sort_index()
    return model, x_cum, y_scaler, max_len, ult
