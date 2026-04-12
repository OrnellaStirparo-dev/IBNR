"""
Transformer encoder over development tokens — sequential AI model for IBNR.

Architecture
------------
Each observed cell (cumulative_claims, lag/max_dev) is embedded as a token.
A small Transformer encoder with sinusoidal positional encoding attends over the
full development path; the model is trained one-step-ahead and ultimates are
produced by autoregressive roll-forward from the last diagonal.

Uncertainty quantification
---------------------------
MC-dropout (Gal & Ghahramani, 2016): the Transformer's internal dropout layers
remain active during ``mc_samples`` stochastic inference passes.  The variance
across draws gives a calibrated predictive uncertainty that can be used for
credibility-weighted ensembling.

References
----------
Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
Gabrielli, Richman & Wüthrich (2020) "Neural Network Embedding of the
  Over-Dispersed Poisson Reserving Model"
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
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MiniDevTransformer(nn.Module):
    """Small Transformer encoder with explicit output dropout for MC-dropout UQ."""

    def __init__(
        self,
        d_model: int = 48,
        nhead: int = 4,
        nlayers: int = 2,
        p_drop: float = 0.10,
    ) -> None:
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
        # Explicit output dropout so MC-dropout fires on the final projection too.
        self.out_drop = nn.Dropout(p_drop)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.pos(self.value_emb(x))
        h = self.enc(h, src_key_padding_mask=pad_mask)
        return self.head(self.out_drop(h)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _pivot(
    triangle_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int], int]:
    t = triangle_df.sort_values(["accident_year", "development_year"])
    p = t.pivot_table(
        index="accident_year",
        columns="development_year",
        values="cumulative_claims",
        aggfunc="first",
    ).sort_index()
    return p, [int(y) for y in p.index], int(p.columns.max())


def _one_step(
    model: MiniDevTransformer,
    prefix_vals: np.ndarray,
    x_cum: StandardScaler,
    y_scaler: StandardScaler,
    max_dev: int,
    max_len: int,
) -> float:
    """
    Predict the next development value given a prefix of observed cumulatives.
    Model must already be in the desired mode (eval / train) by the caller.
    """
    seq: list[list[float]] = []
    for k, ck in enumerate(prefix_vals):
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
        o = model(
            torch.tensor(Xs, dtype=torch.float32),
            pad_mask=torch.tensor(pm1, dtype=torch.bool),
        )
        z = o[0, L - 1].numpy().item()

    return float(y_scaler.inverse_transform([[z]])[0, 0])


def _roll_forward_transformer(
    model: MiniDevTransformer,
    row: np.ndarray,
    x_cum: StandardScaler,
    y_scaler: StandardScaler,
    max_dev: int,
    max_len: int,
) -> float:
    """Extrapolate a triangle row to the ultimate dev period."""
    last_j = next(
        (j for j in range(max_dev - 1, -1, -1) if np.isfinite(row[j]) and row[j] > 0),
        None,
    )
    if last_j is None:
        return float("nan")

    cur = row.copy()
    cur[~np.isfinite(cur)] = np.nan

    for j in range(last_j, max_dev - 1):
        prefix = np.nan_to_num(cur[: j + 1], nan=0.0)
        nxt = _one_step(model, prefix, x_cum, y_scaler, max_dev, max_len)
        cur[j + 1] = nxt

    return float(cur[max_dev - 1])


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def run_transformer_triangle(
    triangle_df: pd.DataFrame,
    epochs: int = 350,
    lr: float = 6e-4,
    mc_samples: int = 150,
    random_state: int = 42,
) -> tuple[MiniDevTransformer, StandardScaler, StandardScaler, int, pd.Series, pd.Series]:
    """
    Fit Transformer on development tokens and return ultimates with MC-dropout UQ.

    Parameters
    ----------
    triangle_df : long-format triangle DataFrame
    epochs      : training epochs
    lr          : AdamW learning rate
    mc_samples  : stochastic inference passes for uncertainty estimation
    random_state: reproducibility seed

    Returns
    -------
    model, x_cum, y_scaler, max_len, ult_mean (pd.Series), ult_std (pd.Series)
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    pivot, years, max_dev = _pivot(triangle_df)
    mat = pivot.reindex(columns=range(1, max_dev + 1)).values.astype(float)
    n_years = mat.shape[0]

    # ----- Scalers -----
    x_cum = StandardScaler()
    obs = triangle_df.loc[triangle_df["cumulative_claims"].notna(), "cumulative_claims"]
    x_cum.fit(obs.values.reshape(-1, 1))

    # ----- Build training sequences -----
    xs: list[list[tuple[float, float]]] = []
    ys: list[float] = []
    for i in range(n_years):
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
    pad_mask = np.ones((B, max_len), dtype=bool)
    for i, s in enumerate(xs):
        L = len(s)
        Xb[i, :L] = np.array(s, dtype=np.float32)
        pad_mask[i, :L] = False

    Xt = torch.tensor(Xb, dtype=torch.float32)
    pm = torch.tensor(pad_mask, dtype=torch.bool)
    yt = torch.tensor(yn, dtype=torch.float32)

    # ----- Training -----
    model = MiniDevTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xt, pad_mask=pm)
        Lidx = torch.tensor([len(xs[i]) - 1 for i in range(B)], dtype=torch.long)
        pred_last = out[torch.arange(B), Lidx]
        loss = loss_fn(pred_last, yt)
        loss.backward()
        opt.step()

    # ----- MC-dropout inference -----
    mc_draws = np.zeros((n_years, mc_samples))
    model.train()  # dropout remains active during MC passes
    for s in range(mc_samples):
        for i in range(n_years):
            mc_draws[i, s] = _roll_forward_transformer(
                model, mat[i], x_cum, y_scaler, max_dev, max_len
            )

    model.eval()

    ult_mean = pd.Series(np.nanmean(mc_draws, axis=1), index=years, name="Transformer")
    ult_std = pd.Series(np.nanstd(mc_draws, axis=1), index=years, name="transformer_std")

    return model, x_cum, y_scaler, max_len, ult_mean, ult_std
