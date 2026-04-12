"""
GRU over the cumulative development path — sequential AI model for IBNR.

Architecture
------------
A two-layer GRU reads the cumulative paid sequence for each accident year and
predicts the next development step.  Trained one-step-ahead on all observed
triangle rows; ultimates are produced by autoregressive roll-forward from the
last observed diagonal cell.

Uncertainty quantification
---------------------------
MC-dropout (Gal & Ghahramani, 2016): dropout remains *active* during inference
over ``mc_samples`` stochastic forward passes.  The spread of resulting
ultimates gives a predictive distribution without retraining.  This is more
principled than a point estimate and allows credibility-weighted ensembling.

References
----------
Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
Kuo (2019) "DeepTriangle: A Deep Learning Approach to Loss Reserving"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class GRUStep(nn.Module):
    """Two-layer GRU with explicit output dropout for MC-dropout UQ."""

    def __init__(self, hidden: int = 48, num_layers: int = 2, p_drop: float = 0.10) -> None:
        super().__init__()
        self.gru = nn.GRU(
            1,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0.0,
        )
        # Explicit dropout before the final projection so MC-dropout fires
        # even when num_layers == 1 (inter-layer dropout is absent then).
        self.out_drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(self.out_drop(h[-1])).squeeze(-1)


# --------------------------------------------------------------------------- #
# Internal helpers — standalone so they can be called with model in any mode  #
# --------------------------------------------------------------------------- #

def _pivot_cumulative(
    triangle_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int], int]:
    t = triangle_df.sort_values(["accident_year", "development_year"])
    p = t.pivot(index="accident_year", columns="development_year", values="cumulative_claims")
    return p.sort_index(), [int(y) for y in p.index], int(p.columns.max())


def _roll_forward(
    model: GRUStep,
    row: np.ndarray,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    max_dev: int,
    max_len: int,
) -> float:
    """
    Extrapolate a single triangle row to the ultimate development period.
    The model must already be set to the desired mode (eval / train) by the caller.
    """
    last_j = next(
        (j for j in range(max_dev - 1, -1, -1) if np.isfinite(row[j]) and row[j] > 0),
        None,
    )
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
            nxt_z = model(
                torch.tensor(pad[None, ...], dtype=torch.float32),
                torch.tensor([pl], dtype=torch.long),
            ).numpy().item()
        cur[j + 1] = float(y_scaler.inverse_transform([[nxt_z]])[0, 0])

    return float(cur[max_dev - 1])


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def run_gru_triangle(
    triangle_df: pd.DataFrame,
    epochs: int = 400,
    lr: float = 1e-3,
    mc_samples: int = 150,
    random_state: int = 42,
) -> tuple[GRUStep, StandardScaler, StandardScaler, pd.Series, pd.Series]:
    """
    Fit GRU on the development triangle and return ultimates with MC-dropout UQ.

    Parameters
    ----------
    triangle_df : long-format triangle DataFrame
    epochs      : training epochs
    lr          : AdamW learning rate
    mc_samples  : number of stochastic forward passes for uncertainty estimation
    random_state: reproducibility seed

    Returns
    -------
    model, x_scaler, y_scaler, ult_mean (pd.Series), ult_std (pd.Series)
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    pivot, years, max_dev = _pivot_cumulative(triangle_df)
    mat = pivot.reindex(columns=range(1, max_dev + 1)).values.astype(float)
    n_years = mat.shape[0]

    # ----- Build training pairs -----
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for i in range(n_years):
        row = mat[i]
        for j in range(1, max_dev):
            if np.isfinite(row[j]) and np.isfinite(row[j - 1]):
                prefix = row[:j].copy()
                prefix[~np.isfinite(prefix)] = 0.0
                xs.append(prefix)
                ys.append(float(row[j]))

    if len(xs) < 8:
        raise ValueError(
            "Not enough sequential observations for GRU training (need ≥ 8 one-step pairs)."
        )

    y_scaler = StandardScaler()
    y_fit = y_scaler.fit_transform(np.array(ys).reshape(-1, 1)).ravel()

    max_len = max(len(p) for p in xs)
    x_scaler = StandardScaler()
    flat = np.concatenate([p[p > 0] for p in xs if p.size > 0])
    x_scaler.fit((flat if len(flat) else np.array([0.0])).reshape(-1, 1))

    X_arr = np.zeros((len(xs), max_len, 1), dtype=np.float32)
    lengths = []
    for i, p in enumerate(xs):
        pl = len(p)
        lengths.append(pl)
        X_arr[i, :pl, 0] = x_scaler.transform(p.reshape(-1, 1)).astype(np.float32).flatten()

    Xt = torch.tensor(X_arr, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.float32)
    lens = torch.tensor(lengths, dtype=torch.long)

    # ----- Training -----
    model = GRUStep()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt, lens), yt)
        loss.backward()
        opt.step()

    # ----- MC-dropout inference -----
    # Keep model.train() so dropout stays active across all mc_samples passes.
    mc_draws = np.zeros((n_years, mc_samples))
    for s in range(mc_samples):
        for i in range(n_years):
            mc_draws[i, s] = _roll_forward(model, mat[i], x_scaler, y_scaler, max_dev, max_len)

    model.eval()  # restore eval mode

    ult_mean = pd.Series(np.nanmean(mc_draws, axis=1), index=years, name="GRU")
    ult_std = pd.Series(np.nanstd(mc_draws, axis=1), index=years, name="gru_std")

    return model, x_scaler, y_scaler, ult_mean, ult_std
