"""
CANN — Combined Actuarial Neural Network for claims reserving.

The idea (Wüthrich & Merz, "Yes, we CANN!", ASTIN Bulletin 2019; Gabrielli,
Richman & Wüthrich 2020) is to embed a classical actuarial model inside a neural
network so that training *starts* at the actuarial answer and the network can
only learn what the actuarial model missed.

Here the actuarial base is the chain ladder in its over-dispersed Poisson GLM
form (England & Verrall, 2002), whose log-link mean for the incremental payment
in cell ``(i, j)`` is::

    log m_ij = c + a_i + b_j

fitted by Poisson maximum likelihood on the observed incremental triangle. The
combined regression function is::

    log m_ij = (c + a_i + b_j) + <beta_DNN, z(x_ij)>

with the GLM part entering as a **non-trainable offset** and the network weights
initialised at zero. At epoch 0 the CANN therefore reproduces the GLM exactly;
gradient descent only ever moves it away from that solution where the residual
structure justifies it.

Two properties matter for reserving practice. Degradation is controlled: when
the data are compatible with the multiplicative chain-ladder structure there is
no residual signal and the model stays at the GLM. And the network's output is
interpretable as an explicit multiplicative correction to the chain ladder,
cell by cell, which is what a reviewer or a regulator can actually inspect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _incremental_from_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Incremental payments from a cumulative triangle."""
    inc = pivot.copy()
    values = pivot.values.astype(float)
    out = np.full_like(values, np.nan)
    for i in range(values.shape[0]):
        prev = 0.0
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                out[i, j] = values[i, j] - prev
                prev = values[i, j]
    inc.iloc[:, :] = out
    return inc


def fit_odp_glm(
    pivot: pd.DataFrame,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[pd.Series, pd.Series, float, float]:
    """
    Fit ``log m_ij = c + a_i + b_j`` by Poisson MLE on the observed increments.

    Returns ``(alpha, beta, intercept, dispersion)``. Fitted by iterative
    proportional fitting, which is the standard route to the Poisson MLE for a
    two-way log-linear model and reproduces the chain-ladder reserve exactly
    when the triangle is complete — the equivalence England & Verrall (2002)
    established, and the reason this is the right base for a CANN.

    ``dispersion`` is the Pearson estimate ``phi``, which scales the ODP
    variance ``Var(m) = phi * m``.
    """
    inc = _incremental_from_pivot(pivot)
    Y = inc.values.astype(float)
    observed = np.isfinite(Y) & (Y > 0)
    if not observed.any():
        raise ValueError("No positive incremental cells to fit the ODP GLM on.")

    n_ay, n_dev = Y.shape
    row_tot = np.nansum(np.where(observed, Y, 0.0), axis=1)
    col_tot = np.nansum(np.where(observed, Y, 0.0), axis=0)

    a = np.where(row_tot > 0, row_tot, 1.0)
    b = np.full(n_dev, 1.0 / max(n_dev, 1))

    for _ in range(max_iter):
        a_prev, b_prev = a.copy(), b.copy()
        for i in range(n_ay):
            denom = b[observed[i]].sum()
            a[i] = row_tot[i] / denom if denom > 0 else a[i]
        for j in range(n_dev):
            denom = a[observed[:, j]].sum()
            b[j] = col_tot[j] / denom if denom > 0 else b[j]
        if np.max(np.abs(a - a_prev)) < tol and np.max(np.abs(b - b_prev)) < tol:
            break

    fitted = np.outer(a, b)
    resid = (Y[observed] - fitted[observed]) ** 2 / np.maximum(fitted[observed], 1e-9)
    dof = max(int(observed.sum()) - (n_ay + n_dev - 1), 1)
    dispersion = float(resid.sum() / dof)

    scale = float(np.mean(a)) if np.mean(a) > 0 else 1.0
    alpha = pd.Series(np.log(np.maximum(a / scale, 1e-12)), index=pivot.index.astype(int))
    beta = pd.Series(np.log(np.maximum(b, 1e-12)), index=list(pivot.columns))
    return alpha, beta, float(np.log(scale)), dispersion


def odp_glm_ultimate(pivot: pd.DataFrame) -> pd.Series:
    """Chain-ladder-equivalent ultimates from the ODP GLM fit."""
    alpha, beta, intercept, _ = fit_odp_glm(pivot)
    inc = _incremental_from_pivot(pivot)
    out = {}
    for i, ay in enumerate(pivot.index.astype(int)):
        paid = np.nansum(inc.iloc[i].values)
        future = sum(
            float(np.exp(intercept + alpha.iloc[i] + beta.iloc[j]))
            for j in range(len(beta))
            if not np.isfinite(inc.iat[i, j])
        )
        out[ay] = float(paid + future)
    return pd.Series(out, name="odp_glm_ultimate").sort_index()


class _ResidualNet(nn.Module):
    """
    Small feed-forward network producing the log-scale correction to the GLM.

    The final layer is initialised at zero so the correction starts at exactly
    zero and the combined model starts at the GLM.
    """

    def __init__(self, n_in: int, hidden: int = 16, p_drop: float = 0.10) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(p_drop),
        )
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


def _features(ay_idx: np.ndarray, dev_idx: np.ndarray, n_ay: int, n_dev: int) -> np.ndarray:
    """
    Covariates for each cell: accident year, development period and calendar
    year, each scaled to [-1, 1].

    The calendar-year axis is the one the chain ladder structurally cannot see —
    it has no term for it — so it is where a residual correction has something
    to find under inflation or a change in settlement speed.
    """
    ay = 2.0 * ay_idx / max(n_ay - 1, 1) - 1.0
    dev = 2.0 * dev_idx / max(n_dev - 1, 1) - 1.0
    cal = ay_idx + dev_idx
    cal = 2.0 * cal / max(cal.max(), 1) - 1.0
    return np.column_stack([ay, dev, cal, ay * dev]).astype(np.float32)


def run_cann(
    triangle_df: pd.DataFrame,
    epochs: int = 800,
    lr: float = 5e-3,
    weight_decay: float = 1e-3,
    p_drop: float = 0.10,
    mc_samples: int = 150,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Fit a CANN on the triangle and return ``(ultimate_mean, ultimate_std)``.

    Training minimises the over-dispersed Poisson deviance of the observed
    increments. ``weight_decay`` regularises the correction towards zero, i.e.
    towards the GLM — the shrinkage target is the actuarial model, not the
    origin, which is what keeps the hybrid from drifting on a small triangle.

    Uncertainty is MC-dropout (Gal & Ghahramani, 2016): ``mc_samples`` stochastic
    forward passes with dropout active at inference. Note this is epistemic
    uncertainty about the *correction* only; it is not comparable to Mack's
    MSEP, which also carries process variance.
    """
    from src.traditional.chain_ladder_manual import cumulative_pivot_from_long

    torch.manual_seed(seed)
    pivot = cumulative_pivot_from_long(triangle_df).sort_index()
    alpha, beta, intercept, _ = fit_odp_glm(pivot)
    inc = _incremental_from_pivot(pivot)

    n_ay, n_dev = inc.shape
    ay_idx, dev_idx, targets, offsets = [], [], [], []
    for i in range(n_ay):
        for j in range(n_dev):
            value = inc.iat[i, j]
            if np.isfinite(value) and value > 0:
                ay_idx.append(i)
                dev_idx.append(j)
                targets.append(float(value))
                offsets.append(float(intercept + alpha.iloc[i] + beta.iloc[j]))

    if not targets:
        raise ValueError("No observed increments to train on.")

    X = torch.tensor(_features(np.array(ay_idx), np.array(dev_idx), n_ay, n_dev))
    y = torch.tensor(np.array(targets, dtype=np.float32))
    off = torch.tensor(np.array(offsets, dtype=np.float32))

    model = _ResidualNet(X.shape[1], p_drop=p_drop)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        log_mu = off + model(X)
        mu = torch.exp(torch.clamp(log_mu, max=30.0))
        # Poisson deviance, the ODP quasi-likelihood loss.
        loss = 2.0 * torch.mean(mu - y + y * (torch.log(y) - torch.clamp(log_mu, max=30.0)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

    # Future cells to be predicted.
    fut_ay, fut_dev, fut_off = [], [], []
    for i in range(n_ay):
        for j in range(n_dev):
            if not np.isfinite(inc.iat[i, j]):
                fut_ay.append(i)
                fut_dev.append(j)
                fut_off.append(float(intercept + alpha.iloc[i] + beta.iloc[j]))

    paid = np.array([np.nansum(inc.iloc[i].values) for i in range(n_ay)])
    years = pivot.index.astype(int)

    if not fut_ay:
        zeros = pd.Series(0.0, index=years)
        return pd.Series(paid, index=years, name="cann_ultimate"), zeros

    Xf = torch.tensor(_features(np.array(fut_ay), np.array(fut_dev), n_ay, n_dev))
    off_f = torch.tensor(np.array(fut_off, dtype=np.float32))

    model.train()  # keep dropout active for MC sampling
    draws = np.zeros((mc_samples, n_ay))
    with torch.no_grad():
        for s in range(mc_samples):
            mu = torch.exp(torch.clamp(off_f + model(Xf), max=30.0)).numpy()
            reserve = np.zeros(n_ay)
            np.add.at(reserve, np.array(fut_ay), mu)
            draws[s] = paid + reserve

    return (
        pd.Series(draws.mean(axis=0), index=years, name="cann_ultimate"),
        pd.Series(draws.std(axis=0), index=years, name="cann_std"),
    )
