"""
Feedforward MLP over the development triangle (Sección 3.2).

The network is fitted on the **log development factor** ``log(C_j / C_{j-1})``
rather than on the cumulative itself, and the projection to the ultimate is a
multiplicative roll-forward. Both choices matter here:

- The cumulative on this triangle spans three orders of magnitude, because
  twenty years of inflation sit inside it. A network fitted on standardised
  levels spends its capacity on scale rather than on development, and an
  autoregressive projection on levels compounds that error every period.
- Predicting a factor recovers the chain-ladder structure exactly when the data
  are multiplicative, so the MLP generalises the classical method instead of
  competing with it from scratch — which is the claim Section 3.2 makes.

The MLP still treats each cell as an independent observation: it has no
mechanism for the serial dependence along an accident year, which is the
limitation that motivates the recurrent model in Section 3.3.

Uncertainty is MC-dropout (Gal & Ghahramani, 2016): dropout stays active at
inference and each stochastic pass produces a full roll-forward, so the spread
reflects uncertainty compounded along the projection. It is epistemic
uncertainty about the fitted function only — not comparable to Mack's MSEP,
which also carries process variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.data.features import matrix_xy_logratio, rollforward_logratio


class _MLP(nn.Module):
    """
    Feedforward network over ``(development period, accident year)``, the
    standard formulation for a cell-wise neural reserving model (Wüthrich, 2018).

    A note on what this architecture can and cannot identify. The cells to be
    predicted are precisely the recent accident years at late development
    periods, and no cell of that kind has ever been observed — late development
    is only seen for old accident years, recent accident years only early. The
    accident-year effect therefore has to be extrapolated into a corner of the
    plane with no data, and on a nominal triangle the effect the network has
    learnt there is inflation. That is why the nominal fit overshoots by an order
    of magnitude while the constant-currency fit is competitive: it is a property
    of the data, not a defect of the training, and it is the argument for
    embedding an actuarial structure (as CANN does) rather than leaving the model
    free.

    Making the two effects additive in log space does not fix it and makes it
    worse: an additive offset scales every remaining development factor equally,
    whereas a settlement slowdown shifts mass between periods. The interaction is
    needed; what is missing is data to identify it.
    """

    def __init__(self, n_in: int, hidden: tuple[int, int] = (64, 32), p_drop: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden[0]),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_pytorch_mlp(
    triangle_df: pd.DataFrame,
    epochs: int = 1500,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    p_drop: float = 0.15,
    mc_samples: int = 150,
    random_state: int = 42,
) -> tuple[_MLP, StandardScaler, StandardScaler, pd.Series, pd.Series]:
    """
    Fit the MLP and return ``(model, x_scaler, y_scaler, ult_mean, ult_std)``.
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X, y = matrix_xy_logratio(triangle_df)
    if len(X) < 8:
        raise ValueError("Not enough observed cell pairs to train the MLP (need ≥ 8).")

    x_scaler = StandardScaler().fit(X.values)
    y_scaler = StandardScaler().fit(y.values.reshape(-1, 1))

    Xt = torch.tensor(x_scaler.transform(X.values), dtype=torch.float32)
    yt = torch.tensor(y_scaler.transform(y.values.reshape(-1, 1)).ravel(), dtype=torch.float32)

    model = _MLP(X.shape[1], p_drop=p_drop)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optim.step()

    def _predict(features: pd.DataFrame) -> np.ndarray:
        scaled = x_scaler.transform(features.values)
        with torch.no_grad():
            raw = model(torch.tensor(scaled, dtype=torch.float32)).numpy()
        return y_scaler.inverse_transform(raw.reshape(-1, 1)).ravel()

    model.train()  # dropout stays on for MC sampling
    draws = [rollforward_logratio(triangle_df, _predict) for _ in range(mc_samples)]
    model.eval()

    years = draws[0].index
    matrix = np.column_stack([d.values for d in draws])
    return (
        model,
        x_scaler,
        y_scaler,
        pd.Series(np.nanmean(matrix, axis=1), index=years, name="MLP"),
        pd.Series(np.nanstd(matrix, axis=1), index=years, name="mlp_std"),
    )
