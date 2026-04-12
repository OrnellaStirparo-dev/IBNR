"""Small PyTorch MLP; targets are standardized for stable training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.ml.features import enrich_triangle_features, matrix_xy, predict_ultimate_per_origin


class _MLP(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_pytorch_mlp(
    triangle_df: pd.DataFrame,
    target: str = "cumulative_claims",
    epochs: int = 200,
    lr: float = 1e-3,
    random_state: int = 42,
) -> tuple[_MLP, StandardScaler, StandardScaler, pd.Series]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    enr = enrich_triangle_features(triangle_df)
    X, y = matrix_xy(enr)
    if len(X) < 8:
        raise ValueError("Not enough triangle cells to train the MLP (need at least 8 observed points).")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xn = x_scaler.fit_transform(X)
    yn = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(yn, dtype=torch.float32)

    model = _MLP(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    model.eval()

    def _predict(df: pd.DataFrame) -> np.ndarray:
        Xp = x_scaler.transform(df)
        with torch.no_grad():
            out = model(torch.tensor(Xp, dtype=torch.float32)).numpy()
        return y_scaler.inverse_transform(out.reshape(-1, 1)).ravel()

    ult = predict_ultimate_per_origin(enrich_triangle_features(triangle_df), _predict)
    return model, x_scaler, y_scaler, ult
