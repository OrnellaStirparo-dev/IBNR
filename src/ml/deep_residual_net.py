"""Deeper tabular residual MLP (LayerNorm) for cumulative loss — stronger DL baseline than the small MLP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.ml.features import enrich_triangle_features, matrix_xy, predict_ultimate_per_origin


class _ResBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(dim, dim),
            nn.Dropout(p_drop),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.net(x))


class DeepResidualTabular(nn.Module):
    def __init__(self, n_in: int, width: int = 128, n_blocks: int = 4, p_drop: float = 0.12) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(n_in, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.ModuleList([_ResBlock(width, p_drop) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)


def run_deep_residual_net(
    triangle_df: pd.DataFrame,
    epochs: int = 350,
    lr: float = 8e-4,
    random_state: int = 42,
) -> tuple[DeepResidualTabular, StandardScaler, StandardScaler, pd.Series]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    enr = enrich_triangle_features(triangle_df)
    X, y = matrix_xy(enr)
    if len(X) < 10:
        raise ValueError("Not enough triangle cells for the deep residual net (need at least 10).")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xn = x_scaler.fit_transform(X)
    yn = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(yn, dtype=torch.float32)

    model = DeepResidualTabular(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
