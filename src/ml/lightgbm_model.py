"""LightGBM gradient boosting on triangle features."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from src.ml.features import enrich_triangle_features, matrix_xy, predict_ultimate_per_origin


def run_lightgbm(
    triangle_df: pd.DataFrame,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[LGBMRegressor, pd.Series]:
    """
    Fit LightGBM on enriched triangle features and return (model, ultimates).

    Hyperparameters
    ---------------
    num_leaves=31
        Default leaf budget; keeps the tree shallow relative to the triangle size.
    n_jobs=1
        Prevents OpenMP conflicts with PyTorch on macOS.
    """
    enr = enrich_triangle_features(triangle_df)
    X, y = matrix_xy(enr)
    if len(X) < 8:
        raise ValueError("Need ≥ 8 observed triangle cells to train LightGBM.")

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y)
    ult = predict_ultimate_per_origin(enrich_triangle_features(triangle_df), model.predict)
    return model, ult
