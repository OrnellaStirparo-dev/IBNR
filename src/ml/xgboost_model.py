"""XGBoost gradient boosting on triangle features."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from src.ml.features import enrich_triangle_features, matrix_xy, predict_ultimate_per_origin


def run_xgboost(
    triangle_df: pd.DataFrame,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    random_state: int = 42,
) -> tuple[XGBRegressor, pd.Series]:
    """
    Fit XGBoost on enriched triangle features and return (model, ultimates).

    Hyperparameters
    ---------------
    n_estimators=400, lr=0.05, max_depth=6, subsample=0.9
        Moderate depth to avoid overfitting on the small triangle; 400 trees
        with a low learning rate gives stable convergence.
    n_jobs=1
        Prevents OpenMP conflicts with PyTorch on macOS.
    """
    enr = enrich_triangle_features(triangle_df)
    X, y = matrix_xy(enr)
    if len(X) < 8:
        raise ValueError("Need ≥ 8 observed triangle cells to train XGBoost.")

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X, y)
    ult = predict_ultimate_per_origin(enrich_triangle_features(triangle_df), model.predict)
    return model, ult
