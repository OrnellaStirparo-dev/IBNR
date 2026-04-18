"""CatBoost gradient boosting on triangle features."""

from __future__ import annotations

import pandas as pd
from catboost import CatBoostRegressor

from src.ml.features import enrich_triangle_features, matrix_xy, predict_ultimate_per_origin


def run_catboost(
    triangle_df: pd.DataFrame,
    iterations: int = 400,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[CatBoostRegressor, pd.Series]:
    """
    Fit CatBoost on enriched triangle features and return (model, ultimates).

    Hyperparameters
    ---------------
    iterations=400, lr=0.05, depth=6
        Matches the XGBoost / LightGBM budget for a fair comparison.
    """
    enr = enrich_triangle_features(triangle_df)
    X, y = matrix_xy(enr)
    if len(X) < 8:
        raise ValueError("Need ≥ 8 observed triangle cells to train CatBoost.")

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=6,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
    )
    model.fit(X, y)
    ult = predict_ultimate_per_origin(enrich_triangle_features(triangle_df), model.predict)
    return model, ult
