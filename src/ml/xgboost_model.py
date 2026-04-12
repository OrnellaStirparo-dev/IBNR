"""XGBoost 2.x gradient boosting."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from src.ml.features import enrich_triangle_features, matrix_xy, predict_ultimate_per_origin


def run_xgboost(
    triangle_df: pd.DataFrame,
    target: str = "cumulative_claims",
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    random_state: int = 42,
) -> tuple[XGBRegressor, pd.Series]:
    enr = enrich_triangle_features(triangle_df)
    X, y = matrix_xy(enr)
    if len(X) < 8:
        raise ValueError("Not enough triangle cells to train XGBoost (need at least 8 observed points).")

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,  # avoid OpenMP conflict with PyTorch on macOS
    )
    model.fit(X, y)
    ult = predict_ultimate_per_origin(enrich_triangle_features(triangle_df), model.predict)
    return model, ult
