"""LightGBM gradient boosting on triangle features (current standard for tabular data)."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from src.ml.features import enrich_triangle_features, matrix_xy, predict_ultimate_per_origin


def run_lightgbm(
    triangle_df: pd.DataFrame,
    target: str = "cumulative_claims",
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[LGBMRegressor, pd.Series]:
    enr = enrich_triangle_features(triangle_df)
    X, y = matrix_xy(enr)
    if len(X) < 8:
        raise ValueError("Not enough triangle cells to train LightGBM (need at least 8 observed points).")

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X, y)
    ult = predict_ultimate_per_origin(enrich_triangle_features(triangle_df), model.predict)
    return model, ult
