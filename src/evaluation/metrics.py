"""Evaluation metrics for comparing IBNR reserve estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """
    Return MAE, RMSE, MAPE, and R² for a pair of aligned series.

    NaN positions in either series are dropped before computation.
    MAPE uses a small epsilon to guard against zero-denominator rows.
    """
    mask = ~(actual.isna() | predicted.isna())
    a = actual[mask]
    p = predicted[mask]

    if len(a) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    residuals = a - p
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mape = float(np.mean(np.abs(residuals / (a + 1e-10))) * 100)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def calculate_ibnr_reserve(ultimate: pd.Series, reported: pd.Series) -> pd.Series:
    """IBNR reserve = ultimate − latest reported (aligned on index)."""
    return ultimate - reported
