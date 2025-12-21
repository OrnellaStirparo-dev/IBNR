import pandas as pd
import numpy as np
from typing import Dict, Optional

def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate evaluation metrics for model predictions.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        Dictionary with metrics (MAE, RMSE, MAPE, R2)
    """
    # Remove NaN values
    mask = ~(actual.isna() | predicted.isna())
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    
    mae = np.mean(np.abs(actual_clean - predicted_clean))
    rmse = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual_clean - predicted_clean) / (actual_clean + 1e-10))) * 100
    
    # R2 Score
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }

def calculate_ibnr_reserve(ultimate: pd.Series, reported: pd.Series) -> pd.Series:
    """
    Calculate IBNR reserve (Ultimate - Reported).
    
    Args:
        ultimate: Ultimate claims estimates
        reported: Reported claims to date
    
    Returns:
        IBNR reserve
    """
    return ultimate - reported

