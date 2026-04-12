import pandas as pd
import numpy as np
from typing import Dict, Optional
from .metrics import calculate_metrics


def _latest_reported_per_accident_year(triangle_df: pd.DataFrame) -> pd.Series:
    """Last **observed** cumulative per accident year (ignores NaN tail cells)."""

    def _last_valid(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    out = (
        triangle_df.sort_values("development_year")
        .groupby("accident_year")["cumulative_claims"]
        .apply(_last_valid)
    )
    out.index = [int(x) for x in out.index]
    return out


def compare_models_results(results: Dict[str, pd.DataFrame], 
                          triangle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare results from multiple models.
    
    Args:
        results: Dictionary with model names as keys and DataFrames with predictions as values
        triangle_df: Original triangle DataFrame with actual values
    
    Returns:
        DataFrame with comparison metrics for each model
    """
    comparison_data = []
    
    for model_name, predictions_df in results.items():
        if predictions_df is None or len(predictions_df) == 0:
            continue
        
        # Try to align predictions with actual values
        # For traditional methods, predictions are usually ultimate claims
        # For ML methods, we need to extract predictions differently
        
        if isinstance(predictions_df, pd.Series):
            ultimate = predictions_df.copy()
        elif isinstance(predictions_df, pd.DataFrame):
            if "ultimate" in predictions_df.columns:
                ultimate = predictions_df["ultimate"]
            else:
                ultimate = predictions_df.iloc[:, -1]
        else:
            continue
        ultimate.index = [int(x) for x in ultimate.index]

        latest_reported = _latest_reported_per_accident_year(triangle_df)
        latest_reported.index = [int(x) for x in latest_reported.index]
        common_years = latest_reported.index.intersection(ultimate.index)
        if len(common_years) == 0:
            continue
        actual = latest_reported[common_years]
        predicted = ultimate[common_years]
        metrics = calculate_metrics(actual, predicted)
        metrics["model"] = model_name
        metrics["total_ultimate"] = predicted.sum()
        metrics["total_ibnr"] = (predicted - actual).sum()
        comparison_data.append(metrics)
    
    return pd.DataFrame(comparison_data)

def create_model_summary(triangle_df: pd.DataFrame, 
                        model_results: Dict[str, any]) -> pd.DataFrame:
    """
    Create a summary DataFrame comparing all models' ultimate estimates.
    
    Args:
        triangle_df: Original triangle DataFrame
        model_results: Dictionary with model names and their ultimate estimates
    
    Returns:
        DataFrame with accident year and ultimate estimates from each model
    """
    accident_years = sorted({int(x) for x in triangle_df["accident_year"].unique()})
    summary = pd.DataFrame({"accident_year": accident_years})
    
    latest_reported = _latest_reported_per_accident_year(triangle_df)
    summary["latest_reported"] = summary["accident_year"].map(latest_reported)
    
    # Add each model's ultimate estimates
    for model_name, result in model_results.items():
        if result is None:
            continue
        
        try:
            # Extract ultimate values
            if isinstance(result, pd.Series):
                ultimate = result.copy()
                ultimate.index = [int(x) for x in ultimate.index]
            elif isinstance(result, pd.DataFrame):
                if 'ultimate' in result.columns:
                    ultimate = result['ultimate']
                elif len(result.columns) > 0:
                    # Use last column or first if only one
                    ultimate = result.iloc[:, -1] if len(result.columns) > 1 else result.iloc[:, 0]
                else:
                    continue
            else:
                continue
            
            # Map to accident years
            if hasattr(ultimate, 'index'):
                # If ultimate has an index, try to map it
                if len(ultimate.index) > 0:
                    # Check if index matches accident years
                    if all(ay in ultimate.index for ay in accident_years):
                        summary[model_name] = summary['accident_year'].map(ultimate)
                    else:
                        # Try to align by position if same length
                        if len(ultimate) == len(accident_years):
                            summary[model_name] = ultimate.values
                        else:
                            # Try to extract values if it's a multi-index or complex structure
                            if hasattr(ultimate, 'values'):
                                if len(ultimate.values) == len(accident_years):
                                    summary[model_name] = ultimate.values
            else:
                # If it's a single value or array, try to align
                if hasattr(ultimate, '__len__') and len(ultimate) == len(accident_years):
                    summary[model_name] = list(ultimate) if not isinstance(ultimate, pd.Series) else ultimate.values
        except Exception as e:
            print(f"⚠️  Warning: Could not extract ultimate for {model_name}: {e}")
            continue
    
    # Calculate IBNR for each model
    for col in summary.columns:
        if col not in ['accident_year', 'latest_reported'] and not col.endswith('_IBNR'):
            if col in summary.columns:
                summary[f'{col}_IBNR'] = summary[col] - summary['latest_reported']
    
    return summary

