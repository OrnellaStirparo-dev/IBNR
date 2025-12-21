import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_features(df: pd.DataFrame, target: str = "cumulative_claims") -> tuple:
    """
    Prepare features for random forest training.
    
    Args:
        df: DataFrame with accident_year, development_year, and target column
        target: Name of target column
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Create features
    if "accident_year" in df.columns and "development_year" in df.columns:
        X = df[["accident_year", "development_year"]].copy()
        # Add interaction features
        X["accident_dev_interaction"] = X["accident_year"] * X["development_year"]
        X["dev_lag"] = X.groupby("accident_year")["development_year"].shift(1).fillna(0)
    else:
        # If triangle format, use all columns except target
        X = df.drop(columns=[target] if target in df.columns else [])
    
    y = df[target] if target in df.columns else df.iloc[:, -1]
    
    return X, y


def run_random_forest(df: pd.DataFrame, target: str = "cumulative_claims",
                      n_estimators: int = 200, max_depth: int = 10, 
                      min_samples_split: int = 5, random_state: int = 42):
    """
    Train a Random Forest regression model for IBNR estimation.
    
    Args:
        df: DataFrame with features and target
        target: Name of target column
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples required to split a node
        random_state: Random seed for reproducibility
    
    Returns:
        Trained Random Forest model
    """
    # Prepare data
    X, y = prepare_features(df, target)
    
    # Remove rows with NaN target values (upper triangle)
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Random Forest - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
    print(f"Feature importance: {dict(zip(X.columns, model.feature_importances_))}")
    
    return model
