import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def prepare_features(df: pd.DataFrame, target: str = "cumulative_claims") -> tuple:
    """
    Prepare features for neural network training.
    
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


def run_neural_network(df: pd.DataFrame, target: str = "cumulative_claims", 
                       epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2):
    """
    Train a neural network regression model for IBNR estimation.
    
    Args:
        df: DataFrame with features and target
        target: Name of target column
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        Trained model and training history
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
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build neural network
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0
    )
    
    # Evaluate
    train_pred = model.predict(X_train_scaled, verbose=0)
    test_pred = model.predict(X_test_scaled, verbose=0)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Neural Network - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
    
    return model, history, scaler
