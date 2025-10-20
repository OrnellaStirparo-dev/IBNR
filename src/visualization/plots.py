import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_comparison(results_df: pd.DataFrame):
    """
    Compare multiple models' performance visually.
    results_df: DataFrame with columns ['model', 'rmse', 'mae']
    """
    plt.figure(figsize=(8,5))
    sns.barplot(data=results_df, x="model", y="rmse", color="skyblue")
    plt.title("Model Comparison (Lower RMSE = Better)")
    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.show()
