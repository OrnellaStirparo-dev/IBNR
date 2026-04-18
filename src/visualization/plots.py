from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Bar chart of RMSE per model (lower is better)."""
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="model", y="rmse", color="steelblue")
    plt.title("Model Comparison — RMSE (lower is better)", fontsize=13)
    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
