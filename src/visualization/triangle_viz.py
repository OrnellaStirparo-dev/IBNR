import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_triangle(df: pd.DataFrame, value_col: str = "cumulative_claims"):
    """
    Visualize the claims triangle as a heatmap.
    """
    pivot = df.pivot(index="accident_year", columns="development_year", values=value_col)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Claims Triangle Heatmap")
    plt.ylabel("Accident Year")
    plt.xlabel("Development Year")
    plt.show()
