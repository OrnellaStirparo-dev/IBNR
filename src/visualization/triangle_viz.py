from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_triangle(
    df: pd.DataFrame,
    value_col: str = "cumulative_claims",
    save_path: str | None = None,
) -> None:
    """Render the cumulative claims triangle as an annotated heatmap."""
    pivot = df.pivot(index="accident_year", columns="development_year", values=value_col)
    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", linewidths=0.4)
    plt.title("Cumulative Claims Triangle", fontsize=14, fontweight="bold")
    plt.ylabel("Accident Year")
    plt.xlabel("Development Year")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
