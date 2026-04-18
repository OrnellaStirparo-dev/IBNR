from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_siniestros_by_company(df: pd.DataFrame, top_n: int = 10) -> None:
    """Bar chart of the top *top_n* companies by total claims importe."""
    top = (
        df.groupby("compania")["importe"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    plt.figure(figsize=(10, 6))
    top.plot(kind="bar")
    plt.title(f"Top {top_n} compañías por importe de siniestros", fontsize=13)
    plt.xlabel("Compañía")
    plt.ylabel("Importe total")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
