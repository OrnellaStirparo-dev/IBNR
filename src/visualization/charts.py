import matplotlib.pyplot as plt

def plot_siniestros_by_company(df, top_n=10):
    """Show top N companies by total siniestros importe."""
    top = (
        df.groupby("compania")["importe"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    top.plot(kind="bar")
    plt.title(f"Top {top_n} compañías por importe de siniestros")
    plt.xlabel("Compañía")
    plt.ylabel("Importe total")
    plt.tight_layout()
    plt.show()
