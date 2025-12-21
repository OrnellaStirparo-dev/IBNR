from pathlib import Path

def export_to_csv(df, filename: str, folder: str = "data/processed"):
    """Export DataFrame to CSV."""
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    fullpath = path / filename
    df.to_csv(fullpath, index=False)
    print(f"💾 Saved processed data to {fullpath}")
