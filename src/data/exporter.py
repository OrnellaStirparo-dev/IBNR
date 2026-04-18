from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_to_csv(df: pd.DataFrame, filename: str, folder: str = "data/processed") -> None:
    """Write *df* to a CSV file under *folder*, creating directories as needed."""
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    fullpath = path / filename
    df.to_csv(fullpath, index=False)
    print(f"💾 Saved to {fullpath}")
