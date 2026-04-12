"""Chain ladder — volume-weighted link ratios (manual implementation, industry standard)."""

from __future__ import annotations

import pandas as pd

from src.traditional.chain_ladder_manual import cumulative_pivot_from_long, ultimate_from_pivot


def run_chainladder(triangle_df: pd.DataFrame) -> pd.Series:
    pivot = cumulative_pivot_from_long(triangle_df)
    return ultimate_from_pivot(pivot)
