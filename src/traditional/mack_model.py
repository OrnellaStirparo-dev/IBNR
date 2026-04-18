"""
Mack chain ladder: mean ultimates coincide with ordinary chain ladder.
(Mack adds process variance; we expose mean ultimates here for comparability.)
"""

from __future__ import annotations

import pandas as pd

from src.traditional.chain_ladder_manual import cumulative_pivot_from_long, ultimate_from_pivot


def run_mack_chainladder(triangle_df: pd.DataFrame) -> pd.Series:
    """
    Return Mack chain-ladder ultimates per accident year.

    Mean ultimates coincide with ordinary chain ladder; the Mack framework
    adds process-variance estimates (not exposed here — use for point comparison).
    """
    pivot = cumulative_pivot_from_long(triangle_df)
    return ultimate_from_pivot(pivot)
