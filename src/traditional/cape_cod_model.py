"""
Cape Cod normally requires premium / on-level exposure by origin.
Without that triangle, we return chain-ladder ultimates as a transparent proxy.
"""

from __future__ import annotations

import pandas as pd

from src.traditional.chain_ladder_manual import cape_cod_proxy, cumulative_pivot_from_long


def run_cape_cod(triangle_df: pd.DataFrame) -> pd.Series:
    """
    Return Cape Cod ultimates per accident year.

    Note: returns chain-ladder ultimates as a proxy — a true Cape Cod requires
    on-level earned premium / exposure by accident year which is not available
    in the triangle-only dataset.
    """
    pivot = cumulative_pivot_from_long(triangle_df)
    return cape_cod_proxy(pivot)
