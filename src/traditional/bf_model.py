"""Bornhuetter–Ferguson using explicit BF formula (see ``chain_ladder_manual``)."""

from __future__ import annotations

import pandas as pd

from src.traditional.chain_ladder_manual import bornhuetter_ferguson_ultimate, cumulative_pivot_from_long


def run_bornhuetter_ferguson(
    triangle_df: pd.DataFrame, apriori: pd.Series | None = None
) -> pd.Series:
    """
    Return Bornhuetter–Ferguson ultimates per accident year.

    Parameters
    ----------
    apriori : accident-year-indexed expected ultimates (prior).
              If None, uses the volume-weighted mean chain-ladder ultimate.
    """
    pivot = cumulative_pivot_from_long(triangle_df)
    return bornhuetter_ferguson_ultimate(pivot, apriori=apriori)
