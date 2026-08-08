"""Bornhuetter–Ferguson and Cape Cod (see ``chain_ladder_manual`` for the formulas)."""

from __future__ import annotations

import pandas as pd

from src.traditional.chain_ladder_manual import (
    bornhuetter_ferguson_ultimate,
    cape_cod_ultimate,
    cumulative_pivot_from_long,
)


def run_bornhuetter_ferguson(
    triangle_df: pd.DataFrame,
    apriori: pd.Series | None = None,
    exposure: pd.Series | None = None,
) -> pd.Series:
    """
    Return Bornhuetter–Ferguson ultimates per accident year.

    Parameters
    ----------
    apriori  : accident-year-indexed expected ultimates, in the nominal currency
               of each accident year. Built from ``exposure`` when omitted.
    exposure : on-level exposure proxy per accident year (e.g. vehicles exposed
               × severity index). Supplying it is what keeps the prior in step
               with severity trend; without it the prior can only extrapolate
               the trend of the fully developed years.
    """
    pivot = cumulative_pivot_from_long(triangle_df)
    return bornhuetter_ferguson_ultimate(pivot, apriori=apriori, exposure=exposure)


def run_cape_cod(
    triangle_df: pd.DataFrame, exposure: pd.Series | None = None
) -> pd.Series:
    """Return Cape Cod (Bühlmann–Straub, 1983) ultimates per accident year."""
    pivot = cumulative_pivot_from_long(triangle_df)
    return cape_cod_ultimate(pivot, exposure=exposure)
