"""
Combine traditional triangle ultimates with deep / boosting predictions (convex blends and simple ensembles).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def convex_blend(u_traditional: pd.Series, u_ml: pd.Series, weight_traditional: float) -> pd.Series:
    """
    U = w * U_trad + (1 - w) * U_ml, aligned on accident year index.
    """
    w = float(np.clip(weight_traditional, 0.0, 1.0))
    years = sorted(
        set(u_traditional.dropna().index.astype(int)).union(set(u_ml.dropna().index.astype(int)))
    )
    idx = pd.Index(years, name=None)
    a = u_traditional.reindex(years).astype(float)
    b = u_ml.reindex(years).astype(float)
    out = w * a + (1.0 - w) * b
    return out


def multi_average(*series: pd.Series) -> pd.Series:
    """Elementwise mean of aligned series (ignores NaN per position)."""
    if not series:
        raise ValueError("multi_average requires at least one series")
    years = set()
    for s in series:
        years |= set(s.dropna().index.astype(int))
    years = sorted(years)
    mat = np.column_stack([s.reindex(years).astype(float).values for s in series])
    return pd.Series(np.nanmean(mat, axis=1), index=years)


def run_hybrid_cl_deepresnet(
    u_chain_ladder: pd.Series, u_deep_resnet: pd.Series, weight_cl: float = 0.5
) -> pd.Series:
    return convex_blend(u_chain_ladder, u_deep_resnet, weight_cl)


def run_hybrid_cl_gru(u_chain_ladder: pd.Series, u_gru: pd.Series, weight_cl: float = 0.5) -> pd.Series:
    return convex_blend(u_chain_ladder, u_gru, weight_cl)


def run_hybrid_cl_transformer(
    u_chain_ladder: pd.Series, u_transformer: pd.Series, weight_cl: float = 0.5
) -> pd.Series:
    return convex_blend(u_chain_ladder, u_transformer, weight_cl)


def run_hybrid_cl_xgboost(u_chain_ladder: pd.Series, u_xgb: pd.Series, weight_cl: float = 0.5) -> pd.Series:
    return convex_blend(u_chain_ladder, u_xgb, weight_cl)


def run_hybrid_cl_bf_dl(
    u_chain_ladder: pd.Series,
    u_bornhuetter: pd.Series,
    u_deep_resnet: pd.Series,
    w_cl: float = 0.35,
    w_bf: float = 0.25,
) -> pd.Series:
    """Three-way convex combination (weights renormalized to sum to 1)."""
    w_dl = 1.0 - w_cl - w_bf
    if w_dl < 0:
        raise ValueError("weights must sum to at most 1")
    years = sorted(
        set(u_chain_ladder.dropna().index.astype(int))
        | set(u_bornhuetter.dropna().index.astype(int))
        | set(u_deep_resnet.dropna().index.astype(int))
    )
    a = u_chain_ladder.reindex(years).astype(float)
    b = u_bornhuetter.reindex(years).astype(float)
    c = u_deep_resnet.reindex(years).astype(float)
    return w_cl * a + w_bf * b + w_dl * c
