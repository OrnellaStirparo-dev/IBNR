"""
Hybrid / ensemble models: convex blends and credibility-weighted AI ensemble.

Two types of combination
------------------------
1. **Convex blend** — simple weighted average of two ultimates (backward compatible).
2. **Credibility hybrid** — uncertainty-weighted ensemble of AI models with an optional
   traditional anchor.

Credibility weighting rationale
---------------------------------
Bühlmann credibility theory assigns higher weight to estimators with lower variance.
We operationalise this using MC-dropout predictive std: models with tighter predictive
distributions (lower std) receive proportionally higher weight.  A traditional model
(e.g., Chain Ladder) can be given a fixed anchor weight, with the remainder distributed
among AI models by inverse-std credibility.

The result is a data-driven replacement for the fixed 50/50 blends.

References
----------
Bühlmann & Gisler (2005) "A Course in Credibility Theory and Its Applications"
Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Low-level blend primitives                                                   #
# --------------------------------------------------------------------------- #

def convex_blend(
    u_traditional: pd.Series, u_model: pd.Series, weight_traditional: float
) -> pd.Series:
    """U = w * U_trad + (1-w) * U_model, aligned on accident year index."""
    w = float(np.clip(weight_traditional, 0.0, 1.0))
    years = sorted(
        set(u_traditional.dropna().index.astype(int))
        | set(u_model.dropna().index.astype(int))
    )
    a = u_traditional.reindex(years).astype(float)
    b = u_model.reindex(years).astype(float)
    return w * a + (1.0 - w) * b


def multi_average(*series: pd.Series) -> pd.Series:
    """Elementwise mean of any number of aligned series (NaN-tolerant)."""
    if not series:
        raise ValueError("multi_average requires at least one series")
    years = sorted(set().union(*(set(s.dropna().index.astype(int)) for s in series)))
    mat = np.column_stack([s.reindex(years).astype(float).values for s in series])
    return pd.Series(np.nanmean(mat, axis=1), index=years)


# --------------------------------------------------------------------------- #
# Credibility-weighted hybrid                                                  #
# --------------------------------------------------------------------------- #

def run_credibility_hybrid(
    candidates: dict[str, tuple[pd.Series, pd.Series]],
    anchor: pd.Series | None = None,
    anchor_weight: float = 0.20,
) -> pd.Series:
    """
    Uncertainty-weighted credibility ensemble of AI model ultimates.

    For each accident year, the AI models are blended with weights:

        w_i(year) = (1 / sigma_i(year)) / sum_j(1 / sigma_j(year))

    where sigma_i is the MC-dropout predictive std from model i.  Models with
    tighter predictions receive higher credibility — consistent with Bühlmann
    theory.  An optional traditional anchor model (e.g., Chain Ladder) can
    receive a fixed weight, with the remaining mass distributed among AI models.

    Parameters
    ----------
    candidates    : {name: (ult_mean, ult_std)} — AI models with UQ
    anchor        : optional traditional-method ultimate series
    anchor_weight : fixed credibility weight for the anchor (0 = pure AI)

    Returns
    -------
    pd.Series of credibility-weighted ultimates per accident year
    """
    if not candidates:
        raise ValueError("At least one AI candidate model is required.")

    all_years: set[int] = set()
    for ult_mean, ult_std in candidates.values():
        all_years |= set(ult_mean.dropna().index.astype(int))
    if anchor is not None:
        all_years |= set(anchor.dropna().index.astype(int))
    years = sorted(all_years)

    model_names = list(candidates.keys())
    means = np.column_stack(
        [candidates[m][0].reindex(years).astype(float).values for m in model_names]
    )  # (n_years, n_models)
    stds = np.column_stack(
        [candidates[m][1].reindex(years).astype(float).values for m in model_names]
    )  # (n_years, n_models)

    # Per-year inverse-std weights (epsilon guards against zero std)
    inv_std = 1.0 / (np.abs(stds) + 1e-6)
    ai_weights = inv_std / inv_std.sum(axis=1, keepdims=True)  # (n_years, n_models)
    ai_blend = (means * ai_weights).sum(axis=1)  # (n_years,)

    if anchor is not None:
        w_ai = 1.0 - float(np.clip(anchor_weight, 0.0, 1.0))
        anc = anchor.reindex(years).astype(float).values
        result = anchor_weight * anc + w_ai * ai_blend
    else:
        result = ai_blend

    return pd.Series(result, index=years, name="Credibility Hybrid")


# --------------------------------------------------------------------------- #
# Named wrappers (backward compatible)                                         #
# --------------------------------------------------------------------------- #

def run_hybrid_cl_deepresnet(
    u_chain_ladder: pd.Series, u_deep_resnet: pd.Series, weight_cl: float = 0.5
) -> pd.Series:
    return convex_blend(u_chain_ladder, u_deep_resnet, weight_cl)


def run_hybrid_cl_gru(
    u_chain_ladder: pd.Series, u_gru: pd.Series, weight_cl: float = 0.5
) -> pd.Series:
    return convex_blend(u_chain_ladder, u_gru, weight_cl)


def run_hybrid_cl_transformer(
    u_chain_ladder: pd.Series, u_transformer: pd.Series, weight_cl: float = 0.5
) -> pd.Series:
    return convex_blend(u_chain_ladder, u_transformer, weight_cl)


def run_hybrid_cl_xgboost(
    u_chain_ladder: pd.Series, u_xgb: pd.Series, weight_cl: float = 0.5
) -> pd.Series:
    return convex_blend(u_chain_ladder, u_xgb, weight_cl)


def run_hybrid_cl_bf_dl(
    u_chain_ladder: pd.Series,
    u_bornhuetter: pd.Series,
    u_deep_resnet: pd.Series,
    w_cl: float = 0.35,
    w_bf: float = 0.25,
) -> pd.Series:
    """Three-way convex combination; weights renormalised to sum to 1."""
    w_dl = 1.0 - w_cl - w_bf
    if w_dl < 0:
        raise ValueError("w_cl + w_bf must be ≤ 1")
    years = sorted(
        set(u_chain_ladder.dropna().index.astype(int))
        | set(u_bornhuetter.dropna().index.astype(int))
        | set(u_deep_resnet.dropna().index.astype(int))
    )
    return (
        w_cl * u_chain_ladder.reindex(years).astype(float)
        + w_bf * u_bornhuetter.reindex(years).astype(float)
        + w_dl * u_deep_resnet.reindex(years).astype(float)
    )
