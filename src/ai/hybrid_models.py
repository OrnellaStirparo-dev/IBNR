"""
Credibility hybrid (Sección 4.2): an uncertainty-weighted blend of the neural
models, anchored on the chain ladder.

Bühlmann credibility gives more weight to the estimator with less variance, so
the blend across the neural models weights each one by its **precision**, the
reciprocal of its MC-dropout predictive variance. Weighting by ``1/σ`` instead
of ``1/σ²`` is a different estimator and is not the credibility-optimal one.

The chain-ladder anchor enters with a fixed weight. That weight is a judgement
call, not a result: :func:`src.evaluation.sensitivity.anchor_weight_sweep`
traces the reserve across the whole range, and on this triangle the curve is
close to linear — the hybrid interpolates between a neural ensemble that
under-reserves and a chain ladder that over-reserves rather than synthesising
anything the components do not already contain. Worth stating plainly when the
result is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def convex_blend(first: pd.Series, second: pd.Series, weight_first: float = 0.5) -> pd.Series:
    """Weighted average of two ultimate series, aligned on the union of years."""
    weight = float(np.clip(weight_first, 0.0, 1.0))
    years = sorted(set(first.dropna().index.astype(int)) | set(second.dropna().index.astype(int)))
    blended = (
        weight * first.reindex(years).astype(float)
        + (1.0 - weight) * second.reindex(years).astype(float)
    )
    return pd.Series(blended.values, index=years, name="hybrid")


def run_credibility_hybrid(
    candidates: dict[str, tuple[pd.Series, pd.Series]],
    anchor: pd.Series | None = None,
    anchor_weight: float = 0.20,
) -> pd.Series:
    """
    Blend the neural ultimates by precision and mix in the anchor.

    For each accident year the neural models are combined with weights
    ``w_i = (1/σ_i²) / Σ_j (1/σ_j²)``, and the result is mixed with *anchor* at
    a fixed ``anchor_weight``.

    Parameters
    ----------
    candidates    : ``{name: (ultimate_mean, ultimate_std)}``
    anchor        : traditional-method ultimates, or None for a pure ensemble
    anchor_weight : credibility given to the anchor, clipped to [0, 1]
    """
    if not candidates:
        raise ValueError("At least one candidate model is required.")

    years: set[int] = set()
    for mean, _ in candidates.values():
        years |= set(mean.dropna().index.astype(int))
    if anchor is not None:
        years |= set(anchor.dropna().index.astype(int))
    years = sorted(years)

    names = list(candidates)
    means = np.column_stack([candidates[n][0].reindex(years).astype(float).values for n in names])
    stds = np.column_stack([candidates[n][1].reindex(years).astype(float).values for n in names])

    # Precision weights. The epsilon keeps a model with zero predictive spread
    # — which happens on the fully developed accident years, where every model
    # returns the observed cumulative — from taking the entire weight.
    precision = 1.0 / (stds.astype(float) ** 2 + 1e-9)
    weights = precision / precision.sum(axis=1, keepdims=True)
    ensemble = np.nansum(means * weights, axis=1)

    if anchor is None:
        return pd.Series(ensemble, index=years, name="Credibility Hybrid")

    weight = float(np.clip(anchor_weight, 0.0, 1.0))
    anchored = weight * anchor.reindex(years).astype(float).values + (1.0 - weight) * ensemble
    return pd.Series(anchored, index=years, name="Credibility Hybrid")
