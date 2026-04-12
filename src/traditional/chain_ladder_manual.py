"""
Volume-weighted chain ladder and related helpers (pandas / numpy only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_ldfs(C: np.ndarray) -> np.ndarray:
    """Volume-weighted LDF from column j to j+1."""
    n_ay, n_dev = C.shape
    ldfs = np.ones(n_dev - 1)
    for j in range(n_dev - 1):
        num = 0.0
        den = 0.0
        for i in range(n_ay):
            a, b = C[i, j], C[i, j + 1]
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                num += b
                den += a
        if den > 0:
            ldfs[j] = num / den
    return ldfs


def cumulative_pivot_from_long(triangle_df: pd.DataFrame) -> pd.DataFrame:
    t = triangle_df.sort_values(["accident_year", "development_year"])
    p = t.pivot(index="accident_year", columns="development_year", values="cumulative_claims")
    p = p.sort_index()
    p.columns = [int(c) for c in p.columns]
    return p


def ultimate_from_pivot(pivot: pd.DataFrame) -> pd.Series:
    """Chain-ladder ultimate per accident year (last development column)."""
    pivot = pivot.sort_index().copy()
    C = pivot.values.astype(float)
    n_ay, n_dev = C.shape
    ldfs = _weighted_ldfs(C)
    ult = np.full(n_ay, np.nan)
    for i in range(n_ay):
        last_j = None
        for j in range(n_dev - 1, -1, -1):
            if np.isfinite(C[i, j]) and C[i, j] > 0:
                last_j = j
                break
        if last_j is None:
            continue
        u = C[i, last_j]
        for j in range(last_j, n_dev - 1):
            u *= ldfs[j]
        ult[i] = u
    return pd.Series(ult, index=pivot.index.astype(int), name="ultimate")


def bornhuetter_ferguson_ultimate(
    pivot: pd.DataFrame,
    apriori: pd.Series | None = None,
) -> pd.Series:
    """
    Bornhuetter–Ferguson ultimates using cumulative LDFs.

    For each origin, ``U = C + (a - C) * (1 - 1/F)`` where ``C`` is cumulative
    at the latest observed development, ``a`` is the prior ultimate for that
    origin, and ``F`` is the cumulative LDF from the current column to the tail.
    If ``apriori`` is None, uses the **volume-weighted mean** chain-ladder ultimate
    across years as a constant prior (classic BF when no external prior exists).
    """
    pivot = pivot.sort_index().copy()
    C = pivot.values.astype(float)
    n_ay, n_dev = C.shape
    ldfs = _weighted_ldfs(C)
    u_cl = ultimate_from_pivot(pivot)
    if apriori is None:
        mean_ult = float(np.nanmean(u_cl.values))
        a = np.full(n_ay, mean_ult)
    else:
        a = apriori.reindex(pivot.index.astype(int)).values
    out = np.full(n_ay, np.nan)
    for i in range(n_ay):
        last_j = None
        for j in range(n_dev - 1, -1, -1):
            if np.isfinite(C[i, j]) and C[i, j] > 0:
                last_j = j
                break
        if last_j is None or not np.isfinite(a[i]):
            continue
        F = 1.0
        for j in range(last_j, n_dev - 1):
            F *= ldfs[j]
        if F <= 0 or not np.isfinite(F):
            continue
        c_ij = C[i, last_j]
        out[i] = c_ij + (a[i] - c_ij) * (1.0 - 1.0 / F)
    return pd.Series(out, index=pivot.index.astype(int), name="bf_ultimate")


def cape_cod_proxy(pivot: pd.DataFrame) -> pd.Series:
    """
    Cape Cod is not replicated here without earned premium / exposure triangles.
    As a practical stand-in, return chain-ladder ultimates (document in README).
    """
    return ultimate_from_pivot(pivot)
