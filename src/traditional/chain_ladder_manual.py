"""
Volume-weighted chain ladder and derived reserving methods (pandas / numpy only).

All public functions accept a pivot DataFrame (rows = accident years,
columns = development years, values = cumulative paid claims).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_ldfs(C: np.ndarray) -> np.ndarray:
    """
    Volume-weighted loss development factors for each development column.

    LDF_j = sum_i(C[i, j+1]) / sum_i(C[i, j])  for all rows i where both
    cells are finite and C[i, j] > 0.
    """
    n_ay, n_dev = C.shape
    ldfs = np.ones(n_dev - 1)
    for j in range(n_dev - 1):
        num = den = 0.0
        for i in range(n_ay):
            a, b = C[i, j], C[i, j + 1]
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                num += b
                den += a
        if den > 0:
            ldfs[j] = num / den
    return ldfs


def _last_observed(row: np.ndarray) -> int | None:
    """Return the 0-based column index of the last finite, positive entry."""
    for j in range(len(row) - 1, -1, -1):
        if np.isfinite(row[j]) and row[j] > 0:
            return j
    return None


def cumulative_pivot_from_long(triangle_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a long-format triangle DataFrame to an accident × development pivot."""
    t = triangle_df.sort_values(["accident_year", "development_year"])
    p = t.pivot(index="accident_year", columns="development_year", values="cumulative_claims")
    p = p.sort_index()
    p.columns = [int(c) for c in p.columns]
    return p


def ultimate_from_pivot(pivot: pd.DataFrame) -> pd.Series:
    """
    Chain-ladder ultimate per accident year.

    Each row is projected to the last development column by multiplying the
    latest observed cumulative by the product of the remaining volume-weighted
    LDFs.
    """
    C = pivot.sort_index().values.astype(float)
    n_ay, n_dev = C.shape
    ldfs = _weighted_ldfs(C)
    ult = np.full(n_ay, np.nan)

    for i in range(n_ay):
        j0 = _last_observed(C[i])
        if j0 is None:
            continue
        u = C[i, j0]
        for j in range(j0, n_dev - 1):
            u *= ldfs[j]
        ult[i] = u

    return pd.Series(ult, index=pivot.index.astype(int), name="ultimate")


def bornhuetter_ferguson_ultimate(
    pivot: pd.DataFrame,
    apriori: pd.Series | None = None,
) -> pd.Series:
    """
    Bornhuetter–Ferguson ultimates.

    Formula: U = C + (a − C) × (1 − 1/F)

    where C is the latest cumulative, a is the a-priori expected ultimate, and
    F is the cumulative LDF from the current development column to the tail.

    If ``apriori`` is None, the volume-weighted mean chain-ladder ultimate is
    used as a constant prior across all accident years.
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
        j0 = _last_observed(C[i])
        if j0 is None or not np.isfinite(a[i]):
            continue
        F = 1.0
        for j in range(j0, n_dev - 1):
            F *= ldfs[j]
        if F <= 0 or not np.isfinite(F):
            continue
        c_ij = C[i, j0]
        out[i] = c_ij + (a[i] - c_ij) * (1.0 - 1.0 / F)

    return pd.Series(out, index=pivot.index.astype(int), name="bf_ultimate")


def cape_cod_proxy(pivot: pd.DataFrame) -> pd.Series:
    """
    Cape Cod stand-in (chain-ladder ultimates).

    A true Cape Cod requires on-level earned premium by accident year.
    Without that exposure triangle this function returns chain-ladder ultimates
    as a transparent proxy; the caller should document this in results.
    """
    return ultimate_from_pivot(pivot)
