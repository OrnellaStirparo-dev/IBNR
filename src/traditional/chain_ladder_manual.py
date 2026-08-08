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


def _cumulative_ldf_to_ultimate(ldfs: np.ndarray, j0: int, n_dev: int) -> float:
    """Cumulative development factor F from column *j0* to the tail."""
    F = 1.0
    for j in range(j0, n_dev - 1):
        F *= ldfs[j]
    return F


def _fully_developed_mask(C: np.ndarray) -> np.ndarray:
    """True for accident years observed all the way to the last development column."""
    n_ay, n_dev = C.shape
    return np.array([_last_observed(C[i]) == n_dev - 1 for i in range(n_ay)])


def trended_apriori(
    pivot: pd.DataFrame,
    exposure: pd.Series | None = None,
) -> pd.Series:
    """
    Build an a-priori expected ultimate per accident year.

    The a-priori must be expressed **in the nominal currency of its own accident
    year**. Using a single constant across accident years is invalid whenever
    severity trends — under Argentine inflation the prior for an early year is
    orders of magnitude below the amount already paid for a recent one, which
    drives the Bornhuetter–Ferguson ultimate below the latest cumulative and
    produces a negative (inadmissible) reserve.

    Two constructions, in order of preference:

    ``exposure`` given
        Burning-cost prior: ``a_i = exposure_i × mean_j(U_j / exposure_j)`` over
        the fully developed accident years *j*. ``exposure`` stands in for
        on-level earned premium (e.g. vehicles exposed × severity index).

    ``exposure`` omitted
        Log-linear trend fitted to the chain-ladder ultimates of the fully
        developed accident years and extrapolated forward. Purely data-driven,
        but it can only extrapolate the *historical* trend — if severity
        accelerated after the fully developed window, the prior will lag.
    """
    pivot = pivot.sort_index()
    C = pivot.values.astype(float)
    idx = pivot.index.astype(int)
    u_cl = ultimate_from_pivot(pivot).values
    developed = _fully_developed_mask(C) & np.isfinite(u_cl)

    if developed.sum() < 2:
        raise ValueError(
            "Need at least 2 fully developed accident years to build an a-priori."
        )

    if exposure is not None:
        expo = exposure.reindex(idx).values.astype(float)
        if not np.all(np.isfinite(expo)) or np.any(expo <= 0):
            raise ValueError("exposure must be positive and defined for every accident year.")
        burning_cost = float(np.mean(u_cl[developed] / expo[developed]))
        return pd.Series(expo * burning_cost, index=idx, name="apriori")

    x = np.arange(len(idx), dtype=float)
    slope, intercept = np.polyfit(x[developed], np.log(u_cl[developed]), 1)
    return pd.Series(np.exp(intercept + slope * x), index=idx, name="apriori")


def bornhuetter_ferguson_ultimate(
    pivot: pd.DataFrame,
    apriori: pd.Series | None = None,
    exposure: pd.Series | None = None,
) -> pd.Series:
    """
    Bornhuetter–Ferguson ultimates.

    Formula (Bornhuetter & Ferguson, 1972)::

        U_i = C_i + (1 − 1/F_i) × a_i

    where ``C_i`` is the latest cumulative paid, ``a_i`` the a-priori expected
    ultimate for that accident year, and ``F_i`` the cumulative development
    factor from the current column to the tail.

    Because ``a_i ≥ 0`` and ``F_i ≥ 1``, this always yields ``U_i ≥ C_i`` and
    therefore a non-negative IBNR. Note that BF applies the unreported
    proportion to the *a-priori*, never to ``a_i − C_i``: the latter is the
    Benktander/credibility blend ``U = C/F + a(1 − 1/F)``, which can fall below
    the amount already paid and is not the Bornhuetter–Ferguson estimator.

    If ``apriori`` is None it is built with :func:`trended_apriori` (using
    ``exposure`` when supplied).
    """
    pivot = pivot.sort_index().copy()
    C = pivot.values.astype(float)
    n_ay, n_dev = C.shape
    ldfs = _weighted_ldfs(C)
    idx = pivot.index.astype(int)

    if apriori is None:
        apriori = trended_apriori(pivot, exposure=exposure)
    a = apriori.reindex(idx).values.astype(float)

    out = np.full(n_ay, np.nan)
    for i in range(n_ay):
        j0 = _last_observed(C[i])
        if j0 is None or not np.isfinite(a[i]):
            continue
        F = _cumulative_ldf_to_ultimate(ldfs, j0, n_dev)
        if F <= 0 or not np.isfinite(F):
            continue
        out[i] = C[i, j0] + (1.0 - 1.0 / F) * a[i]

    return pd.Series(out, index=idx, name="bf_ultimate")


def cape_cod_ultimate(
    pivot: pd.DataFrame,
    exposure: pd.Series | None = None,
) -> pd.Series:
    """
    Cape Cod (Bühlmann–Straub, 1983) ultimates.

    Estimates a single expected loss ratio from the triangle itself rather than
    taking it as external input::

        ELR = Σ_i C_i / Σ_i (exposure_i / F_i)
        U_i = C_i + (1 − 1/F_i) × ELR × exposure_i

    Because the a-priori is re-derived from the observed experience, Cape Cod is
    materially more robust than BF when the external prior is stale — which is
    the failure mode under sustained inflation. With ``exposure`` omitted every
    accident year is given unit exposure, so the ELR absorbs the trend only
    partially; supply a severity/volume index for a meaningful result.
    """
    pivot = pivot.sort_index()
    C = pivot.values.astype(float)
    n_ay, n_dev = C.shape
    ldfs = _weighted_ldfs(C)
    idx = pivot.index.astype(int)

    expo = (
        np.ones(n_ay)
        if exposure is None
        else exposure.reindex(idx).values.astype(float)
    )

    used_c = np.full(n_ay, np.nan)
    used_f = np.full(n_ay, np.nan)
    for i in range(n_ay):
        j0 = _last_observed(C[i])
        if j0 is None:
            continue
        used_c[i] = C[i, j0]
        used_f[i] = _cumulative_ldf_to_ultimate(ldfs, j0, n_dev)

    ok = np.isfinite(used_c) & np.isfinite(used_f) & np.isfinite(expo) & (expo > 0)
    elr = float(np.sum(used_c[ok]) / np.sum(expo[ok] / used_f[ok]))

    out = np.full(n_ay, np.nan)
    out[ok] = used_c[ok] + (1.0 - 1.0 / used_f[ok]) * elr * expo[ok]
    return pd.Series(out, index=idx, name="cape_cod_ultimate")
