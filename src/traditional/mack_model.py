"""
Mack (1993) distribution-free chain ladder.

The point estimate of the ultimate coincides with the ordinary volume-weighted
chain ladder; Mack's contribution is a closed-form mean squared error of
prediction (MSEP) for the reserve of each accident year and for the total,
derived from three assumptions on the *rows* (accident years are independent),
on the conditional expectation of the link ratios, and on the conditional
variance being proportional to the cumulative.

Reference
---------
Mack, T. (1993). Distribution-free calculation of the standard error of chain
ladder reserve estimates. *ASTIN Bulletin, 23*(2), 213–225.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.traditional.chain_ladder_manual import (
    _last_observed,
    _weighted_ldfs,
    cumulative_pivot_from_long,
    ultimate_from_pivot,
)


def _sigma_squared(C: np.ndarray, ldfs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mack's σ̂²_j and the column volumes S_j = Σ_i C[i, j].

    σ̂²_j = 1/(n_j − 1) · Σ_i C[i,j] · (C[i,j+1]/C[i,j] − f_j)²

    over the accident years i where both C[i,j] and C[i,j+1] are observed. The
    last column has no residual degrees of freedom; Mack's log-linear
    extrapolation ``min(σ̂⁴_{J-1}/σ̂²_{J-2}, min(σ̂²_{J-2}, σ̂²_{J-1}))`` is used.
    """
    n_ay, n_dev = C.shape
    sigma2 = np.zeros(n_dev - 1)
    volumes = np.zeros(n_dev - 1)

    for j in range(n_dev - 1):
        rows = [
            i
            for i in range(n_ay)
            if np.isfinite(C[i, j]) and np.isfinite(C[i, j + 1]) and C[i, j] > 0
        ]
        volumes[j] = sum(C[i, j] for i in rows)
        if len(rows) >= 2:
            ss = sum(C[i, j] * (C[i, j + 1] / C[i, j] - ldfs[j]) ** 2 for i in rows)
            sigma2[j] = ss / (len(rows) - 1)
        else:
            sigma2[j] = np.nan

    last = len(sigma2) - 1
    if not np.isfinite(sigma2[last]) or sigma2[last] == 0:
        if last >= 2 and sigma2[last - 2] > 0:
            sigma2[last] = min(
                sigma2[last - 1] ** 2 / sigma2[last - 2],
                min(sigma2[last - 2], sigma2[last - 1]),
            )
        elif last >= 1:
            sigma2[last] = sigma2[last - 1]
        else:
            sigma2[last] = 0.0

    return np.nan_to_num(sigma2, nan=0.0), volumes


def mack_chainladder(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Mack chain ladder with prediction error.

    Returns a DataFrame indexed by accident year with columns:

    ``latest``      latest observed cumulative
    ``ultimate``    projected ultimate (identical to plain chain ladder)
    ``ibnr``        ultimate − latest
    ``mse``         mean squared error of prediction of the reserve
    ``se``          standard error of the reserve, √mse
    ``cv``          coefficient of variation, se / ibnr

    A ``TOTAL`` row is appended; its standard error includes the covariance
    between accident years induced by sharing the estimated link ratios, so it
    is larger than the square root of the sum of the individual MSEs.
    """
    pivot = pivot.sort_index()
    C = pivot.values.astype(float)
    n_ay, n_dev = C.shape
    idx = pivot.index.astype(int)

    ldfs = _weighted_ldfs(C)
    sigma2, volumes = _sigma_squared(C, ldfs)
    ultimate = ultimate_from_pivot(pivot).values

    # Full projected run-off square: observed cells kept, future cells projected.
    proj = C.copy()
    first_future = np.full(n_ay, n_dev, dtype=int)
    for i in range(n_ay):
        j0 = _last_observed(C[i])
        if j0 is None:
            continue
        first_future[i] = j0 + 1
        for j in range(j0, n_dev - 1):
            proj[i, j + 1] = proj[i, j] * ldfs[j]

    latest = np.array(
        [C[i, _last_observed(C[i])] if _last_observed(C[i]) is not None else np.nan
         for i in range(n_ay)]
    )

    # Per-accident-year MSEP.
    mse = np.zeros(n_ay)
    for i in range(n_ay):
        j0 = _last_observed(C[i])
        if j0 is None or j0 >= n_dev - 1:
            continue
        acc = 0.0
        for j in range(j0, n_dev - 1):
            if ldfs[j] <= 0 or volumes[j] <= 0 or proj[i, j] <= 0:
                continue
            acc += (sigma2[j] / ldfs[j] ** 2) * (1.0 / proj[i, j] + 1.0 / volumes[j])
        mse[i] = ultimate[i] ** 2 * acc

    # Total MSEP: Σ_i mse_i + 2 Σ_{i<k} covariance term (Mack 1993, eq. 7).
    total_mse = float(np.nansum(mse))
    for i in range(n_ay):
        j0_i = _last_observed(C[i])
        if j0_i is None or j0_i >= n_dev - 1:
            continue
        cross = 0.0
        for j in range(j0_i, n_dev - 1):
            if ldfs[j] <= 0 or volumes[j] <= 0:
                continue
            cross += 2.0 * sigma2[j] / (ldfs[j] ** 2 * volumes[j])
        others = float(
            np.nansum([ultimate[k] for k in range(i + 1, n_ay)
                       if first_future[k] <= n_dev - 1])
        )
        total_mse += ultimate[i] * others * cross

    ibnr = ultimate - latest
    se = np.sqrt(np.maximum(mse, 0.0))

    out = pd.DataFrame(
        {
            "latest": latest,
            "ultimate": ultimate,
            "ibnr": ibnr,
            "mse": mse,
            "se": se,
            "cv": np.where(np.abs(ibnr) > 0, se / np.where(ibnr == 0, np.nan, ibnr), np.nan),
        },
        index=idx,
    )
    total_se = float(np.sqrt(max(total_mse, 0.0)))
    total_ibnr = float(np.nansum(ibnr))
    out.loc["TOTAL"] = {
        "latest": float(np.nansum(latest)),
        "ultimate": float(np.nansum(ultimate)),
        "ibnr": total_ibnr,
        "mse": total_mse,
        "se": total_se,
        "cv": total_se / total_ibnr if total_ibnr else np.nan,
    }
    return out


def run_mack_chainladder(triangle_df: pd.DataFrame) -> pd.Series:
    """Mack chain-ladder ultimates per accident year (point estimates)."""
    pivot = cumulative_pivot_from_long(triangle_df)
    return ultimate_from_pivot(pivot)


def run_mack_with_error(triangle_df: pd.DataFrame) -> pd.DataFrame:
    """Mack chain ladder with per-accident-year and total prediction error."""
    return mack_chainladder(cumulative_pivot_from_long(triangle_df))
