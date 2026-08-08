"""
Run a reserving model in constant currency and bring the answer back to nominal.

The problem
-----------
Every method here — chain ladder and neural alike — learns a development pattern
from the triangle. On nominal Argentine data that pattern is a mixture of two
things: how claims actually settle, and how much the peso lost between the
accident and the payment. Chain ladder's own assumption is that the pattern is
stable across accident years, and with severity trending by two orders of
magnitude over the window it is not, for reasons that have nothing to do with
claims handling.

The procedure
-------------
1. Deflate the triangle to a base year, by calendar year of payment.
2. Fit and project the model there, where the pattern is closer to stationary.
3. Split the projected real-terms reserve over the calendar years in which the
   payments will actually fall.
4. Re-inflate each of those slices at its own calendar-year index and add them
   back to the nominal paid-to-date.

Step 3 is what makes step 4 correct. A reserve is not money of the accident
year — it is money of the future years in which it will be paid, and under
three-digit inflation the difference between re-inflating at the accident year's
index and at the payment years' indices is most of the reserve.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.data.inflation import deflate_triangle, price_index
from src.traditional.chain_ladder_manual import (
    _last_observed,
    _weighted_ldfs,
    cumulative_pivot_from_long,
)


def _payment_shares(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Share of each accident year's outstanding reserve falling in each future
    development period, from the chain-ladder pattern of the (real-terms)
    triangle.

    Rows are accident years, columns development periods; each row sums to 1
    over the periods still to be paid, and is empty for a fully developed year.
    """
    C = pivot.sort_index().values.astype(float)
    n_ay, n_dev = C.shape
    ldfs = _weighted_ldfs(C)

    shares = np.zeros((n_ay, n_dev))
    for i in range(n_ay):
        j0 = _last_observed(C[i])
        if j0 is None or j0 >= n_dev - 1:
            continue
        path = [C[i, j0]]
        for j in range(j0, n_dev - 1):
            path.append(path[-1] * ldfs[j])
        increments = np.diff(path)
        total = increments.sum()
        if total <= 0:
            continue
        for offset, inc in enumerate(increments):
            shares[i, j0 + 1 + offset] = inc / total

    return pd.DataFrame(shares, index=pivot.sort_index().index.astype(int),
                        columns=list(pivot.columns))


def reinflate_ultimates(
    real_ultimates: pd.Series,
    real_pivot: pd.DataFrame,
    nominal_pivot: pd.DataFrame,
    base_year: int,
) -> pd.Series:
    """
    Convert accident-year ultimates from constant currency back to nominal.

    The reserve implied in real terms is spread over future development periods
    with :func:`_payment_shares`, each slice is inflated at the index of the
    calendar year it will be paid in, and the result is added to the nominal
    amount already paid.
    """
    shares = _payment_shares(real_pivot)
    idx = pivot_calendar_index(real_pivot, base_year)

    real_latest = real_pivot.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)
    nominal_latest = nominal_pivot.apply(
        lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1
    )

    out = {}
    for ay in nominal_pivot.index.astype(int):
        if ay not in real_ultimates.index or not np.isfinite(real_ultimates[ay]):
            out[ay] = np.nan
            continue
        reserve_real = float(real_ultimates[ay]) - float(real_latest[ay])
        if reserve_real <= 0 or ay not in shares.index:
            out[ay] = float(nominal_latest[ay]) + max(reserve_real, 0.0)
            continue
        nominal_reserve = sum(
            reserve_real * float(share) * float(idx.at[ay, dev])
            for dev, share in shares.loc[ay].items()
            if share > 0
        )
        out[ay] = float(nominal_latest[ay]) + nominal_reserve

    return pd.Series(out, name=real_ultimates.name).sort_index()


def pivot_calendar_index(pivot: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """Price index of the calendar year of each (accident year, development) cell."""
    years = sorted(
        {int(ay) + int(dev) - 1 for ay in pivot.index for dev in pivot.columns}
        | {base_year}
    )
    idx = price_index(base_year, years)
    return pd.DataFrame(
        [[float(idx[int(ay) + int(dev) - 1]) for dev in pivot.columns] for ay in pivot.index],
        index=pivot.index.astype(int),
        columns=list(pivot.columns),
    )


def run_in_real_terms(
    triangle_df: pd.DataFrame,
    model: Callable[[pd.DataFrame], pd.Series],
    base_year: int,
) -> pd.Series:
    """
    Fit *model* on the deflated triangle and return **nominal** ultimates.

    ``model`` takes a long-format triangle and returns accident-year ultimates,
    which is the signature every ``run_*`` function in this project already has.
    """
    real_df = deflate_triangle(triangle_df, base_year=base_year, by="calendar")
    real_ultimates = model(real_df)
    return reinflate_ultimates(
        real_ultimates,
        cumulative_pivot_from_long(real_df),
        cumulative_pivot_from_long(triangle_df),
        base_year,
    )
