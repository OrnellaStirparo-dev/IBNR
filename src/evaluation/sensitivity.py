"""
Sensitivity of the reserve to the two free choices the models do not pin down.

Both are judgement calls that a reviewer is entitled to ask about, and neither
is visible in a single headline number:

``anchor_weight``
    How much credibility the hybrid gives the chain-ladder anchor. Fixing it at
    20% without evidence is an assumption dressed as a result.

``future inflation``
    Required to bring a constant-currency reserve back to nominal. A nominal
    chain ladder does not avoid this choice, it only conceals it — its factors
    silently extrapolate whatever inflation the historical triangle happens to
    contain.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.ai.hybrid_models import run_credibility_hybrid


def anchor_weight_sweep(
    ai_candidates: dict[str, tuple[pd.Series, pd.Series]],
    anchor: pd.Series,
    latest: pd.Series,
    true_ultimates: pd.Series | None = None,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Reserve implied by the credibility hybrid across anchor weights 0 → 1.

    Weight 0 is a pure AI ensemble, weight 1 is the anchor itself; the useful
    reading is where in between the error curve actually bottoms out, and how
    flat it is around that point.
    """
    grid = np.linspace(0.0, 1.0, 21) if weights is None else np.asarray(weights)
    rows = []
    for w in grid:
        ult = run_credibility_hybrid(ai_candidates, anchor=anchor, anchor_weight=float(w))
        ibnr = float((ult - latest.reindex(ult.index)).sum())
        row = {"anchor_weight": float(w), "ibnr_total": ibnr}
        if true_ultimates is not None:
            real = float((true_ultimates.reindex(latest.index) - latest).sum())
            row["error_pct"] = 100.0 * (ibnr - real) / real if real else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def future_inflation_sweep(
    triangle_df: pd.DataFrame,
    model: Callable[[pd.DataFrame], pd.Series],
    base_year: int,
    latest: pd.Series,
    true_ultimates: pd.Series | None = None,
    rates: list[float] | None = None,
) -> pd.DataFrame:
    """
    Nominal reserve produced by a constant-currency fit across future-inflation
    assumptions.

    The development pattern is identical at every rate — only the conversion of
    the projected reserve back to nominal changes. What the spread measures is
    how much of the reserve is an inflation forecast rather than a claims
    estimate.
    """
    from src.data.inflation import deflate_triangle
    from src.evaluation.real_terms import reinflate_ultimates
    from src.traditional.chain_ladder_manual import cumulative_pivot_from_long
    import src.data.inflation as inflation_module

    grid = rates if rates is not None else [0.0, 20.0, 40.0, 60.0, 80.0, 120.0]
    nominal_pivot = cumulative_pivot_from_long(triangle_df)

    original = inflation_module.DEFAULT_FUTURE_INFLATION_PCT
    rows = []
    try:
        for rate in grid:
            inflation_module.DEFAULT_FUTURE_INFLATION_PCT = float(rate)
            real_df = deflate_triangle(triangle_df, base_year=base_year, by="calendar")
            real_pivot = cumulative_pivot_from_long(real_df)
            nominal = reinflate_ultimates(
                model(real_df), real_pivot, nominal_pivot, base_year
            )
            ibnr = float((nominal - latest.reindex(nominal.index)).sum())
            row = {"inflacion_futura_pct": float(rate), "ibnr_total": ibnr}
            if true_ultimates is not None:
                real = float((true_ultimates.reindex(latest.index) - latest).sum())
                row["error_pct"] = 100.0 * (ibnr - real) / real if real else np.nan
            rows.append(row)
    finally:
        inflation_module.DEFAULT_FUTURE_INFLATION_PCT = original

    return pd.DataFrame(rows)
