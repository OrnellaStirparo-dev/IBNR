"""Feature engineering for tabular ML reserve models (gradient boosting)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def enrich_triangle_features(triangle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``lag_cumulative`` and ``calendar_year`` to a long triangle.

    - ``lag_cumulative``: cumulative paid from the previous development year
      (within the same accident year).  NaN for the first development period.
    - ``calendar_year``: accident_year + development_year − 1  (diagonal index).
    """
    t = triangle_df.sort_values(["accident_year", "development_year"]).copy()
    t["lag_cumulative"] = t.groupby("accident_year")["cumulative_claims"].shift(1)
    t["calendar_year"] = t["accident_year"] + t["development_year"] - 1
    return t


def training_mask(triangle_df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting rows with observed cumulative claims (lower triangle)."""
    return triangle_df["cumulative_claims"].notna()


def feature_columns() -> list[str]:
    """Ordered list of feature column names used by all tabular models."""
    return ["accident_year", "development_year", "lag_cumulative", "calendar_year"]


def matrix_xy(
    enriched: pd.DataFrame, target_col: str = "cumulative_claims"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from the lower triangle (observed cells only)."""
    m = training_mask(enriched)
    X = enriched.loc[m, feature_columns()].copy()
    X["lag_cumulative"] = X["lag_cumulative"].fillna(0.0)
    y = enriched.loc[m, target_col]
    return X, y


def predict_ultimate_per_origin(
    enriched_template: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
) -> pd.Series:
    """
    Apply *predict_fn* at ``development_year = max_dev`` for every accident year.

    Builds a one-row-per-accident-year feature matrix targeting the last
    development column (the model's proxy for the run-off ultimate), then
    returns a Series indexed by integer accident year.
    """
    max_dev = int(enriched_template["development_year"].max())
    pivot = enriched_template.pivot_table(
        index="accident_year",
        columns="development_year",
        values="cumulative_claims",
        aggfunc="first",
    )
    years = sorted(enriched_template["accident_year"].unique())
    rows: list[dict] = []
    for ay in years:
        lag = np.nan
        if max_dev > 1 and (max_dev - 1) in pivot.columns:
            lag = pivot.loc[ay, max_dev - 1]
        if pd.isna(lag):
            sub = enriched_template[
                (enriched_template["accident_year"] == ay)
                & enriched_template["cumulative_claims"].notna()
            ]
            if len(sub):
                lag = sub.sort_values("development_year")["cumulative_claims"].iloc[-1]
        rows.append({
            "accident_year": ay,
            "development_year": max_dev,
            "lag_cumulative": 0.0 if pd.isna(lag) else float(lag),
            "calendar_year": int(ay) + max_dev - 1,
        })
    pred = predict_fn(pd.DataFrame(rows))
    return pd.Series(pred, index=[int(y) for y in years]).sort_index()
