"""Feature engineering for triangle-based ML reserve models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def enrich_triangle_features(triangle_df: pd.DataFrame) -> pd.DataFrame:
    """Add lag cumulative and calendar year; sorted by accident / development period."""
    t = triangle_df.sort_values(["accident_year", "development_year"]).copy()
    t["lag_cumulative"] = t.groupby("accident_year")["cumulative_claims"].shift(1)
    t["calendar_year"] = t["accident_year"] + t["development_year"] - 1
    return t


def training_mask(triangle_df: pd.DataFrame) -> pd.Series:
    """Rows with observed cumulative (lower triangle) for supervised training."""
    return triangle_df["cumulative_claims"].notna()


def feature_columns() -> list[str]:
    return [
        "accident_year",
        "development_year",
        "lag_cumulative",
        "calendar_year",
    ]


def matrix_xy(enriched: pd.DataFrame, target_col: str = "cumulative_claims"):
    m = training_mask(enriched)
    X = enriched.loc[m, feature_columns()].copy()
    X["lag_cumulative"] = X["lag_cumulative"].fillna(0.0)
    y = enriched.loc[m, target_col]
    return X, y


def predict_ultimate_per_origin(
    enriched_template: pd.DataFrame,
    predict_fn,
) -> pd.Series:
    """
    For each accident year, predict cumulative loss at the global maximum
    development period (proxy for ultimate in a fixed run-off horizon).
    """
    max_dev = int(enriched_template["development_year"].max())
    pivot = enriched_template.pivot_table(
        index="accident_year",
        columns="development_year",
        values="cumulative_claims",
        aggfunc="first",
    )
    years = sorted(enriched_template["accident_year"].unique())
    rows = []
    for ay in years:
        lag = np.nan
        if max_dev > 1 and (max_dev - 1) in pivot.columns:
            lag = pivot.loc[ay, max_dev - 1]
        if pd.isna(lag):
            sub = enriched_template[
                (enriched_template["accident_year"] == ay)
                & enriched_template["cumulative_claims"].notna()
            ]
            if len(sub) > 0:
                lag = sub.sort_values("development_year")["cumulative_claims"].iloc[-1]
        cy = int(ay) + max_dev - 1
        rows.append(
            {
                "accident_year": ay,
                "development_year": max_dev,
                "lag_cumulative": 0.0 if pd.isna(lag) else float(lag),
                "calendar_year": cy,
            }
        )
    X_tail = pd.DataFrame(rows)
    pred = predict_fn(X_tail)
    return pd.Series(pred, index=[int(y) for y in years]).sort_index()
