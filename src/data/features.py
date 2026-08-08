"""
Feature engineering for the tabular reserving models.

Two parameterisations live here, and the difference matters more than the model
architecture does.

**Level** (:func:`matrix_xy`) — predict the cumulative paid directly. Kept for
the gradient-boosting models, which are tree-based and handle a wide target
range by partitioning it.

**Log development factor** (:func:`matrix_xy_logratio`) — predict
``log(C_j / C_{j-1})``. This is the right target for a neural network here. On
this triangle the cumulative spans three orders of magnitude, because twenty
years of Argentine inflation sit inside it; a network fitted on standardised
levels devotes its capacity to the scale and learns almost nothing about
development, and an autoregressive roll-forward on levels compounds that error
period after period. The log factor is scale-free and roughly stationary across
accident years, it makes the roll-forward multiplicative — ``C_{j+1} = C_j ·
exp(ŷ)`` — and it recovers the chain-ladder structure when the data really are
multiplicative, which is what the MLP is supposed to generalise.
"""

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


def logratio_feature_columns() -> list[str]:
    """Feature names for the log-development-factor parameterisation."""
    return ["development_year", "accident_year_idx"]


def _logratio_row(
    accident_year: int,
    development_year: int,
    lag_cumulative: float,
    first_year: int,
    n_years: int,
    max_dev: int,
) -> dict:
    """
    One feature row: development period and accident year, both scaled to [0, 1].

    The calendar year is deliberately **not** a feature. Every cell still to be
    projected falls in a calendar year later than any the model was trained on,
    so including that axis forces the network to extrapolate a covariate outside
    its observed range — and since the calendar-year effect on a nominal
    Argentine triangle is inflation, extrapolating it linearly in a standardised
    space compounds through the roll-forward and blows the reserve up by an order
    of magnitude. Accident year and development period are both interpolated at
    projection time, never extrapolated.

    The inflation dimension is handled where it belongs: by deflating the
    triangle and re-inflating the projected reserve under a stated assumption,
    rather than by asking the network to forecast it implicitly.

    ``lag_cumulative`` is not a feature either, for the same reason — its value
    for the immature years lies above anything seen in training. The target is
    already scale-free, so the level adds nothing.
    """
    return {
        "development_year": development_year / max_dev,
        "accident_year_idx": (accident_year - first_year) / max(n_years - 1, 1),
    }


def matrix_xy_logratio(triangle_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build ``(X, y)`` where ``y = log(C_j / C_{j-1})`` over every observed
    consecutive pair of cells.

    Non-increasing pairs are dropped: a negative development factor is not
    meaningful on paid cumulative claims and would be learnt as signal.
    """
    pivot = _cumulative_pivot(triangle_df)
    years = [int(y) for y in pivot.index]
    max_dev = int(pivot.columns.max())
    first_year, n_years = years[0], len(years)
    values = pivot.values.astype(float)

    rows, targets = [], []
    for i, ay in enumerate(years):
        for j in range(1, max_dev):
            prev, cur = values[i, j - 1], values[i, j]
            if np.isfinite(prev) and np.isfinite(cur) and prev > 0 and cur >= prev:
                rows.append(
                    _logratio_row(ay, j + 1, prev, first_year, n_years, max_dev)
                )
                targets.append(float(np.log(cur / prev)))

    return pd.DataFrame(rows, columns=logratio_feature_columns()), pd.Series(targets)


def _cumulative_pivot(triangle_df: pd.DataFrame) -> pd.DataFrame:
    pivot = triangle_df.pivot_table(
        index="accident_year",
        columns="development_year",
        values="cumulative_claims",
        aggfunc="first",
    ).sort_index()
    return pivot.reindex(columns=range(1, int(pivot.columns.max()) + 1))


def rollforward_logratio(
    triangle_df: pd.DataFrame,
    predict_log_factor: Callable[[pd.DataFrame], np.ndarray],
) -> pd.Series:
    """
    Project each accident year to the ultimate by multiplicative roll-forward.

    Starting from the latest observed cumulative, the model supplies one log
    development factor per remaining period and the cumulative is carried
    forward as ``C_{j+1} = C_j · exp(ŷ_j)``. Predicted factors are floored at
    zero so the projected cumulative can never fall below the amount already
    paid, which is the constraint a paid triangle has and an unconstrained
    network does not respect.
    """
    pivot = _cumulative_pivot(triangle_df)
    years = [int(y) for y in pivot.index]
    max_dev = int(pivot.columns.max())
    first_year, n_years = years[0], len(years)
    values = pivot.values.astype(float)

    out: dict[int, float] = {}
    for i, ay in enumerate(years):
        observed = [j for j in range(max_dev) if np.isfinite(values[i, j]) and values[i, j] > 0]
        if not observed:
            out[ay] = np.nan
            continue

        current = float(values[i, observed[-1]])
        for j in range(observed[-1], max_dev - 1):
            features = pd.DataFrame(
                [_logratio_row(ay, j + 2, current, first_year, n_years, max_dev)],
                columns=logratio_feature_columns(),
            )
            log_factor = float(np.asarray(predict_log_factor(features)).ravel()[0])
            current *= float(np.exp(max(log_factor, 0.0)))
        out[ay] = current

    return pd.Series(out).sort_index()


def predict_ultimate_per_origin(
    enriched_template: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
) -> pd.Series:
    """
    Project every accident year to the ultimate by autoregressive roll-forward.

    The models trained on :func:`matrix_xy` learn a **one-step** relationship —
    the cumulative at development ``j`` given the cumulative at ``j-1`` — so the
    projection has to be applied one period at a time, feeding each prediction
    back in as the next period's lag, exactly as the recurrent models do. Asking
    for ``development_year = max_dev`` in a single call instead pairs a lag taken
    from the first development period with a target ten periods later, a mapping
    the model never saw in training; for the most immature accident years, which
    is where the whole reserve sits, it returns roughly one period of development
    and the reserve collapses.
    """
    max_dev = int(enriched_template["development_year"].max())
    pivot = enriched_template.pivot_table(
        index="accident_year",
        columns="development_year",
        values="cumulative_claims",
        aggfunc="first",
    ).reindex(columns=range(1, max_dev + 1))

    years = sorted(int(y) for y in enriched_template["accident_year"].unique())
    out: dict[int, float] = {}

    for ay in years:
        row = pivot.loc[ay].values.astype(float)
        observed = [j for j in range(max_dev) if np.isfinite(row[j]) and row[j] > 0]
        if not observed:
            out[ay] = np.nan
            continue

        j0 = observed[-1]
        current = float(row[j0])
        for j in range(j0, max_dev - 1):
            features = pd.DataFrame(
                [
                    {
                        "accident_year": ay,
                        "development_year": j + 2,
                        "lag_cumulative": current,
                        "calendar_year": ay + j + 1,
                    }
                ]
            )
            nxt = float(np.asarray(predict_fn(features)).ravel()[0])
            # Cumulative paid cannot decrease.
            current = max(nxt, current)
        out[ay] = current

    return pd.Series(out).sort_index()
