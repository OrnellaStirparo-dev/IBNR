"""Aggregate model results into comparison tables and summary DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import calculate_metrics


def _latest_reported_per_accident_year(triangle_df: pd.DataFrame) -> pd.Series:
    """Last observed cumulative per accident year (ignores NaN upper-triangle cells)."""
    def _last_valid(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    out = (
        triangle_df.sort_values("development_year")
        .groupby("accident_year")["cumulative_claims"]
        .apply(_last_valid)
    )
    out.index = [int(x) for x in out.index]
    return out


def _extract_ultimate(result: pd.Series | pd.DataFrame) -> pd.Series | None:
    """
    Pull an accident-year-indexed ultimate series from whatever a model returns.
    Returns None if the result cannot be interpreted.
    """
    if isinstance(result, pd.Series):
        s = result.copy()
    elif isinstance(result, pd.DataFrame):
        if "ultimate" in result.columns:
            s = result["ultimate"]
        elif len(result.columns) > 0:
            s = result.iloc[:, -1]
        else:
            return None
    else:
        return None
    s = s.copy()
    s.index = [int(x) for x in s.index]
    return s


def compare_models_results(
    results: dict[str, pd.Series | pd.DataFrame],
    triangle_df: pd.DataFrame,
    true_ultimates: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build a per-model metrics table (MAE, RMSE, MAPE, R², total ultimate, total IBNR).

    Scoring target
    --------------
    ``true_ultimates`` given
        Errors are measured against the realised ultimate per accident year.
        This is the only valid accuracy comparison, and it is available here
        because the triangle is synthetic (see the ``ultimates_reales`` sheet
        written by the generator).

    ``true_ultimates`` omitted
        Falls back to the latest observed diagonal. **This is not an accuracy
        measure.** The diagonal is the paid-to-date, so the "error" it reports
        is the reserve itself: a model that books zero IBNR scores a perfect
        MAPE of 0 and an R² of 1. Use it only to read off bias, never to rank
        models.

    On top of the ultimate-level metrics, ``mae_ibnr`` and ``pct_error_ibnr``
    report the error on the reserve, which is the quantity actually being
    estimated and where the models genuinely differ.
    """
    latest_reported = _latest_reported_per_accident_year(triangle_df)

    if true_ultimates is not None:
        actual = true_ultimates.copy()
        actual.index = [int(x) for x in actual.index]
        scored_against = "true_ultimate"
    else:
        actual = latest_reported
        scored_against = "latest_diagonal (BIASED — see docstring)"

    rows: list[dict] = []

    for model_name, result in results.items():
        if result is None:
            continue
        ultimate = _extract_ultimate(result)
        if ultimate is None:
            continue

        common = actual.index.intersection(ultimate.index)
        if len(common) == 0:
            continue

        metrics = calculate_metrics(actual[common], ultimate[common])
        metrics["model"] = model_name
        metrics["scored_against"] = scored_against
        metrics["total_ultimate"] = float(ultimate.sum())

        ibnr_hat = ultimate - latest_reported.reindex(ultimate.index)
        metrics["total_ibnr"] = float(ibnr_hat.sum())

        if true_ultimates is not None:
            ibnr_true = actual[common] - latest_reported.reindex(common)
            metrics["mae_ibnr"] = float(np.mean(np.abs(ibnr_hat[common] - ibnr_true)))
            denom = float(ibnr_true.sum())
            metrics["pct_error_ibnr"] = (
                100.0 * (float(ibnr_hat[common].sum()) - denom) / denom if denom else np.nan
            )

        rows.append(metrics)

    return pd.DataFrame(rows)


def create_model_summary(
    triangle_df: pd.DataFrame,
    model_results: dict[str, pd.Series | pd.DataFrame | None],
) -> pd.DataFrame:
    """
    Build a wide summary DataFrame with one column per model showing ultimate estimates
    and derived IBNR reserves.

    Columns: ``accident_year``, ``latest_reported``, ``<model>``, ``<model>_IBNR``, …
    """
    accident_years = sorted({int(x) for x in triangle_df["accident_year"].unique()})
    summary = pd.DataFrame({"accident_year": accident_years})
    latest_reported = _latest_reported_per_accident_year(triangle_df)
    summary["latest_reported"] = summary["accident_year"].map(latest_reported)

    for model_name, result in model_results.items():
        if result is None:
            continue
        try:
            ultimate = _extract_ultimate(result)
            if ultimate is None:
                continue
            if all(ay in ultimate.index for ay in accident_years):
                summary[model_name] = summary["accident_year"].map(ultimate)
            elif len(ultimate) == len(accident_years):
                summary[model_name] = ultimate.values
        except Exception as exc:
            print(f"Could not extract ultimate for {model_name}: {exc}")

    for col in summary.columns:
        if col not in ("accident_year", "latest_reported") and not col.endswith("_IBNR"):
            summary[f"{col}_IBNR"] = summary[col] - summary["latest_reported"]

    return summary
