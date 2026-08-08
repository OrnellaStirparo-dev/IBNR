"""
Transform raw SSN claims data into a cumulative triangle.

Two input shapes are accepted: transactional payment records
(``fecha_siniestro``, ``fecha_pago``, ``importe_pagado``), which is what the
generator produces and what a real claims extract looks like; and a long
triangle that is already aggregated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and underscore-join column names in place."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df


def _build_full_grid(
    triangle: pd.DataFrame,
    accident_years: list[int],
    max_dev_year: int,
    valuation_year: int | None = None,
) -> pd.DataFrame:
    """
    Build a complete accident-year × development-year grid.

    Cells that exist in *triangle* keep their cumulative value; all other cells
    are filled with NaN (not yet observed).

    If *valuation_year* is given, every cell whose calendar year of development
    (``accident_year + development_year - 1``) falls after it is forced to NaN.
    This is what makes the result a run-off **triangle** rather than a
    rectangle: without it, an accident year can appear more developed than the
    valuation date allows, and cells that are supposed to be predicted are
    handed to the model as observed data.
    """
    rows: list[dict] = []
    for ay in accident_years:
        ay_data = triangle[triangle["accident_year"] == ay]
        for dy in range(1, max_dev_year + 1):
            if valuation_year is not None and ay + dy - 1 > valuation_year:
                cum = np.nan
            else:
                match = ay_data[ay_data["development_year"] == dy]
                cum = float(match["cumulative_claims"].iloc[0]) if len(match) else np.nan
            rows.append({"accident_year": ay, "development_year": dy, "cumulative_claims": cum})
    return pd.DataFrame(rows)


def clean_siniestros_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and lightly clean a raw SSN siniestros DataFrame.

    - Standardises column names (lowercase, underscores).
    - Drops fully-empty columns and rows.
    - Coerces string columns with comma decimals / dash nulls.
    """
    df = _normalize_columns(df)
    df = df.dropna(axis=1, how="all")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace({",": ".", "-": None}, regex=True)

    df = df.rename(columns={"compañía": "compania", "importe_siniestros": "importe"})
    df = df.dropna(how="all")

    print(f"Cleaned data shape: {df.shape}")
    return df


def create_claims_triangle(
    df: pd.DataFrame,
    valuation_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Transform raw claims data into a cumulative triangle
    (columns: ``accident_year``, ``development_year``, ``cumulative_claims``).

    Parameters
    ----------
    df             : payment records, or a long triangle already aggregated
    valuation_date : reserving valuation date. Payments made after it are
                     dropped and the corresponding cells masked, so the result
                     is a run-off triangle observed *as of* that date. ``None``
                     returns the fully developed rectangle, which is useful to
                     inspect the data but must never be used to fit a model.
    """
    df = _normalize_columns(df)
    valuation_ts = pd.Timestamp(valuation_date) if valuation_date is not None else None

    if {"fecha_siniestro", "fecha_pago"}.issubset(df.columns):
        pay_col = next(
            (c for c in ("importe_pagado", "importe", "monto", "pago", "amount") if c in df.columns),
            None,
        )
        if pay_col is None:
            raise ValueError(
                "fecha_siniestro / fecha_pago present but no payment amount column found."
            )

        accident_ts = pd.to_datetime(df["fecha_siniestro"], errors="coerce")
        payment_ts = pd.to_datetime(df["fecha_pago"], errors="coerce")
        df["accident_year"] = accident_ts.dt.year
        df["development_year"] = payment_ts.dt.year - df["accident_year"] + 1
        df["claims"] = pd.to_numeric(df[pay_col], errors="coerce")

        # Drop payments made after the valuation date: at that date they have
        # not happened yet, so they are part of what the model must predict.
        if valuation_ts is not None:
            n_before = len(df)
            future = payment_ts > valuation_ts
            dropped_amount = float(df.loc[future, "claims"].sum())
            df = df[~future]
            if n_before - len(df):
                print(
                    f"   Censored at {valuation_ts.date()}: dropped "
                    f"{n_before - len(df):,} payment records "
                    f"({dropped_amount / 1e6:,.1f} M) dated after the valuation date"
                )

        df = df.dropna(subset=["accident_year", "development_year", "claims"])
        df["accident_year"] = df["accident_year"].astype(int)
        df["development_year"] = df["development_year"].clip(lower=1).astype(int)

        triangle = (
            df.groupby(["accident_year", "development_year"])["claims"]
            .sum()
            .reset_index()
            .sort_values(["accident_year", "development_year"])
        )
        triangle["cumulative_claims"] = triangle.groupby("accident_year")["claims"].cumsum()

        accident_years = sorted(triangle["accident_year"].unique())
        max_dev = int(triangle["development_year"].max())
        result = _build_full_grid(
            triangle,
            accident_years,
            max_dev,
            valuation_year=valuation_ts.year if valuation_ts is not None else None,
        )
        observed = int(result["cumulative_claims"].notna().sum())
        print(
            f"Built triangle from payment dates: "
            f"{len(accident_years)} accident years, {max_dev} development periods, "
            f"{observed} observed cells"
        )
        return result[["accident_year", "development_year", "cumulative_claims"]]

    if {"accident_year", "development_year", "cumulative_claims"}.issubset(df.columns):
        return df[["accident_year", "development_year", "cumulative_claims"]].copy()

    raise ValueError(
        "Unrecognised claims data: expected payment records with "
        "fecha_siniestro/fecha_pago, or a long triangle with accident_year, "
        "development_year and cumulative_claims."
    )
