"""
Transform raw SSN claims data into a cumulative triangle.

Supported input shapes
----------------------
1. **Transactional** (primary): ``fecha_siniestro``, ``fecha_pago``, ``importe_pagado``
2. **Long triangle**: ``accident_year``, ``development_year``, ``cumulative_claims``
3. **Pivot table**: rows = accident years, columns = development periods (auto-detected)
4. **Generic**: auto-detects accident-year / development-year / claims columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

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
) -> pd.DataFrame:
    """
    Build a complete accident-year × development-year grid.

    Cells that exist in *triangle* keep their cumulative value; all other cells
    in the upper-right quadrant are filled with NaN (not yet observed).
    """
    rows: list[dict] = []
    for ay in accident_years:
        ay_data = triangle[triangle["accident_year"] == ay]
        for dy in range(1, max_dev_year + 1):
            match = ay_data[ay_data["development_year"] == dy]
            cum = float(match["cumulative_claims"].iloc[0]) if len(match) else np.nan
            rows.append({"accident_year": ay, "development_year": dy, "cumulative_claims": cum})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

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

    print(f"🧹 Cleaned data shape: {df.shape}")
    return df


def create_claims_triangle(
    df: pd.DataFrame,
    accident_year_col: str | None = None,
    development_year_col: str | None = None,
    claims_col: str | None = None,
    date_col: str | None = None,
    accident_date_col: str | None = None,
) -> pd.DataFrame:
    """
    Transform raw claims data into a cumulative triangle
    (columns: ``accident_year``, ``development_year``, ``cumulative_claims``).

    Detection priority
    ------------------
    1. Transactional: ``fecha_siniestro`` + ``fecha_pago`` + payment amount.
    2. Already-triangulated: ``accident_year`` + ``development_year`` + ``cumulative_claims``.
    3. Pivot table: rows = accident years, columns = development periods.
    4. Generic long format with auto-detected column names.

    Parameters
    ----------
    df                  : raw claims DataFrame (any of the shapes above)
    accident_year_col   : override accident-year column name
    development_year_col: override development-year column name
    claims_col          : override claims-amount column name
    date_col            : payment / development date column (if date-based)
    accident_date_col   : accident date column (if separate from date_col)
    """
    df = _normalize_columns(df)

    # ------------------------------------------------------------------ #
    # Path 1 — transactional payment records                              #
    # ------------------------------------------------------------------ #
    if {"fecha_siniestro", "fecha_pago"}.issubset(df.columns):
        pay_col = next(
            (c for c in ("importe_pagado", "importe", "monto", "pago", "amount") if c in df.columns),
            None,
        )
        if pay_col is None:
            raise ValueError(
                "fecha_siniestro / fecha_pago present but no payment amount column found."
            )

        df["accident_year"] = pd.to_datetime(df["fecha_siniestro"], errors="coerce").dt.year
        df["development_year"] = (
            pd.to_datetime(df["fecha_pago"], errors="coerce").dt.year - df["accident_year"] + 1
        )
        df["claims"] = pd.to_numeric(df[pay_col], errors="coerce")
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
        result = _build_full_grid(triangle, accident_years, max_dev)
        print(
            f"✅ Built triangle from payment dates: "
            f"{len(accident_years)} accident years, {max_dev} development periods"
        )
        return result[["accident_year", "development_year", "cumulative_claims"]]

    # ------------------------------------------------------------------ #
    # Path 2 — already in triangle format                                 #
    # ------------------------------------------------------------------ #
    if {"accident_year", "development_year", "cumulative_claims"}.issubset(df.columns):
        print("✅ Data already in triangle format")
        return df[["accident_year", "development_year", "cumulative_claims"]].copy()

    # ------------------------------------------------------------------ #
    # Path 3 — pivot table (rows = accident years, cols = dev periods)    #
    # ------------------------------------------------------------------ #
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if len(numeric_cols) >= 1 and len(non_numeric_cols) <= 2:
        print("🔍 Detected pivot table format — attempting to convert...")
        try:
            if non_numeric_cols:
                acc_col = non_numeric_cols[0]
                df["accident_year"] = pd.to_numeric(df[acc_col], errors="coerce")
                if df["accident_year"].isna().mean() > 0.5:
                    extracted = df[acc_col].astype(str).str.extract(r"(\d{4})")[0]
                    df["accident_year"] = pd.to_numeric(extracted, errors="coerce")
                print(f"   Using column '{acc_col}' as accident years")
            else:
                raise ValueError("Cannot auto-detect accident years in pivot format.")

            if df["accident_year"].isna().all():
                raise ValueError("Could not extract accident years.")

            triangle_list: list[dict] = []
            for idx, row in df.iterrows():
                ay = df.at[idx, "accident_year"]
                if pd.isna(ay):
                    continue
                for col_idx, col in enumerate(numeric_cols):
                    dev_match = pd.Series([str(col)]).str.extract(r"(\d+)")[0]
                    dev_year = (
                        int(dev_match.iloc[0])
                        if len(dev_match) and not dev_match.isna().iloc[0]
                        else col_idx + 1
                    )
                    value = row[col]
                    if pd.notna(value):
                        triangle_list.append({
                            "accident_year": int(ay),
                            "development_year": dev_year,
                            "cumulative_claims": float(value),
                        })

            if triangle_list:
                result_df = (
                    pd.DataFrame(triangle_list)
                    .sort_values(["accident_year", "development_year"])
                )
                print(f"✅ Converted pivot table: {len(result_df)} records")
                return result_df[["accident_year", "development_year", "cumulative_claims"]]
        except Exception as exc:
            print(f"⚠️  Pivot table conversion failed: {exc}")

    # ------------------------------------------------------------------ #
    # Path 4 — generic long format with column auto-detection             #
    # ------------------------------------------------------------------ #
    if accident_year_col is None:
        for candidate in ("año_siniestro", "accident_year", "ay", "año", "year",
                          "periodo_siniestro", "fecha_siniestro"):
            if candidate in df.columns:
                accident_year_col = candidate
                break

    if development_year_col is None:
        for candidate in ("development_year", "dev_year", "dy", "periodo_desarrollo",
                          "año_desarrollo", "lag", "development_period"):
            if candidate in df.columns:
                development_year_col = candidate
                break

    if claims_col is None:
        for candidate in ("importe", "monto", "claims", "cumulative_claims",
                          "siniestros", "valor", "amount"):
            if candidate in df.columns:
                claims_col = candidate
                break

    # Date-based accident year extraction
    if date_col or accident_date_col:
        src = accident_date_col if accident_date_col and accident_date_col in df.columns else date_col
        if src and src in df.columns:
            df["accident_year"] = pd.to_datetime(df[src], errors="coerce").dt.year
        if date_col in df.columns and accident_date_col in df.columns:
            df["development_year"] = (
                pd.to_datetime(df[date_col], errors="coerce").dt.year
                - pd.to_datetime(df[accident_date_col], errors="coerce").dt.year
                + 1
            )

    if accident_year_col and accident_year_col in df.columns:
        df["accident_year"] = pd.to_numeric(df[accident_year_col], errors="coerce")
    elif "accident_year" not in df.columns:
        raise ValueError(
            "Could not identify accident year column. Specify accident_year_col."
        )

    if development_year_col and development_year_col in df.columns:
        df["development_year"] = pd.to_numeric(df[development_year_col], errors="coerce")
    elif "development_year" not in df.columns:
        print("⚠️  No development year column found. Defaulting to development year 1.")
        df["development_year"] = 1

    if claims_col and claims_col in df.columns:
        df["claims"] = pd.to_numeric(df[claims_col], errors="coerce")
    elif "cumulative_claims" in df.columns:
        df["claims"] = df["cumulative_claims"]
    else:
        all_numeric = df.select_dtypes(include=np.number).columns
        if len(all_numeric) == 0:
            raise ValueError("Could not identify claims column. Specify claims_col.")
        df["claims"] = df[all_numeric[0]]
        print(f"⚠️  Using '{all_numeric[0]}' as claims column")

    df = df.dropna(subset=["accident_year", "development_year", "claims"])
    triangle = (
        df.groupby(["accident_year", "development_year"])["claims"]
        .sum()
        .reset_index()
        .sort_values(["accident_year", "development_year"])
    )
    triangle["cumulative_claims"] = triangle.groupby("accident_year")["claims"].cumsum()

    accident_years = sorted(triangle["accident_year"].unique())
    max_dev = int(triangle["development_year"].max())
    result = _build_full_grid(triangle, accident_years, max_dev)

    print(
        f"✅ Created triangle: {len(accident_years)} accident years, {max_dev} development periods "
        f"({min(accident_years)}–{max(accident_years)})"
    )
    return result[["accident_year", "development_year", "cumulative_claims"]]
