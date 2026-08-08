"""
Reader for the SSN *Anuario Estadístico — Automotores, Coberturas de Casco*.

The workbook (``ssn_2025_siniestro_casco.xlsx``, Anexo II, ejercicio 2024) holds
33 tables. Three of them carry everything the reserving exercise can take from
the official source:

``Cuadro 1``
    Vehículos expuestos a riesgo, by cover category, by year. Category **A** is
    Responsabilidad Civil only and carries **no casco cover**, so the exposure
    relevant here is ``TOTAL − A``. Reading the TOTAL row as casco exposure
    overstates it by roughly 35%.

``Cuadro 2``
    Siniestros by hecho generador (Robo Total/Parcial, Incendio Total/Parcial,
    Daño Total/Parcial), by year — the claim-type mix, which moves year to year.

``Cuadro 4``
    Frecuencia siniestral by hecho generador, by year. Its denominator is the
    casco-exposed fleet, not the TOTAL of Cuadro 1: ``Cuadro 2 / Cuadro 4``
    reproduces ``TOTAL − A`` to rounding, which is how the definition above is
    verified in :func:`verify_consistency`.

What the Anuario does **not** contain is money: no premiums, no paid claims, no
severity, and no development by accident year. Severity and the payment pattern
therefore remain modelling assumptions, and this module keeps the boundary
between "taken from the SSN" and "assumed" explicit.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Row/column layout of the workbook. The tables are formatted for printing, so
# there is no usable header row; positions are addressed directly.
_YEAR_ROW = 5
_FIRST_DATA_COL = {"Cuadro 1": 2, "Cuadro 2": 1, "Cuadro 4": 1}
_HECHO_GENERADOR = [
    "Robo Total",
    "Robo Parcial",
    "Incendio Total",
    "Incendio Parcial",
    "Daño Total",
    "Daño Parcial",
]


def anuario_path() -> Path:
    """Location of the SSN workbook (project root)."""
    return Path(__file__).resolve().parent.parent.parent / "ssn_2025_siniestro_casco.xlsx"


def _read_sheet(sheet: str) -> pd.DataFrame:
    path = anuario_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"SSN Anuario not found at {path}. Download 'Anexo II — Automotores, "
            f"Coberturas de Casco' from the SSN and place it there."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")


def _years(df: pd.DataFrame, sheet: str) -> tuple[list[int], list[int]]:
    """Return (years, column indices) for the year columns of *sheet*."""
    row = df.iloc[_YEAR_ROW if sheet != "Cuadro 4" else 4]
    years, cols = [], []
    for col in range(_FIRST_DATA_COL[sheet], df.shape[1]):
        value = row.iloc[col]
        if isinstance(value, (int, float, np.number)) and not pd.isna(value):
            year = int(value)
            if 1990 <= year <= 2100:
                years.append(year)
                cols.append(col)
    return years, cols


def _row_by_label(df: pd.DataFrame, label: str, label_cols: tuple[int, ...]) -> int:
    for row in range(df.shape[0]):
        for col in label_cols:
            if col < df.shape[1] and str(df.iat[row, col]).strip() == label:
                return row
    raise KeyError(f"Row '{label}' not found in sheet.")


def _series(sheet: str, label: str, label_cols: tuple[int, ...]) -> pd.Series:
    df = _read_sheet(sheet)
    years, cols = _years(df, sheet)
    row = _row_by_label(df, label, label_cols)
    values = [pd.to_numeric(df.iat[row, col], errors="coerce") for col in cols]
    return pd.Series(values, index=years, dtype=float).dropna()


@lru_cache(maxsize=1)
def exposure_casco() -> pd.Series:
    """
    Vehicles exposed to casco risk per year (Cuadro 1: TOTAL − categoría A).

    Category A is RC-only cover, which bears no casco risk. It is a third of the
    fleet, so including it deflates every frequency and severity derived from it.
    """
    total = _series("Cuadro 1", "TOTAL", (0, 1))
    rc_only = _series("Cuadro 1", "Responsabilidad Civil (RC)", (1,))
    return (total - rc_only.reindex(total.index)).rename("exposure_casco")


@lru_cache(maxsize=1)
def claims_by_type() -> pd.DataFrame:
    """Claim counts by hecho generador and year (Cuadro 2), rows = years."""
    data = {label: _series("Cuadro 2", label, (0,)) for label in _HECHO_GENERADOR}
    out = pd.DataFrame(data)
    out.index.name = "year"
    return out


@lru_cache(maxsize=1)
def claims_total() -> pd.Series:
    """Total casco claims per year (Cuadro 2, fila Total)."""
    return _series("Cuadro 2", "Total", (0,)).rename("claims_total")


@lru_cache(maxsize=1)
def frequency_total() -> pd.Series:
    """Total casco claim frequency per year (Cuadro 4, fila Total)."""
    return _series("Cuadro 4", "Total", (0,)).rename("frequency_total")


@lru_cache(maxsize=1)
def claim_mix() -> pd.DataFrame:
    """
    Share of each hecho generador in total claims, by year (from Cuadro 2).

    The mix is **not** constant: Robo Parcial runs from 15.4% in 2015 to 17.7%
    in 2024 and Daño Parcial moves in the opposite direction, so pinning a
    single year's participation across a twenty-year triangle bakes in a trend
    that is not there.
    """
    counts = claims_by_type()
    return counts.div(counts.sum(axis=1), axis=0)


def verify_consistency(tolerance: float = 0.01) -> pd.DataFrame:
    """
    Cross-check the three tables against each other.

    ``Cuadro 2 / Cuadro 4`` must reproduce the casco-exposed fleet derived from
    Cuadro 1, and the hecho-generador counts must add up to the Total row. Any
    row failing either check means the layout assumptions above no longer match
    the workbook and the calibration must not be trusted.
    """
    expo = exposure_casco()
    implied = (claims_total() / frequency_total()).rename("exposure_implied")
    counts = claims_by_type()

    out = pd.DataFrame(
        {
            "exposure_cuadro1": expo,
            "exposure_implied_c2_c4": implied,
            "claims_total": claims_total(),
            "claims_sum_by_type": counts.sum(axis=1),
        }
    ).dropna()
    out["exposure_gap_pct"] = 100 * (out.exposure_implied_c2_c4 / out.exposure_cuadro1 - 1)
    out["claims_gap_pct"] = 100 * (out.claims_sum_by_type / out.claims_total - 1)
    out["ok"] = (out.exposure_gap_pct.abs() < tolerance * 100) & (
        out.claims_gap_pct.abs() < tolerance * 100
    )
    return out


def calibration_table() -> pd.DataFrame:
    """
    The SSN inputs used to calibrate the synthetic triangle, one row per year.

    This is the table to reproduce in the thesis annex: every column is traceable
    to a numbered cuadro, which is what makes a synthetic triangle auditable.
    """
    mix = claim_mix()
    out = pd.DataFrame(
        {
            "vehiculos_expuestos_casco": exposure_casco(),
            "siniestros": claims_total(),
            "frecuencia": frequency_total(),
        }
    )
    for label in _HECHO_GENERADOR:
        out[f"mix_{label.lower().replace(' ', '_')}"] = mix[label]
    out.index.name = "anio"
    return out.dropna(subset=["siniestros"])
