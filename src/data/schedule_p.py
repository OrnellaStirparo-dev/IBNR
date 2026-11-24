"""
CAS Loss Reserving Database (NAIC Schedule P), Meyers & Shi (2011).

Real aggregate triangles of US insurers, accident years 1988-1997, ten
development years, **with the lower triangle actually observed** (development
followed up to calendar year 2006). That is what the synthetic SSN triangle
cannot offer: the realised ultimate against which every estimator is measured
is the one the company really paid, not one produced by our own generator.

Source of the file in ``data/schedule_p``: the ``schedule_p`` dataset of the R
package ``insurance`` (Kuo, github.com/kasaai/insurance), a tidy copy of the
CAS files. The company subset (50 groups per line, four lines) is the one
selected by Meyers (2015) and used by Kuo (2019), so results are comparable
with the literature reviewed in Section 3.1.

Amounts are in thousands of USD. US inflation over the window was low and
stable, so the triangles are used as published: the homogeneous unit of
account is a premise here, not something to be tested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "schedule_p"
FIRST_AY, N_DEV = 1988, 10
VALUATION_YEAR = 1997

LINES = {
    "private_passenger_auto": "Automotores particulares",
    "commercial_auto": "Automotores comerciales",
    "workers_compensation": "Riesgos del trabajo",
    "other_liability": "Responsabilidad civil",
}


def load_schedule_p(meyers_subset: bool = True) -> pd.DataFrame:
    """Long table, one row per (line, company, accident year, development lag)."""
    d = pd.read_csv(DATA_DIR / "schedule_p.csv.gz")
    d["group_code"] = d["group_code"].astype(str)
    if meyers_subset:
        codes = pd.read_csv(DATA_DIR / "meyers_group_codes.csv", dtype=str)
        d = d.merge(codes, on=["lob", "group_code"], how="inner")
    return d


def full_square(d: pd.DataFrame, lob: str, code: str, value: str = "cumulative_paid_loss") -> pd.DataFrame:
    """10 × 10 square (observed + realised future) for one company and line."""
    g = d[(d["lob"] == lob) & (d["group_code"] == code)]
    sq = g.pivot_table(index="accident_year", columns="development_lag", values=value, aggfunc="first")
    return sq.reindex(index=range(FIRST_AY, FIRST_AY + N_DEV), columns=range(1, N_DEV + 1))


def upper_triangle(square: pd.DataFrame) -> pd.DataFrame:
    """Mask every cell paid after the valuation date (calendar year > 1997)."""
    out = square.copy().astype(float)
    for ay in out.index:
        for j in out.columns:
            if ay + j - 1 > VALUATION_YEAR:
                out.loc[ay, j] = np.nan
    return out


def to_long(pivot: pd.DataFrame) -> pd.DataFrame:
    """Pivot → the long format every runner in ``src`` expects."""
    long = pivot.stack(future_stack=True).rename("cumulative_claims").reset_index()
    long.columns = ["accident_year", "development_year", "cumulative_claims"]
    long["accident_year"] = long["accident_year"].astype(int)
    long["development_year"] = long["development_year"].astype(int)
    return long


def premium(d: pd.DataFrame, lob: str, code: str) -> pd.Series:
    """Net earned premium per accident year."""
    g = d[(d["lob"] == lob) & (d["group_code"] == code)]
    return g.groupby("accident_year")["earned_premium_net"].first().reindex(
        range(FIRST_AY, FIRST_AY + N_DEV)
    ).astype(float)


def usable(square: pd.DataFrame, prem: pd.Series) -> bool:
    """
    Minimal data-quality filter: complete square, positive premium every year,
    positive first paid cell every year (a multiplicative method cannot start
    from zero) and a positive realised reserve (it is the denominator of the
    error). Small decreases of the paid cumulative are kept: they are salvage
    and subrogation recoveries, frequent in auto, and part of real experience.
    """
    v = square.values.astype(float)
    if not np.all(np.isfinite(v)) or np.any(prem.values <= 0) or np.any(v[:, 0] <= 0):
        return False
    up = upper_triangle(square).values
    paid = np.array([r[np.isfinite(r)][-1] for r in up])
    return bool(np.sum(v[:, -1] - paid) > 0)
