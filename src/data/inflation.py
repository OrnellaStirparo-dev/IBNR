"""
Argentine CPI series and currency conversion for the reserving triangle.

Why this module exists
----------------------
An Argentine run-off triangle expressed in nominal pesos mixes two things that
move together but mean different things: how claims actually settle, and how
much the peso lost between the accident and the payment. A cumulative
development factor computed on nominal amounts absorbs both, so the observed
drift in ``f_1`` is partly a settlement effect and partly pure currency. Chain
ladder assumes the pattern is stable across accident years; under three-digit
inflation that assumption fails for a reason that has nothing to do with claims
handling.

Working in constant currency separates the two: deflate the triangle to a base
year, project the development pattern there, then re-inflate the projected
ultimates. That is the standard treatment (Taylor's separation method makes the
same split explicit) and it is the single adjustment most specific to the
Argentine case.

Source of the series
--------------------
``_CPI_ANNUAL`` holds December-to-December variation, per year, together with
the source of each figure. INDEC's national CPI was intervened between 2007 and
2015 and its published figures for those years are widely held to understate
actual inflation, so the series splices provincial and alternative indices over
that window — a standard choice in Argentine applied work, and one that has to
be stated rather than assumed.

**Validate before citing.** These are working values, not a substitute for the
official download. Point ``IBNR_CPI_CSV`` at a two-column CSV
(``anio,variacion_pct``) to override the whole series with a source you can cite
directly; :func:`cpi_source_table` prints what is in force.

For claim severity a general CPI is itself a proxy — vehicle and spare-part
prices are the right deflator and they ran above headline CPI over much of this
window. Treat the deflated results as a first-order correction.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# year: (December-to-December variation in %, source)
_CPI_ANNUAL: dict[int, tuple[float, str]] = {
    2004: (6.1, "INDEC IPC-GBA"),
    2005: (12.3, "INDEC IPC-GBA"),
    2006: (9.8, "INDEC IPC-GBA"),
    2007: (25.7, "IPC alternativo (INDEC intervenido)"),
    2008: (23.0, "IPC alternativo (INDEC intervenido)"),
    2009: (14.8, "IPC alternativo (INDEC intervenido)"),
    2010: (25.0, "IPC alternativo (INDEC intervenido)"),
    2011: (22.8, "IPC alternativo (INDEC intervenido)"),
    2012: (25.6, "IPC alternativo (INDEC intervenido)"),
    2013: (28.4, "IPC alternativo (INDEC intervenido)"),
    2014: (38.5, "IPC alternativo (INDEC intervenido)"),
    2015: (26.9, "IPC CABA (sin publicación nacional)"),
    2016: (39.4, "IPC CABA / INDEC parcial"),
    2017: (24.8, "INDEC IPC Nacional"),
    2018: (47.6, "INDEC IPC Nacional"),
    2019: (53.8, "INDEC IPC Nacional"),
    2020: (36.1, "INDEC IPC Nacional"),
    2021: (50.9, "INDEC IPC Nacional"),
    2022: (94.8, "INDEC IPC Nacional"),
    2023: (211.4, "INDEC IPC Nacional"),
    2024: (117.8, "INDEC IPC Nacional"),
}


def _override_from_csv() -> pd.Series | None:
    path = os.environ.get("IBNR_CPI_CSV", "").strip()
    if not path or not Path(path).is_file():
        return None
    df = pd.read_csv(path)
    if not {"anio", "variacion_pct"}.issubset(df.columns):
        raise ValueError("IBNR_CPI_CSV must have columns 'anio' and 'variacion_pct'.")
    return df.set_index("anio")["variacion_pct"].astype(float)


def annual_variation() -> pd.Series:
    """December-to-December CPI variation in %, per year."""
    override = _override_from_csv()
    if override is not None:
        return override.rename("variacion_pct")
    return pd.Series(
        {year: value for year, (value, _) in _CPI_ANNUAL.items()},
        name="variacion_pct",
    ).sort_index()


def cpi_source_table() -> pd.DataFrame:
    """Per-year variation and the source it comes from, for the thesis annex."""
    if _override_from_csv() is not None:
        var = annual_variation()
        return pd.DataFrame({"variacion_pct": var, "fuente": os.environ["IBNR_CPI_CSV"]})
    return pd.DataFrame(
        {
            "variacion_pct": {y: v for y, (v, _) in _CPI_ANNUAL.items()},
            "fuente": {y: s for y, (_, s) in _CPI_ANNUAL.items()},
        }
    ).sort_index()


#: Assumed annual inflation for calendar years beyond the published series.
#:
#: Reserving in constant currency makes this assumption unavoidable and explicit:
#: the reserve is paid in years that have not happened yet, so converting it back
#: to nominal requires a forecast. A nominal chain ladder does not escape the
#: assumption, it merely hides it — projecting nominal development factors
#: implicitly extrapolates whatever inflation the historical triangle contains.
#: Report results at more than one value; the reserve is highly sensitive to it.
DEFAULT_FUTURE_INFLATION_PCT = 40.0


def price_index(
    base_year: int,
    years: list[int] | None = None,
    future_pct: float | None = None,
) -> pd.Series:
    """
    Cumulative price index with ``index[base_year] == 1.0``.

    Values above 1 mean the peso of that year buys less than a peso of the base
    year, so a nominal amount of year *t* is converted to base-year currency by
    dividing by ``index[t]``.

    Calendar years past the end of the published series are extended at
    ``future_pct`` per year (default :data:`DEFAULT_FUTURE_INFLATION_PCT`).
    """
    var = annual_variation()
    future = DEFAULT_FUTURE_INFLATION_PCT if future_pct is None else float(future_pct)
    last_published = int(var.index.max())
    span = years or list(range(min(var.index.min(), base_year), max(var.index.max(), base_year) + 1))
    span = sorted(set(span) | {base_year})

    def rate(year: int) -> float:
        if year in var.index:
            return float(var[year])
        return future if year > last_published else 0.0

    lo, hi = min(span), max(span)
    idx: dict[int, float] = {base_year: 1.0}
    for year in range(base_year + 1, hi + 1):
        idx[year] = idx[year - 1] * (1.0 + rate(year) / 100.0)
    for year in range(base_year - 1, lo - 1, -1):
        idx[year] = idx[year + 1] / (1.0 + rate(year + 1) / 100.0)

    return pd.Series({y: idx[y] for y in span}, name=f"indice_base_{base_year}").sort_index()


def deflate_triangle(
    triangle_df: pd.DataFrame,
    base_year: int,
    by: str = "calendar",
) -> pd.DataFrame:
    """
    Convert a long-format triangle to constant currency of *base_year*.

    ``by='calendar'`` (default)
        Deflates each cell by the index of its **calendar year of payment**
        (``accident_year + development_year - 1``). This is the correct choice:
        a payment made in 2022 for a 2019 accident is 2022 money.

    ``by='accident'``
        Deflates the whole row by the accident year's index. Cruder, but it is
        what an accident-year on-level adjustment does, and it is useful to show
        that the conclusions do not hinge on the choice.

    Deflation is applied to the **incremental** payments, which are each money of
    a single calendar year, and the result is re-cumulated. Dividing a cumulative
    by one year's index instead would deflate payments made years apart at the
    same rate, which drives the real-terms development factors below 1 and
    projects a shrinking run-off.
    """
    out = triangle_df.copy()
    accident = out["accident_year"].astype(int)
    development = out["development_year"].astype(int)
    year = accident + development - 1 if by == "calendar" else accident
    idx = price_index(base_year, sorted(set(year.tolist()) | set(accident.tolist())))

    out = out.sort_values(["accident_year", "development_year"]).copy()
    grouped = out.groupby("accident_year")["cumulative_claims"]
    incremental = out["cumulative_claims"] - grouped.shift(1).fillna(0.0)
    out["cumulative_claims"] = (
        (incremental / year.loc[out.index].map(idx).values)
        .groupby(out["accident_year"])
        .cumsum()
    )
    # Cells never observed stay unobserved.
    out.loc[triangle_df.loc[out.index, "cumulative_claims"].isna(), "cumulative_claims"] = np.nan
    return out.loc[triangle_df.index]


def reinflate(values: pd.Series, base_year: int, target_years: pd.Series | None = None) -> pd.Series:
    """
    Convert amounts expressed in *base_year* currency back to nominal.

    ``target_years`` gives the calendar year each amount should be expressed in;
    when omitted the series index is used, which is the right default for
    accident-year ultimates.
    """
    years = target_years if target_years is not None else pd.Series(values.index, index=values.index)
    idx = price_index(base_year, sorted({int(y) for y in years}))
    return values * years.map(idx).values
