"""
Synthetic claim payments for the Automotores–Casco run-off triangle,
calibrated against the SSN *Anuario Estadístico* and the Argentine CPI.

Why the triangle is synthetic
-----------------------------
The SSN Anuario publishes exposure, claim counts and frequency by hecho
generador and by year — but no money and no development. Claim-level payment
records are confidential. A run-off triangle therefore cannot be read off the
official source; it has to be simulated from it. What can be made auditable is
the boundary between the two, and this module keeps that boundary explicit:

Taken from the SSN Anuario (:mod:`src.data.ssn_anuario`)
    Vehicles exposed to casco risk (Cuadro 1, TOTAL − categoría A), claim counts
    and frequency (Cuadros 2 and 4), and the hecho-generador mix year by year
    (Cuadro 2). All of it real, for 2015–2024.

Taken from the CPI series (:mod:`src.data.inflation`)
    Severity trend. Nominal severity is the base severity carried forward by the
    price index, rather than an invented multiplier.

Assumed, and only assumed
    Base severity per claim type, the lognormal dispersion, and the payment
    pattern with its year-specific slowdowns. These are the modelling choices;
    the results should be read as conditional on them.

Accident years before 2015 are back-extrapolated from the SSN trend and are
flagged as such by :func:`calibration_provenance`.

The simulated book is a single insurer of ``MARKET_SHARE`` of the market, not
the market itself: reserving is done on a company, and a company-sized portfolio
carries the sampling volatility that makes the exercise meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.inflation import price_index
from src.data.ssn_anuario import claim_mix, exposure_casco, frequency_total


FIRST_ACCIDENT_YEAR = 2005
LAST_ACCIDENT_YEAR = 2024
VALUATION_YEAR = 2024
BASE_YEAR = 2005  # currency the severity assumption is expressed in

#: Share of the national casco book the simulated insurer represents.
MARKET_SHARE = 0.0025


#: Mean severity of a Daño Parcial claim in BASE_YEAR pesos. Carried forward by
#: the CPI, this lands 2024 average severity in the low millions of pesos.
_BASE_SEVERITY_DANO_PARCIAL = 1_100.0

#: Lognormal shape of individual claim severity around that mean.
_SEVERITY_SIGMA_LOG = 0.9

#: Fraction of payment-year inflation passed through to an outstanding claim.
#: A repair settled two years after the accident is invoiced at the prices of
#: the settlement date, not the accident date, so casco payments are close to
#: fully indexed. This is what makes reserving hard in Argentina: the reserve is
#: exposed to inflation that has not happened yet at the valuation date. Setting
#: it to 0 would fix each claim's cost at its accident year and remove the
#: calendar-year effect from the triangle entirely.
_PAYMENT_INDEXATION = 1.0

#: Severity of each hecho generador relative to Daño Parcial.
_SEVERITY_MULT = {
    "Daño Parcial": 1.0,
    "Robo Parcial": 1.8,
    "Incendio Parcial": 2.5,
    "Daño Total": 5.0,
    "Incendio Total": 6.0,
    "Robo Total": 7.5,
}

#: Cumulative share of the ultimate paid by development period, casco baseline.
_DEV_PATTERN_CUM_BASE = [0.62, 0.86, 0.95, 0.98, 0.993, 0.998, 0.9993, 1.0]

#: Slowdown of the first development period, by accident year (negative = slower).
#: Anchored on documented disruptions: FX crisis 2018–2019, COVID workshop
#: closures in 2020, and the inflation acceleration from 2021 on, which lengthens
#: settlement because amounts are renegotiated.
_DEV_PATTERN_SHIFT: dict[int, float] = {
    2013: -0.02, 2014: -0.03, 2015: -0.04, 2016: -0.04, 2017: -0.05,
    2018: -0.09, 2019: -0.10,
    2020: -0.15,
    2021: -0.11, 2022: -0.13, 2023: -0.14, 2024: -0.14,
}


def _accident_years() -> list[int]:
    return list(range(FIRST_ACCIDENT_YEAR, LAST_ACCIDENT_YEAR + 1))


def _back_extrapolate(series: pd.Series, years: list[int]) -> pd.Series:
    """
    Extend *series* backwards over *years* along its own log-linear trend.

    Only years before the first SSN observation are filled; observed years are
    always kept as published.
    """
    observed = series.dropna()
    x = np.array(observed.index, dtype=float)
    slope, intercept = np.polyfit(x, np.log(observed.values), 1)

    out = {}
    for year in years:
        out[year] = (
            float(observed[year])
            if year in observed.index
            else float(np.exp(intercept + slope * year))
        )
    return pd.Series(out).sort_index()


def calibrated_exposure() -> pd.Series:
    """Vehicles exposed to casco risk per accident year (SSN, extrapolated pre-2015)."""
    return _back_extrapolate(exposure_casco(), _accident_years()).rename("exposure")


def calibrated_frequency() -> pd.Series:
    """
    Casco claim frequency per accident year.

    Before the SSN window the frequency is held at the earliest published value
    rather than extrapolated: frequency is a bounded ratio with no trend to
    project, and extrapolating its log would drift without justification.
    """
    freq = frequency_total()
    earliest = float(freq.loc[freq.index.min()])
    return pd.Series(
        {y: float(freq[y]) if y in freq.index else earliest for y in _accident_years()},
        name="frequency",
    ).sort_index()


def calibrated_mix() -> pd.DataFrame:
    """Hecho-generador mix per accident year (SSN; earliest published year before 2015)."""
    mix = claim_mix()
    earliest = mix.loc[mix.index.min()]
    rows = {y: (mix.loc[y] if y in mix.index else earliest) for y in _accident_years()}
    return pd.DataFrame(rows).T.sort_index()


def severity_index() -> pd.Series:
    """Cumulative price index per accident year, base ``BASE_YEAR``."""
    return price_index(BASE_YEAR, _accident_years()).reindex(_accident_years())


def exposure_index() -> pd.Series:
    """
    On-level exposure proxy per accident year, for a-priori-based methods.

    ``vehicles exposed × frequency × severity index`` — the expected claim cost
    of the year in that year's own currency, which is what Bornhuetter-Ferguson
    and Cape Cod need their prior to be denominated in. A prior held flat across
    accident years is what drives BF to inadmissible negative reserves once
    severity has trended by two orders of magnitude.
    """
    return (
        calibrated_exposure() * calibrated_frequency() * severity_index()
    ).rename("exposure")


def calibration_provenance() -> pd.DataFrame:
    """Per-year calibration inputs and whether each comes from the SSN or is extrapolated."""
    ssn_years = set(exposure_casco().index)
    out = pd.DataFrame(
        {
            "vehiculos_expuestos": calibrated_exposure(),
            "frecuencia": calibrated_frequency(),
            "indice_precios": severity_index(),
            "siniestros_mercado": calibrated_exposure() * calibrated_frequency(),
        }
    )
    out["fuente"] = [
        "SSN Cuadros 1/2/4" if y in ssn_years else "extrapolado"
        for y in out.index
    ]
    out.index.name = "anio"
    return out


def _shifted_incremental_pattern(shift: float) -> list[float]:
    """
    Incremental payment pattern with a slowdown of *shift* applied to dev-1.

    A negative ``shift`` moves that fraction of the ultimate out of the first
    development period and redistributes it over dev-2…dev-n **in proportion to
    the base incremental weights**, so the payout is genuinely pushed into the
    tail while the pattern still sums to 1 and stays monotone.

    Redistributing the displaced mass onto dev-2 alone instead has the opposite
    effect of the one intended: the cumulative pattern reaches 1.0 within two or
    three periods and the tail collapses, so the accident years meant to settle
    *slowest* end up fully developed first and carry almost no reserve.
    """
    base_inc = [_DEV_PATTERN_CUM_BASE[0]] + [
        _DEV_PATTERN_CUM_BASE[i] - _DEV_PATTERN_CUM_BASE[i - 1]
        for i in range(1, len(_DEV_PATTERN_CUM_BASE))
    ]
    inc = list(base_inc)

    moved = -shift
    if moved <= 0:
        return inc

    moved = min(moved, inc[0] - 0.05)
    tail_weight = sum(base_inc[1:])
    inc[0] -= moved
    for i in range(1, len(inc)):
        inc[i] += moved * base_inc[i] / tail_weight
    return inc


def development_patterns() -> pd.DataFrame:
    """Cumulative payment pattern actually used, per accident year."""
    rows = {
        y: np.cumsum(_shifted_incremental_pattern(_DEV_PATTERN_SHIFT.get(y, 0.0)))
        for y in _accident_years()
    }
    out = pd.DataFrame(rows).T
    out.columns = [f"dev_{j + 1}" for j in out.columns]
    out.index.name = "anio"
    return out


def write_demo_policies_excel(
    dest: Path,
    n_policies: int | None = None,
    seed: int = 42,
) -> None:
    """
    Generate the synthetic payment-level dataset.

    The generator emits the **complete, uncensored** payment history of every
    claim, including payments falling after the valuation date. Censoring is the
    responsibility of :func:`src.data.transformer.create_claims_triangle`, which
    is where a real reserving exercise applies it — keeping the two apart is what
    makes the ground truth below meaningful.

    Sheets written:

    ``pagos``
        One row per incremental payment.
    ``ultimates_reales``
        True ultimate per accident year over the entire payment history. Only
        knowable because the data is synthetic, and the only valid target for
        accuracy metrics — never the latest observed diagonal, which is the
        paid-to-date and would reward a model for booking no reserve at all.
    ``calibracion``
        Per-year SSN inputs and their provenance, for the thesis annex.
    """
    rng = np.random.default_rng(seed)
    dest.parent.mkdir(parents=True, exist_ok=True)

    years = _accident_years()
    exposure = calibrated_exposure()
    frequency = calibrated_frequency()
    mix = calibrated_mix()
    sev_index = severity_index()
    claim_types = list(mix.columns)

    # Payment years run past the last accident year by the length of the pattern.
    n_dev = len(_DEV_PATTERN_CUM_BASE)
    full_index = price_index(
        BASE_YEAR, list(range(FIRST_ACCIDENT_YEAR, LAST_ACCIDENT_YEAR + n_dev + 2))
    )

    rows: list[dict] = []
    claim_counter = 0

    for ay in years:
        market_claims = float(exposure[ay]) * float(frequency[ay])
        n_claims = max(int(round(market_claims * MARKET_SHARE)), 1)

        probs = mix.loc[ay].values.astype(float)
        probs = probs / probs.sum()
        inc_pattern = _shifted_incremental_pattern(_DEV_PATTERN_SHIFT.get(ay, 0.0))
        severity_scale = _BASE_SEVERITY_DANO_PARCIAL * float(sev_index[ay])

        types = rng.choice(len(claim_types), size=n_claims, p=probs)
        months = rng.integers(1, 13, size=n_claims)
        days = rng.integers(1, 29, size=n_claims)

        for k in range(n_claims):
            claim_counter += 1
            claim_type = claim_types[types[k]]
            fecha_siniestro = pd.Timestamp(year=ay, month=int(months[k]), day=int(days[k]))

            ultimate = (
                float(rng.lognormal(0.0, _SEVERITY_SIGMA_LOG))
                * severity_scale
                * _SEVERITY_MULT[claim_type]
            )

            # Total losses and thefts are settled through a longer legal process.
            extra_months = int(rng.integers(2, 8)) if claim_type.endswith("Total") else 0

            for dev_idx, inc_frac in enumerate(inc_pattern):
                if inc_frac <= 0:
                    continue
                lag_days = int(dev_idx * 365 + rng.integers(0, 365) + extra_months * 30)
                fecha_pago = fecha_siniestro + pd.Timedelta(days=lag_days)

                # The payment is settled at the prices of the year it is made.
                indexation = (
                    float(full_index[fecha_pago.year]) / float(sev_index[ay])
                ) ** _PAYMENT_INDEXATION
                pay_amount = (
                    ultimate * inc_frac * indexation * float(rng.uniform(0.85, 1.15))
                )

                rows.append(
                    {
                        "poliza": f"P-{ay}-{claim_counter:06d}",
                        "fecha_siniestro": fecha_siniestro,
                        "fecha_pago": fecha_pago,
                        "hecho_generador": claim_type,
                        "importe_pagado": round(max(pay_amount, 1.0), 2),
                    }
                )

    df = pd.DataFrame(rows)
    true_ultimates = (
        df.assign(accident_year=df["fecha_siniestro"].dt.year)
        .groupby("accident_year")["importe_pagado"]
        .sum()
        .rename("true_ultimate")
        .reset_index()
    )

    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="pagos", index=False)
        true_ultimates.to_excel(writer, sheet_name="ultimates_reales", index=False)
        calibration_provenance().reset_index().to_excel(
            writer, sheet_name="calibracion", index=False
        )

    print(
        f"Dataset sintético calibrado a la SSN: {dest}\n"
        f"   {len(years)} años de accidente ({years[0]}–{years[-1]}), "
        f"{claim_counter:,} siniestros, {len(df):,} pagos\n"
        f"   Cartera simulada: {MARKET_SHARE:.2%} del mercado casco\n"
        f"   Ground truth: {true_ultimates['true_ultimate'].sum() / 1e6:,.1f} M ARS nominales"
    )
