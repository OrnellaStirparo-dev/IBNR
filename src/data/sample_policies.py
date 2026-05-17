"""
Synthetic claims data calibrated to SSN Automotores Casco market parameters.

Calibration sources
-------------------
- Frecuencia siniestral total: SSN Cuadro 4 (2015–2024), ~14.8%–22% por año
- Mix de siniestros: SSN Cuadro 2 (Daño Parcial 74%, Robo Total 5%, etc.)
- Severidad base 2003: estimada en ~2.500 ARS, escalada por inflación acumulada
- Patrón de desarrollo: casco automotores (liquidación rápida, 60% en dev-1)
- Heterogeneidades explícitas: caída COVID 2020, aceleración inflacionaria 2018+

Triangle dimensions: 20 accident years (2003–2022), 8 development periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# SSN-calibrated market parameters
# ---------------------------------------------------------------------------

# Frecuencia siniestral por año (fraction of exposed vehicles with a claim)
# Source: SSN Cuadro 4. Years before 2015 are interpolated from the trend.
_FREQ_BY_YEAR: dict[int, float] = {
    2003: 0.155, 2004: 0.158, 2005: 0.162, 2006: 0.165, 2007: 0.168,
    2008: 0.172, 2009: 0.175, 2010: 0.180, 2011: 0.185, 2012: 0.190,
    2013: 0.194, 2014: 0.198,
    # SSN observed 2015–2022
    2015: 0.200, 2016: 0.205, 2017: 0.212, 2018: 0.217,
    2019: 0.216, 2020: 0.148, 2021: 0.193, 2022: 0.220,
}

# Cumulative inflation multiplier on claim severity (base = 2003 = 1.0)
# Approximate: low inflation 2003-2011, moderate 2012-2017, high 2018+
_INFLATION_MULT: dict[int, float] = {
    2003: 1.0,   2004: 1.06,  2005: 1.13,  2006: 1.21,  2007: 1.30,
    2008: 1.42,  2009: 1.55,  2010: 1.72,  2011: 1.95,  2012: 2.22,
    2013: 2.60,  2014: 3.10,  2015: 3.70,  2016: 4.50,  2017: 5.50,
    2018: 8.00,  2019: 12.0,  2020: 16.0,  2021: 23.0,  2022: 36.0,
}

# Base severity in 2003 ARS for a Daño Parcial claim (log-normal parameters)
_BASE_SEVERITY_MEAN_LOG = 7.8   # e^7.8 ≈ 2,440 ARS base 2003
_BASE_SEVERITY_SIGMA_LOG = 0.9

# Claim type mix (source: SSN Cuadro 2, approximate 2015–2022 average)
_CLAIM_MIX = {
    "daño_parcial":  0.740,
    "robo_parcial":  0.177,
    "robo_total":    0.052,
    "daño_total":    0.022,
    "incendio":      0.009,
}

# Severity multiplier by claim type (relative to Daño Parcial = 1.0)
_SEVERITY_MULT = {
    "daño_parcial":  1.0,
    "robo_parcial":  1.8,
    "robo_total":    7.5,
    "daño_total":    5.0,
    "incendio":      4.0,
}

# Base development pattern: cumulative % of ultimate paid by development period
# Typical for Argentine auto casco (fast-settling, Daño Parcial dominant)
# dev 1 = year of accident, dev 2 = year+1, etc.
_DEV_PATTERN_CUM_BASE = [0.62, 0.86, 0.95, 0.98, 0.993, 0.998, 0.9993, 1.0]

# Pattern modifiers by year (shift in dev-1 cumulative %; rest redistributed)
# Positive = faster settlement, negative = slower.
# Calibrated to: Argentina high-inflation → disputes delay settlement;
# COVID 2020 → workshops closed, slower dev-1; robo-total mix shift → slower tail.
_DEV_PATTERN_SHIFT: dict[int, float] = {
    # Pre-2012: stable baseline (no shift)
    2003: 0.00, 2004: 0.00, 2005: 0.00, 2006: 0.00, 2007: 0.00,
    2008: 0.00, 2009: 0.00, 2010: 0.00, 2011: 0.00, 2012: 0.00,
    # 2013-2017: mild inflation begins to delay settlement slightly
    2013: -0.02, 2014: -0.03, 2015: -0.04, 2016: -0.04, 2017: -0.05,
    # 2018-2019: FX crisis → major disputes, significant slowdown
    2018: -0.09, 2019: -0.10,
    # 2020: COVID → workshops closed in Q2-Q3, big dev-1 slowdown
    2020: -0.15,
    # 2021-2022: hyperinflation → settlement takes longer (negotiation delays)
    2021: -0.11, 2022: -0.13,
}

# Scaled volume: number of policies per accident year
# Real market has ~11M vehicles × freq, but we scale to ~4,000–8,000 for speed.
_BASE_POLICIES = 5_000


def _sample_claim_type(rng: np.random.Generator) -> str:
    types = list(_CLAIM_MIX.keys())
    probs = list(_CLAIM_MIX.values())
    return str(rng.choice(types, p=probs))


def _incremental_from_cumulative(cum_pattern: list[float]) -> list[float]:
    inc = [cum_pattern[0]]
    for i in range(1, len(cum_pattern)):
        inc.append(cum_pattern[i] - cum_pattern[i - 1])
    return inc


def write_demo_policies_excel(
    dest: Path,
    n_policies: int | None = None,
    seed: int = 42,
) -> None:
    """
    Generate a synthetic payment-level dataset calibrated to SSN Automotores Casco.

    Parameters
    ----------
    dest        : destination Excel path
    n_policies  : total policies across all years (default: _BASE_POLICIES per year × 20 years)
    seed        : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    dest.parent.mkdir(parents=True, exist_ok=True)

    accident_years = list(range(2003, 2023))  # 20 accident years
    n_dev = len(_DEV_PATTERN_CUM_BASE)

    rows: list[dict] = []
    policy_counter = 0

    for ay in accident_years:
        freq = _FREQ_BY_YEAR[ay]
        inflation = _INFLATION_MULT[ay]

        # Build year-specific cumulative pattern by applying the shift
        shift = _DEV_PATTERN_SHIFT.get(ay, 0.0)
        cum = list(_DEV_PATTERN_CUM_BASE)
        # Shift dev-1 cumulative downward (slower settlement), redistribute to dev-2
        cum[0] = max(0.10, cum[0] + shift)
        cum[1] = min(0.99, cum[1] + max(0.0, -shift * 0.7))
        # Re-normalise tail to stay monotone and reach 1.0
        for i in range(2, len(cum) - 1):
            cum[i] = cum[i - 1] + (_DEV_PATTERN_CUM_BASE[i] - _DEV_PATTERN_CUM_BASE[i - 1])
            cum[i] = min(cum[i], 1.0)
        cum[-1] = 1.0
        inc_pattern = _incremental_from_cumulative(cum)

        # Scale policy count by frequency (COVID year gets fewer claims)
        year_policies = int(_BASE_POLICIES * freq / 0.20)  # normalise around 20%

        for _ in range(year_policies):
            policy_counter += 1
            month = int(rng.integers(1, 13))
            day = int(rng.integers(1, 29))
            fecha_siniestro = pd.Timestamp(year=ay, month=month, day=day)

            claim_type = _sample_claim_type(rng)
            sev_mult = _SEVERITY_MULT[claim_type]

            # Ultimate severity for this claim (nominal ARS of accident year)
            ultimate = (
                float(rng.lognormal(_BASE_SEVERITY_MEAN_LOG, _BASE_SEVERITY_SIGMA_LOG))
                * sev_mult
                * inflation
            )

            # Distribute ultimate across development periods
            for dev_idx, inc_frac in enumerate(inc_pattern):
                if inc_frac <= 0:
                    continue
                pay_amount = ultimate * inc_frac

                # Add noise (±15%) to each incremental payment
                pay_amount *= float(rng.uniform(0.85, 1.15))

                # Robo total takes longer to settle → shift some payments forward
                extra_months = 0
                if claim_type == "robo_total":
                    extra_months = int(rng.integers(2, 8))

                # Payment date: middle of dev year + noise
                lag_days = int(dev_idx * 365 + rng.integers(0, 365) + extra_months * 30)
                fecha_pago = fecha_siniestro + pd.Timedelta(days=lag_days)

                rows.append({
                    "poliza": f"P-{ay}-{policy_counter:06d}",
                    "fecha_siniestro": fecha_siniestro,
                    "fecha_pago": fecha_pago,
                    "importe_pagado": round(max(pay_amount, 1.0), 2),
                })

    df = pd.DataFrame(rows)
    df.to_excel(dest, index=False, engine="openpyxl")
    print(
        f"✅ Wrote SSN-calibrated synthetic dataset: {dest}\n"
        f"   {len(accident_years)} accident years (2003–2022), "
        f"{policy_counter:,} policies, {len(df):,} payment records"
    )
