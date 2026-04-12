"""Create a valid demo Excel file under data/raw when no real workbook is present."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def write_demo_policies_excel(dest: Path, n_policies: int = 800, seed: int = 42) -> None:
    """
    Synthetic hull (casco) style policy payments: one row per payment.
    Columns match what `create_claims_triangle` recognizes via fecha_siniestro / fecha_pago.
    """
    rng = np.random.default_rng(seed)
    dest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _ in range(n_policies):
        ay = int(rng.integers(2014, 2023))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        fecha_siniestro = pd.Timestamp(year=ay, month=month, day=day)
        # 1–4 payments spread over development years
        n_pay = int(rng.integers(1, 5))
        base = float(rng.lognormal(mean=8.5, sigma=1.2))
        for k in range(n_pay):
            lag_months = int(rng.integers(0, 12 * (k + 1) + 6))
            fecha_pago = fecha_siniestro + pd.Timedelta(days=lag_months * 30)
            pay = base * float(rng.uniform(0.15, 0.55)) * (0.9 ** k)
            rows.append(
                {
                    "poliza": f"P-{ay}-{len(rows):05d}",
                    "fecha_siniestro": fecha_siniestro,
                    "fecha_pago": fecha_pago,
                    "importe_pagado": round(pay, 2),
                }
            )

    df = pd.DataFrame(rows)
    df.to_excel(dest, index=False, engine="openpyxl")
    print(f"✅ Wrote demo policy payment file: {dest} ({len(df)} rows)")
