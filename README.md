# IBNR estimation — traditional vs ML

Thesis-style workflow: load **insurance policy / claims** data from an Excel file under `data/raw/`, build a **loss development triangle**, then compare **classical reserving methods** (via [chainladder](https://github.com/casact/chainladder)) with **modern tabular ML** (gradient boosting and a small PyTorch MLP).

## Data

- Put your workbook in `data/raw/` (default name: `ssn_2025_siniestro_casco.xlsx`), or set `IBNR_RAW_XLSX` to a path.
- Supported shapes:
  - **Transactional payments** (recommended): `fecha_siniestro`, `fecha_pago`, `importe_pagado` (column names are normalized to lowercase with underscores).
  - **Long triangle**: `accident_year`, `development_year`, `cumulative_claims`.
  - **Pivot-style** triangles (Excel tables with year rows and development columns) — auto-detected when possible.

If the file is missing or not a valid `.xlsx` zip file, a small **demo** workbook is generated so `main.py` runs.

## Traditional models

1. **Chain Ladder**
2. **Mack chain ladder** (same mean ultimates; Mack machinery for diagnostics)
3. **Bornhuetter–Ferguson** (chain-ladder ultimates used as `sample_weight` in chainladder)
4. **Cape Cod** (chain-ladder ultimates as `sample_weight`)

## ML / AI models

1. **LightGBM** — fast gradient boosting on tabular features  
2. **CatBoost** — gradient boosting with strong defaults for mixed data  
3. **XGBoost 2.x** — industry-standard boosted trees  
4. **PyTorch MLP** — small fully connected network on the same features  

Features are built from the triangle: accident year, development year, calendar year, and lagged cumulative severity.

ML models predict cumulative loss at the **maximum development period** in the triangle as a proxy for “ultimate” in a fixed horizon (aligned across models for comparison).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Inspect raw Excel layout:

```bash
python inspect_data.py
```

## Layout

```
IBNR/
├── data/raw/              # Place your .xlsx here
├── src/
│   ├── data/              # Load, demo data, triangle transform, chainladder I/O
│   ├── traditional/       # Chain ladder, Mack, BF, Cape Cod
│   ├── ml/                # LightGBM, CatBoost, XGBoost, PyTorch
│   ├── evaluation/        # Metrics and comparison tables
│   └── visualization/
├── main.py
└── requirements.txt
```
