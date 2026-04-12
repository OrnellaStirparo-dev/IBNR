# IBNR estimation — traditional · Bayesian · tabular ML · deep AI · hybrids

Load **insurance policy / claims** data from an Excel file under `data/raw/`, build a **loss development triangle**, then compare **classical reserving**, **gradient boosting**, **deep learning (PyTorch)**, and **hybrid** estimators that mix traditional and AI ultimates.

## Data

- Put your workbook in `data/raw/` (default name: `ssn_2025_siniestro_casco.xlsx`), or set `IBNR_RAW_XLSX` to a path.
- Supported shapes:
  - **Transactional payments** (recommended): `fecha_siniestro`, `fecha_pago`, `importe_pagado` (column names are normalized to lowercase with underscores).
  - **Long triangle**: `accident_year`, `development_year`, `cumulative_claims`.
  - **Pivot-style** triangles (Excel tables with year rows and development columns) — auto-detected when possible.

If the file is missing or not a valid `.xlsx` zip file, a small **demo** workbook is generated so `main.py` runs.

## Tier 1 — Traditional actuarial (`src/traditional/`)

1. **Chain Ladder** — volume-weighted development factors
2. **Mack Chain Ladder** — stochastic CL with analytic prediction error
3. **Bornhuetter–Ferguson** — credibility blend of a-priori and emerging experience
4. **Cape Cod** — experience-based expected loss ratio

## Tier 1+ — Bayesian Chain Ladder (`src/ai/bayesian_chainladder.py`)

Conjugate log-Normal posterior on development factors — no MCMC required.
Returns full predictive distribution: posterior mean, std, and 90% credible interval.
Treats the classical CL estimate as an informative prior; data updates the belief.

> Key thesis point: bridges traditional and AI — adds principled uncertainty
> without changing the chain ladder's structural assumptions.

## Tier 2 — Tabular ML (`src/ml/`)

1. **LightGBM** — histogram-based gradient boosting  
2. **CatBoost** — symmetric trees, robust to heterogeneous features  
3. **XGBoost 2.x** — regularised gradient boosting  

Shared engineered features: accident year, development year, calendar year, lagged cumulative.  
**SHAP attribution** (`src/ml/shap_explainer.py`) decomposes each prediction into feature contributions — critical for actuarial interpretability and regulatory acceptance.

## Tier 3 — Deep AI (`src/ai/`)

All neural models include **MC-dropout uncertainty quantification** (Gal & Ghahramani, 2016):
dropout stays active during 150 stochastic inference passes; the spread of resulting ultimates
forms a calibrated predictive distribution.

1. **MLP** — compact feedforward net  
2. **Deep ResNet** — residual MLP with LayerNorm + GELU  
3. **GRU** — recurrent model on the cumulative development path; no hand-crafted features  
4. **Transformer** — self-attention over development tokens `(scaled cumulative, normalised lag)`  

> Key thesis distinction vs Tier 2: sequential AI models learn *representations* of
> the development path directly — no feature engineering required.

## Tier 4 — Hybrid / ensemble (`src/ai/hybrid_models.py`)

- **Fixed convex blends** — CL + DeepResNet / GRU / Transformer / XGBoost (50/50)
- **Three-way blend** — CL + BF + DeepResNet (35/25/40)
- **Credibility Hybrid** — Bühlmann-style weighting: AI models weighted inversely by
  their MC-dropout std; a traditional anchor receives a fixed 20% credibility weight.
  This is a *data-driven* replacement for the fixed blends.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Use the venv interpreter when running scripts (`./venv/bin/python main.py`).

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
├── data/raw/
├── src/
│   ├── data/
│   ├── traditional/           # Chain Ladder, Mack, BF, Cape Cod
│   ├── ml/                    # Tabular gradient boosting + SHAP
│   ├── ai/                    # Bayesian CL, neural nets (UQ), hybrids
│   ├── evaluation/
│   └── visualization/
├── main.py
└── requirements.txt
```

## Key literature

| Reference | Relevance |
|---|---|
| Mack (1993) | Stochastic chain ladder — variance estimation |
| England & Verrall (2002) | Bayesian stochastic reserving |
| Gal & Ghahramani (2016) | MC-dropout as Bayesian approximation |
| Kuo (2019) DeepTriangle | GRU / deep learning for loss reserving |
| Gabrielli, Richman & Wüthrich (2020) | Neural networks for stochastic reserving |
| Lundberg et al. (2020) | TreeSHAP — exact Shapley values for trees |
| Wüthrich & Merz (2021) | From GLMs to neural networks and back |
