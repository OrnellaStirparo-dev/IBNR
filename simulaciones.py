"""
Robustez del experimento: repite el ejercicio sobre múltiples realizaciones
del proceso generador (comentario del tutor sobre "una sola realización").

Para cada semilla s:
  1. Genera un dataset sintético nuevo (misma calibración SSN, semilla s).
  2. Ajusta los ocho estimadores en pesos corrientes y en moneda constante.
  3. Mide, contra el ultimate realizado, el error relativo sobre la reserva
     total (E), el WMAPE, el MAE y el RMSE por año de ocurrencia.

Opcionalmente (--gen-future X), el generador usa una inflación futura X
distinta del supuesto del modelo (DEFAULT_FUTURE_INFLATION_PCT). Por defecto
ambas coinciden (40 %), lo que equivale a que el modelo en moneda constante
conozca la inflación futura exacta.

Uso:
    python simulaciones.py --seeds 30 --jobs 4
    python simulaciones.py --seeds 15 --jobs 4 --gen-future 60
    python simulaciones.py --seeds 15 --jobs 4 --gen-future 25
    python simulaciones.py --seeds 30 --sin-redes      # sin PyTorch

Salidas en resultados/: simulaciones_detalle[_genXX].csv y
simulaciones_resumen[_genXX].csv.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import src.data.inflation as inflation_module
try:  # ubicación actual del Bayesian Chain Ladder (métodos tradicionales)
    from src.traditional.bayesian_chainladder import run_bayesian_chainladder
except ImportError:  # versiones anteriores del repo
    from src.ai.bayesian_chainladder import run_bayesian_chainladder
from src.data.inflation import deflate_triangle, price_index
from src.data.sample_policies import BASE_YEAR, VALUATION_YEAR
from src.traditional.bf_model import run_bornhuetter_ferguson
from src.traditional.chainladder_model import run_chainladder

VALUATION_DATE = f"{VALUATION_YEAR}-12-31"
DEFLATION_BASE_YEAR = BASE_YEAR

try:  # Los modelos neuronales requieren PyTorch.
    import torch  # noqa: F401
    from main import _fit_all_models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def deflate_exposure(exposure: pd.Series, base_year: int) -> pd.Series:
    idx = price_index(base_year, list(exposure.index.astype(int)))
    return exposure / idx.reindex(exposure.index).values


def _fit(triangle_df: pd.DataFrame, exposure: pd.Series, label: str, neural: bool):
    """Los ocho estimadores (con PyTorch) o los cuatro no neuronales (sin él)."""
    if neural:
        return _fit_all_models(triangle_df, exposure, label)
    ult = {
        "Chain Ladder": run_chainladder(triangle_df),
        "Bornhuetter-Ferguson": run_bornhuetter_ferguson(triangle_df, exposure=exposure),
        "Bayesian Chain Ladder": run_bayesian_chainladder(triangle_df)[0],
    }
    return ult, {}
from src.data.load_data import load_siniestros_data
from src.data.sample_policies import exposure_index, write_demo_policies_excel
from src.data.transformer import clean_siniestros_data, create_claims_triangle
from src.evaluation.real_terms import reinflate_ultimates
from src.traditional.chain_ladder_manual import cumulative_pivot_from_long


def _latest(pivot: pd.DataFrame) -> pd.Series:
    s = pivot.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)
    s.index = s.index.astype(int)
    return s


def _score(ult: pd.Series, true_ult: pd.Series, latest: pd.Series) -> dict[str, float]:
    years = latest.index
    r_hat = ult.reindex(years).astype(float) - latest
    r = true_ult.reindex(years).astype(float) - latest
    res = r_hat - r
    return {
        "E_pct": 100 * (r_hat.sum() - r.sum()) / r.sum(),
        "WMAPE_pct": 100 * res.abs().sum() / r.sum(),
        "MAE_M": res.abs().mean() / 1e6,
        "RMSE_M": np.sqrt((res**2).mean()) / 1e6,
        "reserva_real_M": r.sum() / 1e6,
    }


def run_one(seed: int, gen_future: float | None, neural: bool) -> list[dict]:
    model_future = inflation_module.DEFAULT_FUTURE_INFLATION_PCT
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sim.xlsx"
        if gen_future is not None:
            inflation_module.DEFAULT_FUTURE_INFLATION_PCT = float(gen_future)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                write_demo_policies_excel(path, seed=seed)
        finally:
            inflation_module.DEFAULT_FUTURE_INFLATION_PCT = model_future

        with contextlib.redirect_stdout(io.StringIO()):
            raw = load_siniestros_data(str(path))
        true_df = pd.read_excel(path, sheet_name="ultimates_reales", engine="openpyxl")
    true_ult = true_df.set_index("accident_year")["true_ultimate"]
    true_ult.index = true_ult.index.astype(int)

    with contextlib.redirect_stdout(io.StringIO()):
        tri = create_claims_triangle(clean_siniestros_data(raw), valuation_date=VALUATION_DATE)
        pivot = cumulative_pivot_from_long(tri)
        latest = _latest(pivot)
        exposure = exposure_index()
        real_df = deflate_triangle(tri, base_year=DEFLATION_BASE_YEAR, by="calendar")
        real_pivot = cumulative_pivot_from_long(real_df)

        nom, _ = _fit(tri, exposure, "nominal", neural)
        real_raw, _ = _fit(
            real_df, deflate_exposure(exposure, DEFLATION_BASE_YEAR), "deflactado", neural
        )
    real = {
        k: reinflate_ultimates(v, real_pivot, pivot, DEFLATION_BASE_YEAR)
        for k, v in real_raw.items()
    }

    rows = []
    for unidad, res in (("corriente", nom), ("constante", real)):
        for model, ult in res.items():
            s = pd.Series(ult).copy()
            s.index = s.index.astype(int)
            rows.append({"seed": seed, "unidad": unidad, "modelo": model, **_score(s, true_ult, latest)})
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    detail = detail.copy()
    detail["absE"] = detail["E_pct"].abs()
    # Ranking por realización (Mack se excluye: misma estimación que CL).
    ranked = detail[detail.modelo != "Mack Chain Ladder"].copy()
    ranked["rank_absE"] = ranked.groupby(["seed", "unidad"])["absE"].rank(method="min")
    ranked["rank_WMAPE"] = ranked.groupby(["seed", "unidad"])["WMAPE_pct"].rank(method="min")

    g = ranked.groupby(["unidad", "modelo"])
    out = pd.DataFrame(
        {
            "E_media": g.E_pct.mean(),
            "E_p05": g.E_pct.quantile(0.05),
            "E_p95": g.E_pct.quantile(0.95),
            "absE_media": g.absE.mean(),
            "WMAPE_media": g.WMAPE_pct.mean(),
            "WMAPE_desvio": g.WMAPE_pct.std(),
            "MAE_media_M": g.MAE_M.mean(),
            "RMSE_media_M": g.RMSE_M.mean(),
            "rank_absE_medio": g.rank_absE.mean(),
            "pct_primero_absE": g.rank_absE.apply(lambda r: 100 * (r == 1).mean()),
            "pct_primero_WMAPE": g.rank_WMAPE.apply(lambda r: 100 * (r == 1).mean()),
        }
    ).reset_index()
    return out.sort_values(["unidad", "absE_media"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--gen-future", type=float, default=None,
                    help="inflación futura (%% anual) del generador; por defecto = supuesto del modelo")
    ap.add_argument("--jobs", type=int, default=1,
                    help="semillas en paralelo (procesos); p. ej. 4 en una Mac M1/M2")
    ap.add_argument("--sin-redes", action="store_true",
                    help="solo CL, Mack, BF y Bayesian CL (no requiere PyTorch)")
    args = ap.parse_args()
    neural = HAS_TORCH and not args.sin_redes
    if not neural:
        print("Corriendo sin modelos neuronales (CL, BF, Bayesian CL).")

    suffix = ("" if args.gen_future is None else f"_gen{int(args.gen_future)}") + (
        "" if neural else "_sin_redes"
    )
    out_dir = Path(__file__).parent / "resultados"
    out_dir.mkdir(exist_ok=True)
    detail_path = out_dir / f"simulaciones_detalle{suffix}.csv"

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    rows: list[dict] = []
    t0 = time.time()

    def _done(seed: int, k: int) -> None:
        pd.DataFrame(rows).to_csv(detail_path, index=False)
        print(f"semilla {seed} lista ({k}/{args.seeds}, {time.time() - t0:.0f}s)", flush=True)

    if args.jobs <= 1:
        for k, seed in enumerate(seeds, 1):
            rows += run_one(seed, args.gen_future, neural)
            _done(seed, k)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_one, s, args.gen_future, neural): s for s in seeds}
            for k, fut in enumerate(as_completed(futures), 1):
                rows += fut.result()
                _done(futures[fut], k)

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    summary.to_csv(out_dir / f"simulaciones_resumen{suffix}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
