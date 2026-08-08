"""
IBNR reserving on the SSN Automotores–Casco book: traditional methods,
neural models and hybrids, in nominal and in constant currency.

Models
------
Tradicionales (Sección 2)
    Chain Ladder, Bornhuetter-Ferguson, Mack Chain Ladder (with MSEP)

Inteligencia Artificial (Sección 3)
    Bayesian Chain Ladder, MLP, GRU/DeepTriangle (Kuo 2019),
    all with MC-Dropout uncertainty

Híbridos (Sección 4)
    CANN (Wüthrich & Merz 2019) over the over-dispersed Poisson GLM,
    Credibility Hybrid (Bühlmann-style, uncertainty-weighted)

The central comparison
----------------------
Every model is fitted twice: once on the nominal triangle and once on the
triangle deflated to constant currency, with the projected reserve re-inflated
at the calendar years it will be paid in. Under Argentine inflation the two
answers differ by more than the models differ from each other, which is the
point the thesis turns on.

Accuracy is scored against the realised ultimate per accident year — available
because the triangle is synthetic — never against the latest observed diagonal.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from src.ai.bayesian_chainladder import run_bayesian_chainladder
from src.ai.cann import odp_glm_ultimate, run_cann
from src.ai.gru_triangle import run_gru_triangle
from src.ai.hybrid_models import run_credibility_hybrid
from src.ai.pytorch_mlp import run_pytorch_mlp
from src.data.inflation import DEFAULT_FUTURE_INFLATION_PCT, deflate_triangle
from src.data.load_data import load_claims_triangle, load_true_ultimates
from src.data.inflation import cpi_source_table
from src.data.sample_policies import (
    BASE_YEAR,
    VALUATION_YEAR,
    calibration_provenance,
    exposure_index,
)
from src.data.transformer import clean_siniestros_data, create_claims_triangle
from src.evaluation.comparison import compare_models_results, create_model_summary
from src.evaluation.export import write_thesis_tables
from src.evaluation.real_terms import reinflate_ultimates
from src.evaluation.sensitivity import anchor_weight_sweep, future_inflation_sweep
from src.traditional.bf_model import run_bornhuetter_ferguson
from src.traditional.chain_ladder_manual import _weighted_ldfs, cumulative_pivot_from_long
from src.traditional.chainladder_model import run_chainladder
from src.traditional.mack_model import run_mack_chainladder, run_mack_with_error
from src.visualization.comparison import (
    plot_error_by_model,
    plot_reserve_by_accident_year,
    plot_sensitivity,
    plot_uncertainty_bands,
)
from src.visualization.triangle_viz import plot_triangle

_SEP = "=" * 78
_MODELS_ORDER = [
    "Chain Ladder",
    "Bornhuetter-Ferguson",
    "Mack Chain Ladder",
    "Bayesian Chain Ladder",
    "MLP",
    "GRU",
    "CANN",
    "Credibility Hybrid",
]

# Payments after this date have not happened yet and must be predicted.
VALUATION_DATE = f"{VALUATION_YEAR}-12-31"

# Base year for the constant-currency run.
DEFLATION_BASE_YEAR = BASE_YEAR

MC_SAMPLES = 150


def _header(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def _latest_diagonal(triangle_df: pd.DataFrame) -> pd.Series:
    pivot = cumulative_pivot_from_long(triangle_df)
    return pivot.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)


def _fit_all_models(triangle_df: pd.DataFrame, exposure: pd.Series, label: str):
    """
    Fit every model on *triangle_df*.

    Returns ``(ultimates, uncertainty)`` where ``ultimates`` maps model name to
    an accident-year ultimate series, and ``uncertainty`` maps the subset of
    models that quantify it to ``(mean, std)``.
    """
    ultimates: dict[str, pd.Series] = {}
    uncertainty: dict[str, tuple[pd.Series, pd.Series]] = {}

    print(f"\n  ── Tradicionales ({label}) ──")
    for name, fn in [
        ("Chain Ladder", lambda: run_chainladder(triangle_df)),
        ("Bornhuetter-Ferguson", lambda: run_bornhuetter_ferguson(triangle_df, exposure=exposure)),
        ("Mack Chain Ladder", lambda: run_mack_chainladder(triangle_df)),
    ]:
        try:
            ultimates[name] = fn()
            print(f"     {name}")
        except Exception as exc:
            print(f"     fallo: {name}: {exc}")

    print(f"  ── Inteligencia artificial ({label}) ──")
    # Each runner is adapted to (mean, std) explicitly. Unpacking the last two
    # elements positionally is not safe: the neural runners return
    # (model, scalers…, mean, std) but the Bayesian one returns
    # (mean, std, q05, q95), so a positional `*_, mean, std` silently picks up
    # its 5th and 95th percentiles and reports the lower credible bound as the
    # point estimate.
    for name, fn in [
        (
            "Bayesian Chain Ladder",
            lambda: run_bayesian_chainladder(triangle_df)[:2],
        ),
        ("MLP", lambda: run_pytorch_mlp(triangle_df)[-2:]),
        ("GRU", lambda: run_gru_triangle(triangle_df)[-2:]),
    ]:
        try:
            mean, std = fn()
            ultimates[name] = mean
            uncertainty[name] = (mean, std)
            print(f"     {name}")
        except Exception as exc:
            print(f"     fallo: {name}: {exc}")

    print(f"  ── Híbridos ({label}) ──")
    try:
        mean, std = run_cann(triangle_df, mc_samples=MC_SAMPLES)
        ultimates["CANN"] = mean
        uncertainty["CANN"] = (mean, std)
        print("     CANN")
    except Exception as exc:
        print(f"     fallo: CANN: {exc}")

    ai_candidates = {
        name: uncertainty[name]
        for name in ("MLP", "GRU")
        if name in uncertainty
    }
    if ai_candidates and "Chain Ladder" in ultimates:
        try:
            ultimates["Credibility Hybrid"] = run_credibility_hybrid(
                ai_candidates, anchor=ultimates["Chain Ladder"], anchor_weight=0.20
            )
            print("     Credibility Hybrid")
        except Exception as exc:
            print(f"     fallo: Credibility Hybrid: {exc}")

    return ultimates, uncertainty


def _metrics_table(
    ultimates: dict[str, pd.Series],
    triangle_df: pd.DataFrame,
    true_ultimates: pd.Series | None,
) -> pd.DataFrame:
    table = compare_models_results(ultimates, triangle_df, true_ultimates)
    if table.empty:
        return table
    order = {m: i for i, m in enumerate(_MODELS_ORDER)}
    table["_order"] = table["model"].map(order).fillna(99)
    return table.sort_values("_order").drop(columns="_order")


def _print_metrics(table: pd.DataFrame, title: str) -> None:
    if table.empty:
        print(f"  (sin resultados para {title})")
        return
    cols = ["model", "mape", "r2", "total_ibnr"]
    if "pct_error_ibnr" in table.columns:
        cols.append("pct_error_ibnr")
    shown = table.copy()
    shown["total_ibnr"] = shown["total_ibnr"] / 1e6
    print(f"\n  {title}  (IBNR en M ARS nominales)")
    print(shown[cols].round(3).to_string(index=False))


def main() -> None:
    print(_SEP)
    print("IBNR — Automotores Casco (SSN) · tradicionales vs. IA · nominal vs. moneda constante")
    print(_SEP)

    print("\nCargando datos...")
    triangle_df = create_claims_triangle(
        clean_siniestros_data(load_claims_triangle()), valuation_date=VALUATION_DATE
    )

    pivot = cumulative_pivot_from_long(triangle_df)
    print(f"\nTriángulo nominal al {VALUATION_DATE} (millones ARS):")
    print((pivot / 1e6).round(1).to_string())

    exposure = exposure_index()
    true_ultimates = load_true_ultimates()
    latest = _latest_diagonal(triangle_df)

    if true_ultimates is None:
        print(
            "\nSin ultimates reales: las métricas se calculan contra la última "
            "diagonal y NO miden precisión."
        )
        real_ibnr = None
    else:
        real_ibnr = float((true_ultimates.reindex(latest.index) - latest).sum())
        print(
            f"\nReserva IBNR real (ground truth): {real_ibnr / 1e6:,.0f} M ARS "
            f"— {100 * real_ibnr / true_ultimates.sum():.1f}% del ultimate"
        )

    real_df = deflate_triangle(triangle_df, base_year=DEFLATION_BASE_YEAR, by="calendar")
    real_pivot = cumulative_pivot_from_long(real_df)
    print(
        f"\nTriángulo deflactado a pesos de {DEFLATION_BASE_YEAR}; "
        f"inflación futura supuesta {DEFAULT_FUTURE_INFLATION_PCT:.0f}% anual."
    )

    _header("MACK (1993) — ERROR DE PREDICCIÓN")
    mack_err = pd.DataFrame()
    try:
        mack_err = run_mack_with_error(triangle_df)
        shown = mack_err[mack_err["ibnr"].abs() > 0].copy()
        for col in ("latest", "ultimate", "ibnr", "se"):
            shown[col] = shown[col] / 1e6
        print(shown[["latest", "ultimate", "ibnr", "se", "cv"]].round(2).to_string())
        total = mack_err.loc["TOTAL"]
        lo, hi = total["ibnr"] - 1.645 * total["se"], total["ibnr"] + 1.645 * total["se"]
        print(f"\n  IC 90% de la reserva total: [{lo / 1e6:,.0f} ; {hi / 1e6:,.0f}] M ARS")
        if real_ibnr is not None:
            inside = lo <= real_ibnr <= hi
            print(
                f"  Reserva real {real_ibnr / 1e6:,.0f} M — "
                f"{'DENTRO' if inside else 'FUERA'} del intervalo"
            )
            if not inside:
                print(
                    "  El MSEP de Mack cubre error de proceso y de estimación bajo los\n"
                    "  supuestos del CL, no el error de modelo por no estacionariedad."
                )
    except Exception as exc:
        print(f"  fallo: {exc}")

    try:
        glm = odp_glm_ultimate(pivot)
        cl = run_chainladder(triangle_df)
        gap = float(((glm - cl).abs() / cl.abs().clip(lower=1)).max())
        print(
            f"\n  Verificación England-Verrall (2002): el GLM de Poisson sobredisperso\n"
            f"  reproduce Chain Ladder con desvío relativo máximo {gap:.2e}."
        )
    except Exception as exc:
        print(f"  fallo: Verificación GLM-ODP: {exc}")

    _header("CORRIDA A — TRIÁNGULO NOMINAL")
    nominal_ult, nominal_uq = _fit_all_models(triangle_df, exposure, "nominal")

    _header(f"CORRIDA B — MONEDA CONSTANTE (base {DEFLATION_BASE_YEAR})")
    real_ult_raw, _ = _fit_all_models(
        real_df,
        deflate_exposure(exposure, DEFLATION_BASE_YEAR),
        "deflactado",
    )
    real_ult = {
        name: reinflate_ultimates(series, real_pivot, pivot, DEFLATION_BASE_YEAR)
        for name, series in real_ult_raw.items()
    }

    _header("RESULTADOS")

    nominal_table = _metrics_table(nominal_ult, triangle_df, true_ultimates)
    real_table = _metrics_table(real_ult, triangle_df, true_ultimates)
    _print_metrics(nominal_table, "A — Nominal")
    _print_metrics(real_table, "B — Moneda constante, re-inflado")

    if real_ibnr is not None and not nominal_table.empty:
        merged = nominal_table[["model", "pct_error_ibnr"]].merge(
            real_table[["model", "pct_error_ibnr"]],
            on="model",
            suffixes=("_nominal", "_real"),
        )
        merged["mejora_pp"] = (
            merged.pct_error_ibnr_nominal.abs() - merged.pct_error_ibnr_real.abs()
        )
        print("\n  Error sobre el IBNR total: nominal vs. moneda constante (%)")
        print(merged.round(1).to_string(index=False))

    summary_df = create_model_summary(triangle_df, nominal_ult)
    if true_ultimates is not None:
        summary_df["ultimate_real"] = summary_df["accident_year"].map(true_ultimates)

    _header("SENSIBILIDAD")

    anchor_sweep = pd.DataFrame()
    ai_candidates = {
        name: nominal_uq[name] for name in ("MLP", "GRU") if name in nominal_uq
    }
    if ai_candidates and "Chain Ladder" in nominal_ult:
        try:
            anchor_sweep = anchor_weight_sweep(
                ai_candidates, nominal_ult["Chain Ladder"], latest, true_ultimates
            )
            best = anchor_sweep.loc[anchor_sweep.error_pct.abs().idxmin()]
            print("\n  Peso del ancla Chain Ladder en el híbrido:")
            print(anchor_sweep.iloc[::4].round(2).to_string(index=False))
            print(
                f"\n  Error mínimo en peso {best.anchor_weight:.2f} "
                f"({best.error_pct:+.1f}%). El 20% fijo da "
                f"{anchor_sweep.loc[anchor_sweep.anchor_weight.sub(0.20).abs().idxmin(), 'error_pct']:+.1f}%."
            )
        except Exception as exc:
            print(f"  fallo: sensibilidad del ancla: {exc}")

    inflation_sweep = pd.DataFrame()
    try:
        inflation_sweep = future_inflation_sweep(
            triangle_df,
            lambda d: run_chainladder(d),
            DEFLATION_BASE_YEAR,
            latest,
            true_ultimates,
        )
        shown = inflation_sweep.copy()
        shown["ibnr_total"] = shown["ibnr_total"] / 1e6
        print("\n  Inflación futura supuesta → reserva nominal (Chain Ladder deflactado):")
        print(shown.round(1).to_string(index=False))
        spread = shown.ibnr_total.max() / max(shown.ibnr_total.min(), 1e-9)
        print(
            f"  La reserva varía {spread:.1f}× entre el 0% y el {shown.inflacion_futura_pct.max():.0f}% "
            "anual, con el mismo patrón de desarrollo."
        )
    except Exception as exc:
        print(f"  fallo: sensibilidad a la inflación futura: {exc}")

    out_dir = Path(__file__).parent / "resultados"
    out_dir.mkdir(exist_ok=True)
    nominal_table.to_csv(out_dir / "metricas_nominal.csv", index=False)
    real_table.to_csv(out_dir / "metricas_moneda_constante.csv", index=False)
    summary_df.to_csv(out_dir / "ultimates_por_modelo.csv", index=False)
    (pivot / 1e6).round(2).to_csv(out_dir / "triangulo_nominal_MARS.csv")
    (real_pivot / 1e6).round(3).to_csv(out_dir / "triangulo_real_MARS.csv")
    if not anchor_sweep.empty:
        anchor_sweep.to_csv(out_dir / "sensibilidad_ancla.csv", index=False)
    if not inflation_sweep.empty:
        inflation_sweep.to_csv(out_dir / "sensibilidad_inflacion.csv", index=False)

    try:
        write_thesis_tables(
            out_dir / "tablas_tesis.md",
            triangle_pivot=pivot,
            real_pivot=real_pivot,
            ldfs_nominal=_weighted_ldfs(pivot.values.astype(float)),
            ldfs_real=_weighted_ldfs(real_pivot.values.astype(float)),
            calibration=calibration_provenance(),
            cpi_sources=cpi_source_table(),
            mack=mack_err,
            metrics_nominal=nominal_table,
            metrics_real=real_table,
            ultimates=summary_df,
            anchor_sweep=anchor_sweep,
            inflation_sweep=inflation_sweep,
            valuation_date=VALUATION_DATE,
            base_year=DEFLATION_BASE_YEAR,
        )
    except Exception as exc:
        print(f"  fallo: tablas de la tesis: {exc}")

    print(f"\n  Resultados → {out_dir}/")

    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    charts = [
        ("triángulo", lambda: plot_triangle(triangle_df, save_path=str(fig_dir / "triangulo.png"))),
        ("error por modelo", lambda: plot_error_by_model(
            nominal_table, real_table, save_path=str(fig_dir / "error_por_modelo.png"))),
        ("reserva por año", lambda: plot_reserve_by_accident_year(
            real_ult, latest, true_ultimates,
            save_path=str(fig_dir / "reserva_por_anio.png"))),
        ("incertidumbre", lambda: plot_uncertainty_bands(
            nominal_uq, latest, true_ultimates,
            save_path=str(fig_dir / "incertidumbre.png"))),
        ("sensibilidad", lambda: plot_sensitivity(
            anchor_sweep, inflation_sweep, save_path=str(fig_dir / "sensibilidad.png"))),
    ]
    for name, fn in charts:
        try:
            fn()
        except Exception as exc:
            print(f"  gráfico {name}: {exc}")

    print(f"  Gráficos  → {fig_dir}/\n\nListo.")


def deflate_exposure(exposure: pd.Series, base_year: int) -> pd.Series:
    """Express the a-priori exposure index in constant currency of *base_year*."""
    from src.data.inflation import price_index

    idx = price_index(base_year, list(exposure.index.astype(int)))
    return exposure / idx.reindex(exposure.index).values


if __name__ == "__main__":
    main()
