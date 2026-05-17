"""
IBNR reserving: comparación de métodos tradicionales vs. Inteligencia Artificial.

Modelos incluidos
-----------------
Tradicionales (Sección 2 de la tesis):
    Chain Ladder, Bornhuetter-Ferguson, Mack Chain Ladder

Inteligencia Artificial (Sección 3):
    GRU / DeepTriangle (Kuo 2019)
    Transformer (Vaswani et al. 2017 / Richman & Wüthrich 2026)
    Ambos con cuantificación de incertidumbre via MC-Dropout (Gal & Ghahramani 2016).

Híbrido (Sección 4):
    Credibility Hybrid — Bühlmann-style ensemble de GRU y Transformer
    ponderado por inversa de desviación MC-Dropout, con Chain Ladder como ancla.

Datos:
    Triángulo sintético calibrado a parámetros reales de la SSN
    (frecuencia, inflación, mix de siniestros, período 2003–2022).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd

from src.ai.gru_triangle import run_gru_triangle
from src.ai.hybrid_models import run_credibility_hybrid, run_hybrid_cl_transformer
from src.ai.transformer_triangle import run_transformer_triangle
from src.data.load_data import load_claims_triangle
from src.data.transformer import clean_siniestros_data, create_claims_triangle
from src.evaluation.comparison import compare_models_results, create_model_summary
from src.traditional.bf_model import run_bornhuetter_ferguson
from src.traditional.chainladder_model import run_chainladder
from src.traditional.mack_model import run_mack_chainladder
from src.visualization.comparison import (
    plot_ibnr_comparison,
    plot_model_comparison_metrics,
    plot_ultimate_claims_comparison,
    plot_uncertainty_bands,
)
from src.visualization.triangle_viz import plot_triangle

_SEP = "=" * 70
_MODELS_ORDER = [
    "Chain Ladder",
    "Bornhuetter-Ferguson",
    "Mack Chain Ladder",
    "GRU",
    "Transformer",
    "Credibility Hybrid",
]


def _header(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def main() -> None:
    print(_SEP)
    print("IBNR  —  Tradicionales · GRU · Transformer · Híbrido (Credibilidad)")
    print(_SEP)

    # ------------------------------------------------------------------ #
    # Datos                                                                #
    # ------------------------------------------------------------------ #
    print("\nCargando datos...")
    raw_df = load_claims_triangle()
    cleaned_df = clean_siniestros_data(raw_df)

    print("\nConstruyendo triángulo acumulado...")
    triangle_df = create_claims_triangle(cleaned_df)

    pivot = triangle_df.pivot(
        index="accident_year", columns="development_year", values="cumulative_claims"
    )
    print("\nTriángulo (millones ARS):")
    print((pivot / 1e6).round(2).to_string())

    model_results: dict[str, pd.Series | None] = {}
    uq_results: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}

    # ------------------------------------------------------------------ #
    # TRADICIONALES                                                        #
    # ------------------------------------------------------------------ #
    _header("TIER 1 — MÉTODOS TRADICIONALES")

    for name, fn in [
        ("Chain Ladder",         lambda: run_chainladder(triangle_df)),
        ("Bornhuetter-Ferguson", lambda: run_bornhuetter_ferguson(triangle_df)),
        ("Mack Chain Ladder",    lambda: run_mack_chainladder(triangle_df)),
    ]:
        print(f"\n  {name}...")
        try:
            result = fn()
            model_results[name] = result
            print(result.to_string())
        except Exception as exc:
            print(f"  ❌  {exc}")
            model_results[name] = None

    u_cl = model_results.get("Chain Ladder")

    # ------------------------------------------------------------------ #
    # INTELIGENCIA ARTIFICIAL                                              #
    # ------------------------------------------------------------------ #
    _header("TIER 2 — INTELIGENCIA ARTIFICIAL  (MC-Dropout uncertainty)")
    print("  150 pasadas estocásticas forward para cuantificación de incertidumbre.\n")

    u_gru = u_gru_std = u_trf = u_trf_std = None

    for name, run_fn, key in [
        ("GRU",         run_gru_triangle,       "gru"),
        ("Transformer", run_transformer_triangle, "trf"),
    ]:
        print(f"\n  {name}...")
        try:
            *_, ult_mean, ult_std = run_fn(triangle_df)
            model_results[name] = ult_mean
            uq_results[name] = (
                ult_mean, ult_std,
                ult_mean - 1.645 * ult_std,
                ult_mean + 1.645 * ult_std,
            )
            if key == "gru":
                u_gru, u_gru_std = ult_mean, ult_std
            else:
                u_trf, u_trf_std = ult_mean, ult_std

            ci_df = pd.DataFrame({
                "ultimate (media)":   ult_mean,
                "std (MC-Dropout)":   ult_std,
                "IC 90% inferior":    ult_mean - 1.645 * ult_std,
                "IC 90% superior":    ult_mean + 1.645 * ult_std,
            })
            print(ci_df.to_string())
        except Exception as exc:
            print(f"  ❌  {exc}")
            model_results[name] = None

    # ------------------------------------------------------------------ #
    # HÍBRIDO — Credibility Hybrid (Bühlmann)                             #
    # ------------------------------------------------------------------ #
    _header("TIER 3 — HÍBRIDO  (Credibilidad Bühlmann)")
    print("  Ensemble GRU + Transformer ponderado por 1/sigma MC-Dropout.")
    print("  Chain Ladder como ancla con peso fijo 20%.\n")

    ai_candidates: dict[str, tuple[pd.Series, pd.Series]] = {}
    if u_gru is not None and u_gru_std is not None:
        ai_candidates["GRU"] = (u_gru, u_gru_std)
    if u_trf is not None and u_trf_std is not None:
        ai_candidates["Transformer"] = (u_trf, u_trf_std)

    if ai_candidates and u_cl is not None:
        try:
            cred_ult = run_credibility_hybrid(
                ai_candidates, anchor=u_cl, anchor_weight=0.20
            )
            model_results["Credibility Hybrid"] = cred_ult
            print(cred_ult.to_string())
        except Exception as exc:
            print(f"  ❌  {exc}")
            model_results["Credibility Hybrid"] = None

    # ------------------------------------------------------------------ #
    # Resumen y métricas                                                   #
    # ------------------------------------------------------------------ #
    _header("RESUMEN Y MÉTRICAS")

    results_valid = {k: v for k, v in model_results.items() if v is not None}

    summary_df = create_model_summary(triangle_df, results_valid)
    print("\n", summary_df.to_string())

    comparison_df = compare_models_results(results_valid, triangle_df)
    if comparison_df is not None and len(comparison_df) > 0:
        # Reorder rows to match thesis presentation order
        order = {m: i for i, m in enumerate(_MODELS_ORDER)}
        comparison_df["_order"] = comparison_df["model"].map(order).fillna(99)
        comparison_df = comparison_df.sort_values("_order").drop(columns="_order")
        print("\n", comparison_df[["model", "mae", "rmse", "mape", "r2"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # Visualizaciones                                                      #
    # ------------------------------------------------------------------ #
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    print(f"\n  Generando gráficos → {fig_dir}/")

    try:
        plot_triangle(triangle_df, save_path=str(fig_dir / "triangulo.png"))
    except Exception as exc:
        print(f"  triángulo: {exc}")

    try:
        if summary_df is not None and len(summary_df) > 0:
            plot_ultimate_claims_comparison(
                summary_df,
                save_path=str(fig_dir / "ultimates_comparison.png"),
            )
            plot_ibnr_comparison(
                summary_df,
                save_path=str(fig_dir / "ibnr_comparison.png"),
            )
    except Exception as exc:
        print(f"  comparación ultimates/IBNR: {exc}")

    try:
        if comparison_df is not None and len(comparison_df) > 0:
            plot_model_comparison_metrics(
                comparison_df,
                save_path=str(fig_dir / "metricas_modelos.png"),
            )
    except Exception as exc:
        print(f"  métricas: {exc}")

    try:
        if uq_results:
            acc_years = sorted({int(x) for x in triangle_df["accident_year"].unique()})
            plot_uncertainty_bands(
                acc_years,
                results_valid,
                uq_results,
                save_path=str(fig_dir / "bandas_incertidumbre.png"),
            )
    except Exception as exc:
        print(f"  bandas de incertidumbre: {exc}")

    print("\nListo.")


if __name__ == "__main__":
    main()
