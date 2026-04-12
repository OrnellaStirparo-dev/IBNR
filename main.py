"""
IBNR comparison: traditional actuarial methods · tabular ML · deep AI · Bayesian · hybrids.

Architecture of this experiment
--------------------------------
Tier 1 — Traditional (rule-based actuarial)
    Chain Ladder, Mack, Bornhuetter-Ferguson, Cape Cod

Tier 1+ — Bayesian extension of traditional
    Bayesian Chain Ladder: conjugate log-Normal posterior on development factors,
    full predictive distribution via Monte Carlo draws.

Tier 2 — Tabular ML (gradient boosting, feature-engineered)
    XGBoost, LightGBM, CatBoost  +  SHAP feature attribution

Tier 3 — Sequential / deep AI (representation learning, no hand-crafted features)
    MLP, Deep ResNet (tabular), GRU development path, Transformer (dev tokens)
    All include MC-dropout uncertainty quantification.

Tier 4 — Hybrid / ensemble
    Fixed convex blends (CL + AI)
    Credibility hybrid: Bühlmann-style weighting by inverse MC-dropout std
"""

from __future__ import annotations

import os

# Must be set before PyTorch/XGBoost/LightGBM are imported — prevents the
# OpenMP duplicate-library segfault on macOS (each ships its own libomp).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd

from src.ai.bayesian_chainladder import run_bayesian_chainladder
from src.ai.deep_residual_net import run_deep_residual_net
from src.ai.gru_triangle import run_gru_triangle
from src.ai.hybrid_models import (
    multi_average,
    run_credibility_hybrid,
    run_hybrid_cl_bf_dl,
    run_hybrid_cl_deepresnet,
    run_hybrid_cl_gru,
    run_hybrid_cl_transformer,
    run_hybrid_cl_xgboost,
)
from src.ai.pytorch_mlp import run_pytorch_mlp
from src.ai.transformer_triangle import run_transformer_triangle
from src.data.load_data import load_claims_triangle
from src.data.transformer import clean_siniestros_data, create_claims_triangle
from src.evaluation.comparison import compare_models_results, create_model_summary
from src.ml.catboost_model import run_catboost
from src.ml.features import enrich_triangle_features, matrix_xy
from src.ml.lightgbm_model import run_lightgbm
from src.ml.shap_explainer import explain_all_boosting
from src.ml.xgboost_model import run_xgboost
from src.traditional.bf_model import run_bornhuetter_ferguson
from src.traditional.cape_cod_model import run_cape_cod
from src.traditional.chainladder_model import run_chainladder
from src.traditional.mack_model import run_mack_chainladder
from src.visualization.comparison import (
    plot_ibnr_comparison,
    plot_model_comparison_metrics,
    plot_ultimate_claims_comparison,
    plot_uncertainty_bands,
)
from src.visualization.triangle_viz import plot_triangle


def _run(label: str, fn, results: dict) -> None:
    print(f"\n  {label}...")
    try:
        s = fn()
        results[label] = s
        print(f"  {s.to_string()}")
    except Exception as e:
        print(f"  ❌  {e}")
        results[label] = None


def main() -> None:
    print("=" * 70)
    print("IBNR  —  Traditional · Bayesian · Tabular ML · Deep AI · Hybrids")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Data                                                                #
    # ------------------------------------------------------------------ #
    print("\n  Loading raw Excel from data/raw (or IBNR_RAW_XLSX)...")
    raw_df = load_claims_triangle()
    print(f"  Shape: {raw_df.shape};  columns: {raw_df.columns.tolist()}")

    print("\n  Cleaning...")
    cleaned_df = clean_siniestros_data(raw_df)

    print("\n  Building cumulative triangle...")
    triangle_df = create_claims_triangle(cleaned_df)
    print(triangle_df.head(12).to_string())
    print(f"  Cells: {len(triangle_df)}")

    model_results: dict[str, pd.Series | None] = {}

    # ------------------------------------------------------------------ #
    # TIER 1 — Traditional                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("TIER 1 — TRADITIONAL ACTUARIAL")
    print("=" * 70)

    for name, fn in [
        ("Chain Ladder", run_chainladder),
        ("Mack Chain Ladder", run_mack_chainladder),
        ("Bornhuetter-Ferguson", run_bornhuetter_ferguson),
        ("Cape Cod", run_cape_cod),
    ]:
        _run(name, lambda f=fn: f(triangle_df), model_results)

    u_cl = model_results.get("Chain Ladder")
    u_bf = model_results.get("Bornhuetter-Ferguson")

    # ------------------------------------------------------------------ #
    # TIER 1+ — Bayesian Chain Ladder                                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("TIER 1+  — BAYESIAN CHAIN LADDER")
    print("  Conjugate log-Normal posterior on development factors.")
    print("  Full predictive distribution via 5,000 Monte Carlo draws.")
    print("=" * 70)

    uq_results: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}

    print("\n  Bayesian Chain Ladder...")
    try:
        bcl_mean, bcl_std, bcl_q05, bcl_q95 = run_bayesian_chainladder(triangle_df)
        model_results["Bayesian CL"] = bcl_mean
        uq_results["Bayesian CL"] = (bcl_mean, bcl_std, bcl_q05, bcl_q95)
        print(f"  Posterior mean ultimate:\n{bcl_mean.to_string()}")
        summary_uq = pd.DataFrame({
            "mean": bcl_mean,
            "std":  bcl_std,
            "q05":  bcl_q05,
            "q95":  bcl_q95,
        })
        print(f"\n  Credible interval (90%):\n{summary_uq.to_string()}")
    except Exception as e:
        print(f"  ❌  {e}")

    # ------------------------------------------------------------------ #
    # TIER 2 — Tabular ML (gradient boosting) + SHAP                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("TIER 2 — TABULAR ML  (gradient boosting + SHAP attribution)")
    print("=" * 70)

    fitted_boosting: dict[str, object] = {}
    u_xgb: pd.Series | None = None

    for name, run_fn, key in [
        ("LightGBM", run_lightgbm, "lgbm"),
        ("CatBoost", run_catboost, "cat"),
        ("XGBoost",  run_xgboost,  "xgb"),
    ]:
        print(f"\n  {name}...")
        try:
            fitted_model, ult = run_fn(triangle_df)
            model_results[name] = ult
            fitted_boosting[name] = fitted_model
            if key == "xgb":
                u_xgb = ult
            print(ult.to_string())
        except Exception as e:
            print(f"  ❌  {e}")
            model_results[name] = None

    # SHAP explanations on the shared training features
    print("\n  SHAP feature attribution...")
    try:
        enr = enrich_triangle_features(triangle_df)
        X_train, _ = matrix_xy(enr)
        shap_results = explain_all_boosting(fitted_boosting, X_train, plot=True)
        for name, imp_df in shap_results.items():
            if len(imp_df) > 0:
                print(f"\n  {name} — SHAP feature ranking:\n{imp_df.to_string(index=False)}")
    except Exception as e:
        print(f"  SHAP step failed: {e}")

    # ------------------------------------------------------------------ #
    # TIER 3 — Deep AI  (MC-dropout UQ)                                  #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("TIER 3 — DEEP AI  (neural networks · MC-dropout uncertainty)")
    print("  MC-dropout: dropout active during 150 stochastic inference passes.")
    print("  Predictive std used for credibility weighting in Tier 4.")
    print("=" * 70)

    u_mlp = u_deep = u_gru = u_trf = None
    u_mlp_std = u_deep_std = u_gru_std = u_trf_std = None

    for name, run_fn, ult_key in [
        ("MLP",          run_pytorch_mlp,       "mlp"),
        ("Deep ResNet",  run_deep_residual_net,  "deep"),
        ("GRU",          run_gru_triangle,       "gru"),
        ("Transformer",  run_transformer_triangle, "trf"),
    ]:
        print(f"\n  {name}...")
        try:
            *_, ult_mean, ult_std = run_fn(triangle_df)
            model_results[name] = ult_mean
            uq_results[name] = (ult_mean, ult_std, ult_mean - 1.645 * ult_std, ult_mean + 1.645 * ult_std)
            if ult_key == "mlp":
                u_mlp, u_mlp_std = ult_mean, ult_std
            elif ult_key == "deep":
                u_deep, u_deep_std = ult_mean, ult_std
            elif ult_key == "gru":
                u_gru, u_gru_std = ult_mean, ult_std
            elif ult_key == "trf":
                u_trf, u_trf_std = ult_mean, ult_std
            ci_df = pd.DataFrame({
                "mean": ult_mean,
                "std": ult_std,
                "q05 (approx)": ult_mean - 1.645 * ult_std,
                "q95 (approx)": ult_mean + 1.645 * ult_std,
            })
            print(ci_df.to_string())
        except Exception as e:
            print(f"  ❌  {e}")
            model_results[name] = None

    # ------------------------------------------------------------------ #
    # TIER 4 — Hybrids                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("TIER 4 — HYBRID / ENSEMBLE")
    print("=" * 70)

    # Fixed convex blends (traditional style)
    if u_cl is not None and u_deep is not None:
        _run("Hybrid CL+DeepResNet (50/50)", lambda: run_hybrid_cl_deepresnet(u_cl, u_deep, 0.5), model_results)
    if u_cl is not None and u_gru is not None:
        _run("Hybrid CL+GRU (50/50)", lambda: run_hybrid_cl_gru(u_cl, u_gru, 0.5), model_results)
    if u_cl is not None and u_trf is not None:
        _run("Hybrid CL+Transformer (50/50)", lambda: run_hybrid_cl_transformer(u_cl, u_trf, 0.5), model_results)
    if u_cl is not None and u_xgb is not None:
        _run("Hybrid CL+XGBoost (50/50)", lambda: run_hybrid_cl_xgboost(u_cl, u_xgb, 0.5), model_results)
    if u_cl is not None and u_bf is not None and u_deep is not None:
        _run(
            "Hybrid CL+BF+DeepResNet (35/25/40)",
            lambda: run_hybrid_cl_bf_dl(u_cl, u_bf, u_deep, w_cl=0.35, w_bf=0.25),
            model_results,
        )

    # Credibility hybrid — Bühlmann-style weighting by inverse MC-dropout std
    ai_candidates: dict[str, tuple[pd.Series, pd.Series]] = {}
    if u_deep is not None and u_deep_std is not None:
        ai_candidates["Deep ResNet"] = (u_deep, u_deep_std)
    if u_gru is not None and u_gru_std is not None:
        ai_candidates["GRU"] = (u_gru, u_gru_std)
    if u_trf is not None and u_trf_std is not None:
        ai_candidates["Transformer"] = (u_trf, u_trf_std)

    if ai_candidates:
        print("\n  Credibility Hybrid (uncertainty-weighted, CL anchor 20%)...")
        try:
            cred_ult = run_credibility_hybrid(
                ai_candidates, anchor=u_cl, anchor_weight=0.20
            )
            model_results["Credibility Hybrid"] = cred_ult
            print(cred_ult.to_string())
        except Exception as e:
            print(f"  ❌  {e}")
            model_results["Credibility Hybrid"] = None

    # ------------------------------------------------------------------ #
    # Summary & metrics                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SUMMARY & METRICS")
    print("=" * 70)

    summary_df = create_model_summary(triangle_df, model_results)
    print("\n", summary_df.to_string())

    results_valid = {k: v for k, v in model_results.items() if v is not None}
    comparison_df = compare_models_results(results_valid, triangle_df)
    if len(comparison_df) > 0:
        print("\n", comparison_df.to_string())

    # ------------------------------------------------------------------ #
    # Visualisation                                                       #
    # ------------------------------------------------------------------ #
    print("\n  Plots (close windows to continue)...")

    try:
        plot_triangle(triangle_df)
    except Exception as e:
        print(f"  Triangle plot: {e}")

    try:
        if summary_df is not None and len(summary_df) > 0:
            plot_ultimate_claims_comparison(summary_df)
            plot_ibnr_comparison(summary_df)
    except Exception as e:
        print(f"  Summary plots: {e}")

    try:
        if comparison_df is not None and len(comparison_df) > 0:
            plot_model_comparison_metrics(comparison_df)
    except Exception as e:
        print(f"  Metrics plot: {e}")

    try:
        if uq_results:
            acc_years = sorted({int(x) for x in triangle_df["accident_year"].unique()})
            plot_uncertainty_bands(acc_years, results_valid, uq_results)
    except Exception as e:
        print(f"  Uncertainty bands plot: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
