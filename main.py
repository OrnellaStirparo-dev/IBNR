"""
IBNR comparison: traditional triangle methods vs modern ML tabular models.
"""

from __future__ import annotations

import pandas as pd

from src.data.load_data import load_claims_triangle
from src.data.transformer import clean_siniestros_data, create_claims_triangle
from src.evaluation.comparison import compare_models_results, create_model_summary
from src.ml.catboost_model import run_catboost
from src.ml.lightgbm_model import run_lightgbm
from src.ml.pytorch_mlp import run_pytorch_mlp
from src.ml.xgboost_model import run_xgboost
from src.traditional.bf_model import run_bornhuetter_ferguson
from src.traditional.cape_cod_model import run_cape_cod
from src.traditional.chainladder_model import run_chainladder
from src.traditional.mack_model import run_mack_chainladder
from src.visualization.comparison import (
    plot_ibnr_comparison,
    plot_model_comparison_metrics,
    plot_ultimate_claims_comparison,
)
from src.visualization.triangle_viz import plot_triangle


def main() -> None:
    print("=" * 60)
    print("IBNR — Traditional vs ML (LightGBM, CatBoost, XGBoost, PyTorch)")
    print("=" * 60)

    print("\n📂 Loading raw Excel from data/raw (or IBNR_RAW_XLSX)...")
    raw_df = load_claims_triangle()
    print(f"   Shape: {raw_df.shape}; columns: {raw_df.columns.tolist()}")

    print("\n🧹 Cleaning...")
    cleaned_df = clean_siniestros_data(raw_df)

    print("\n🔺 Building cumulative triangle...")
    triangle_df = create_claims_triangle(cleaned_df)
    print(triangle_df.head(12).to_string())
    print(f"   Cells: {len(triangle_df)}")

    model_results: dict[str, pd.Series | None] = {}

    print("\n" + "=" * 60)
    print("TRADITIONAL (chain ladder, Mack mean, BF, Cape Cod proxy)")
    print("=" * 60)

    for name, fn in [
        ("Chain Ladder", run_chainladder),
        ("Mack Chain Ladder", run_mack_chainladder),
        ("Bornhuetter-Ferguson", run_bornhuetter_ferguson),
        ("Cape Cod", run_cape_cod),
    ]:
        print(f"\n▶ {name}...")
        try:
            model_results[name] = fn(triangle_df)
            print(model_results[name].to_string())
        except Exception as e:
            print(f"   ❌ {e}")
            model_results[name] = None

    print("\n" + "=" * 60)
    print("ML / AI (gradient boosting + neural net)")
    print("=" * 60)

    ml_runners = [
        ("LightGBM", lambda: run_lightgbm(triangle_df)[1]),
        ("CatBoost", lambda: run_catboost(triangle_df)[1]),
        ("XGBoost", lambda: run_xgboost(triangle_df)[1]),
        ("PyTorch MLP", lambda: run_pytorch_mlp(triangle_df)[3]),
    ]
    for name, run in ml_runners:
        print(f"\n▶ {name}...")
        try:
            model_results[name] = run()
            print(model_results[name].to_string())
        except Exception as e:
            print(f"   ❌ {e}")
            model_results[name] = None

    print("\n" + "=" * 60)
    print("SUMMARY & METRICS")
    print("=" * 60)

    summary_df = create_model_summary(triangle_df, model_results)
    print("\n", summary_df.to_string())

    results_dict = {k: v for k, v in model_results.items() if v is not None}
    comparison_df = compare_models_results(results_dict, triangle_df)
    if len(comparison_df) > 0:
        print("\n", comparison_df.to_string())

    print("\n📊 Plots (close windows to continue)...")
    try:
        plot_triangle(triangle_df)
    except Exception as e:
        print(f"Triangle plot: {e}")
    try:
        if summary_df is not None and len(summary_df) > 0:
            plot_ultimate_claims_comparison(summary_df)
            plot_ibnr_comparison(summary_df)
    except Exception as e:
        print(f"Summary plots: {e}")
    try:
        if comparison_df is not None and len(comparison_df) > 0:
            plot_model_comparison_metrics(comparison_df)
    except Exception as e:
        print(f"Metrics plot: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
