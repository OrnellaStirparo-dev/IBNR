"""
SHAP feature importance for tree-based reserving models (XGBoost, LightGBM, CatBoost).

SHAP (SHapley Additive exPlanations) decomposes each prediction into the marginal
contribution of every feature, grounded in cooperative game theory.  For tree models
the exact Shapley values are computed efficiently via TreeSHAP (Lundberg et al., 2020).

In the actuarial context this answers: "how much does accident year vs development
year vs calendar year vs lagged cumulative drive each prediction?"  This is critical
for regulatory acceptance — the model's reasoning becomes auditable.

References
----------
Lundberg et al. (2020) "From local explanations to global understanding with
  explainable AI for trees"
SHAP documentation: https://shap.readthedocs.io
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def explain_model(
    model,
    X_train: pd.DataFrame,
    plot: bool = True,
    max_display: int = 10,
) -> pd.DataFrame:
    """
    Compute SHAP values for a tree model and return a feature importance table.

    Works with XGBoost, LightGBM, and CatBoost fitted models.

    Parameters
    ----------
    model       : fitted tree model (XGBRegressor, LGBMRegressor, CatBoostRegressor)
    X_train     : the feature matrix used for training (pd.DataFrame)
    plot        : whether to render a SHAP beeswarm summary plot
    max_display : number of features shown on the beeswarm plot

    Returns
    -------
    pd.DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending
    """
    try:
        import shap
    except ImportError:
        warnings.warn(
            "shap is not installed — run `pip install shap` to enable SHAP explanations.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        # Some multi-output models return a list; take the first output
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": X_train.columns.tolist(), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    if plot:
        import matplotlib.pyplot as plt

        shap.summary_plot(
            shap_values,
            X_train,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Feature Importance (XGBoost — triangle features)", fontsize=13)
        plt.tight_layout()
        plt.show()

    return importance_df


def explain_all_boosting(
    models: dict[str, object],
    X_train: pd.DataFrame,
    plot: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run SHAP explanations for all supplied tree models and return results dict.

    Parameters
    ----------
    models  : {model_name: fitted_model}
    X_train : shared training feature matrix
    plot    : show a beeswarm plot per model

    Returns
    -------
    {model_name: importance_df}
    """
    results: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        print(f"   SHAP → {name}")
        try:
            results[name] = explain_model(model, X_train, plot=plot)
        except Exception as exc:
            print(f"   SHAP failed for {name}: {exc}")
            results[name] = pd.DataFrame(columns=["feature", "mean_abs_shap"])
    return results
