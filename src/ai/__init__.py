"""
src.ai — Deep learning and Bayesian models for IBNR estimation.

All neural models include MC-dropout uncertainty quantification.
The Bayesian Chain Ladder provides analytic credible intervals.
The credibility hybrid weights models by their predictive uncertainty.
"""

from .bayesian_chainladder import run_bayesian_chainladder
from .deep_residual_net import DeepResidualTabular, run_deep_residual_net
from .gru_triangle import GRUStep, run_gru_triangle
from .hybrid_models import (
    convex_blend,
    multi_average,
    run_credibility_hybrid,
    run_hybrid_cl_bf_dl,
    run_hybrid_cl_deepresnet,
    run_hybrid_cl_gru,
    run_hybrid_cl_transformer,
    run_hybrid_cl_xgboost,
)
from .pytorch_mlp import run_pytorch_mlp
from .transformer_triangle import MiniDevTransformer, run_transformer_triangle

__all__ = [
    "DeepResidualTabular",
    "GRUStep",
    "MiniDevTransformer",
    "convex_blend",
    "multi_average",
    "run_bayesian_chainladder",
    "run_credibility_hybrid",
    "run_deep_residual_net",
    "run_gru_triangle",
    "run_hybrid_cl_bf_dl",
    "run_hybrid_cl_deepresnet",
    "run_hybrid_cl_gru",
    "run_hybrid_cl_transformer",
    "run_hybrid_cl_xgboost",
    "run_pytorch_mlp",
    "run_transformer_triangle",
]
