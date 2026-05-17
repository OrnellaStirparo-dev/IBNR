"""
src.ai — Modelos de IA para estimación IBNR.

Modelos incluidos en la tesis:
  - GRU / DeepTriangle  (Kuo 2019)
  - Transformer          (Vaswani et al. 2017 / Richman & Wüthrich 2026)
  - Credibility Hybrid   (Bühlmann-style, sección 4.2)

Todos los modelos neuronales incluyen cuantificación de incertidumbre
via MC-Dropout (Gal & Ghahramani 2016).
"""

from .gru_triangle import GRUStep, run_gru_triangle
from .hybrid_models import (
    convex_blend,
    multi_average,
    run_credibility_hybrid,
    run_hybrid_cl_transformer,
)
from .transformer_triangle import MiniDevTransformer, run_transformer_triangle

__all__ = [
    "GRUStep",
    "MiniDevTransformer",
    "convex_blend",
    "multi_average",
    "run_credibility_hybrid",
    "run_gru_triangle",
    "run_hybrid_cl_transformer",
    "run_transformer_triangle",
]
