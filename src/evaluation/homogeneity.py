"""
Homogeneity diagnostics run *before* any model is fitted (comment 5 of the
reviewer): is the portfolio behind a triangle stable across accident years?

Every estimator compared in this thesis assumes, explicitly (chain ladder,
Mack) or implicitly (the networks learn one development pattern per
triangle), that accident years are exchangeable once the level is removed.
Three observable symptoms of a portfolio that changed over time:

1. **Volume shifts** — earned premium jumping between accident years signals a
   change in size or mix (acquisitions, exits, a new product).
2. **Loss-ratio drift** — the paid-to-premium ratio at the first development
   period trending across accident years signals a change in pricing, mix or
   claims handling.
3. **Unstable development** — individual first link ratios f_{i,1}
   = C_{i,2}/C_{i,1} that are dispersed or trending across accident years signal
   a change in the settlement pattern, which is exactly what chain ladder
   cannot absorb.

Only information available at the valuation date is used (upper triangle and
premium), so the filter itself does not leak the outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Thresholds. They are judgement calls, stated explicitly so they can be
# defended and varied: a portfolio is flagged non-homogeneous if ANY holds.
MAX_PREMIUM_JUMP = 0.50   # |ΔP/P| between consecutive accident years
MAX_LR_CV = 0.35          # coefficient of variation of C_{i,1}/P_i
MAX_F1_CV = 0.25          # coefficient of variation of individual f_{i,1}
TREND_P = 0.05            # Spearman p-value for a monotone trend in f_{i,1}


def diagnose(upper: pd.DataFrame, prem: pd.Series) -> dict:
    C = upper.values.astype(float)
    P = prem.values.astype(float)

    jumps = np.abs(np.diff(P) / P[:-1])
    lr1 = C[:, 0] / P
    f1 = np.array([C[i, 1] / C[i, 0] for i in range(C.shape[0]) if np.isfinite(C[i, 1])])
    rho, pval = spearmanr(np.arange(len(f1)), f1) if len(f1) >= 4 else (np.nan, np.nan)

    out = {
        "max_premium_jump": float(np.max(jumps)),
        "lr1_cv": float(np.std(lr1) / np.mean(lr1)),
        "f1_cv": float(np.std(f1) / np.mean(f1)),
        "f1_trend_rho": float(rho),
        "f1_trend_p": float(pval),
    }
    out["volume_flag"] = out["max_premium_jump"] > MAX_PREMIUM_JUMP
    out["lr_flag"] = out["lr1_cv"] > MAX_LR_CV
    out["dev_flag"] = (out["f1_cv"] > MAX_F1_CV) or (out["f1_trend_p"] < TREND_P)
    out["homogeneous"] = not (out["volume_flag"] or out["lr_flag"] or out["dev_flag"])
    return out


def pattern_distance(f_a: np.ndarray, f_b: np.ndarray) -> float:
    """
    Distance between two development patterns, used to justify which lines
    may be pooled: maximum absolute gap between their cumulative
    percent-developed curves (1/F_j), a Kolmogorov–Smirnov-type statistic.
    """
    def pct(f):
        F = np.cumprod(f[::-1])[::-1]
        return np.append(1.0 / F, 1.0)
    return float(np.max(np.abs(pct(f_a) - pct(f_b))))
