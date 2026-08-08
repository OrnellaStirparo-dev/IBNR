"""
Bayesian Chain Ladder — conjugate log-Normal model on the development factors
(Sección 3.1).

Each observed log link ratio is modelled as

    x_{i,k} = log(C_{i,k+1} / C_{i,k}) ~ Normal(mu_k, sigma_k^2)

with a Normal prior on ``mu_k``. The Normal-Normal conjugacy gives an analytic
posterior, so no MCMC is needed, and the posterior mean of ``mu_k`` is a
credibility blend of the prior and the observed mean — the connection to
Bühlmann that makes the method a generalisation of what the actuary already
does rather than a separate technique.

Two things this implementation is careful about.

**The prior is not taken from the data it is applied to.** Centring the prior on
the volume-weighted factor estimated over the whole triangle is circular: the
"prior" then carries the same information as the likelihood, the posterior
merely shrinks towards the classical estimate, and the method collapses onto the
chain ladder while appearing to add information. Here the default prior comes
from the accident years that are already fully developed, which is genuine prior
experience with respect to the immature years actually being projected. An
external prior can be supplied instead, which is what the method is for.

**The predictive distribution carries process variance, not only parameter
uncertainty.** Sampling ``mu_k`` alone gives a credible interval for the *mean*
development factor, which is far narrower than the distribution of the ultimate
and would overstate precision. Each draw here samples the individual log factor
``x ~ Normal(mu_draw, sigma_k^2)``, so the spread reflects both the uncertainty
about the parameter and the variability of a single accident year around it.
This is what makes the interval comparable in kind to Mack's MSEP.

References
----------
England & Verrall (2002); Wüthrich & Merz (2008); Bühlmann & Gisler (2005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _build_triangle_matrix(
    triangle_df: pd.DataFrame,
) -> tuple[np.ndarray, list[int], int]:
    pivot = triangle_df.pivot_table(
        index="accident_year",
        columns="development_year",
        values="cumulative_claims",
        aggfunc="first",
    ).sort_index()
    years = [int(y) for y in pivot.index]
    max_dev = int(pivot.columns.max())
    return pivot.reindex(columns=range(1, max_dev + 1)).values.astype(float), years, max_dev


def _log_ratios(mat: np.ndarray, k: int, rows: range | list[int]) -> list[float]:
    out = []
    for i in rows:
        prev, cur = mat[i, k], mat[i, k + 1]
        if np.isfinite(prev) and np.isfinite(cur) and prev > 0 and cur > 0:
            out.append(float(np.log(cur / prev)))
    return out


def _mature_rows(mat: np.ndarray) -> list[int]:
    """Accident years observed all the way to the last development column."""
    n_ay, n_dev = mat.shape
    return [i for i in range(n_ay) if np.isfinite(mat[i, n_dev - 1]) and mat[i, n_dev - 1] > 0]


def prior_from_mature_years(triangle_df: pd.DataFrame) -> pd.Series:
    """
    Prior mean log development factor per period, from the fully developed
    accident years only.

    These years contribute no information about the immature years' own
    emerging experience, so using them as the prior keeps the credibility blend
    honest: prior and likelihood come from disjoint data.
    """
    mat, _, max_dev = _build_triangle_matrix(triangle_df)
    mature = _mature_rows(mat)
    out = {}
    for k in range(max_dev - 1):
        obs = _log_ratios(mat, k, mature)
        out[k + 1] = float(np.mean(obs)) if obs else 0.0
    return pd.Series(out, name="prior_log_factor")


def run_bayesian_chainladder(
    triangle_df: pd.DataFrame,
    n_samples: int = 5_000,
    prior_tau: float = 0.15,
    prior_log_factors: pd.Series | None = None,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bayesian Chain Ladder with a conjugate log-Normal posterior.

    Parameters
    ----------
    n_samples         : posterior predictive draws
    prior_tau         : prior standard deviation on the log factors. Small values
                        make the prior dominate; large values converge to the
                        classical chain ladder.
    prior_log_factors : prior mean log factor per development period, indexed
                        1…max_dev-1. Defaults to :func:`prior_from_mature_years`.

    Returns
    -------
    ``(ult_mean, ult_std, ult_q05, ult_q95)`` per accident year.
    """
    rng = np.random.default_rng(random_state)
    mat, years, max_dev = _build_triangle_matrix(triangle_df)
    n_years = mat.shape[0]

    prior = (
        prior_from_mature_years(triangle_df)
        if prior_log_factors is None
        else prior_log_factors
    )

    mu_post = np.zeros(max_dev - 1)
    sd_post = np.zeros(max_dev - 1)
    sigma_process = np.zeros(max_dev - 1)

    for k in range(max_dev - 1):
        observed = _log_ratios(mat, k, range(n_years))
        m0 = float(prior.get(k + 1, 0.0))

        if not observed:
            mu_post[k], sd_post[k], sigma_process[k] = m0, prior_tau, prior_tau
            continue

        sigma = (
            max(float(np.std(observed, ddof=1)), 1e-4)
            if len(observed) > 1
            else max(abs(observed[0] - m0), 1e-4)
        )
        prec_prior = 1.0 / prior_tau**2
        prec_data = len(observed) / sigma**2
        prec_post = prec_prior + prec_data

        mu_post[k] = (m0 * prec_prior + float(np.mean(observed)) * prec_data) / prec_post
        sd_post[k] = float(np.sqrt(1.0 / prec_post))
        sigma_process[k] = sigma

    latest_val = np.full(n_years, np.nan)
    latest_dev = np.full(n_years, -1, dtype=int)
    for i in range(n_years):
        for j in range(max_dev - 1, -1, -1):
            if np.isfinite(mat[i, j]) and mat[i, j] > 0:
                latest_val[i], latest_dev[i] = mat[i, j], j
                break

    # Parameter uncertainty (mu) and process variance (sigma) in each draw.
    mu_draws = rng.normal(mu_post, sd_post, size=(n_samples, max_dev - 1))
    x_draws = rng.normal(mu_draws, sigma_process, size=(n_samples, max_dev - 1))
    f_draws = np.exp(x_draws)

    samples = np.full((n_years, n_samples), np.nan)
    for i in range(n_years):
        if latest_dev[i] < 0:
            continue
        if latest_dev[i] == max_dev - 1:
            samples[i, :] = latest_val[i]
        else:
            samples[i, :] = latest_val[i] * f_draws[:, latest_dev[i] : max_dev - 1].prod(axis=1)

    return (
        pd.Series(np.nanmean(samples, axis=1), index=years, name="Bayesian CL"),
        pd.Series(np.nanstd(samples, axis=1), index=years, name="bayesian_cl_std"),
        pd.Series(np.nanpercentile(samples, 5, axis=1), index=years, name="bayesian_cl_q05"),
        pd.Series(np.nanpercentile(samples, 95, axis=1), index=years, name="bayesian_cl_q95"),
    )
