"""
Bayesian Chain Ladder — conjugate log-Normal posterior on development factors.

Each log-development factor x_{i,k} = log(C_{i,k+1} / C_{i,k}) is modelled as:

    x_{i,k} ~ Normal(mu_k, sigma_k^2)

with a weakly informative Normal prior on mu_k centred on the volume-weighted MLE
log-factor.  The conjugate posterior is analytic (Normal-Normal update), so no MCMC
is required.  Posterior ultimate distributions are obtained by Monte Carlo draws from
the joint posterior, giving full predictive credible intervals.

References
----------
England & Verrall (2002) "Stochastic Claims Reserving in General Insurance"
Venter & Tampubolon (2010) "Bayesian Development Factors"
Wüthrich & Merz (2008) "Stochastic Claims Reserving Methods in Insurance"
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _build_triangle_matrix(
    triangle_df: pd.DataFrame,
) -> tuple[np.ndarray, list[int], int]:
    t = triangle_df.sort_values(["accident_year", "development_year"])
    p = t.pivot_table(
        index="accident_year",
        columns="development_year",
        values="cumulative_claims",
        aggfunc="first",
    ).sort_index()
    years = [int(y) for y in p.index]
    max_dev = int(p.columns.max())
    mat = p.reindex(columns=range(1, max_dev + 1)).values.astype(float)
    return mat, years, max_dev


def run_bayesian_chainladder(
    triangle_df: pd.DataFrame,
    n_samples: int = 5_000,
    prior_tau: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bayesian Chain Ladder with conjugate log-Normal posterior on development factors.

    Parameters
    ----------
    triangle_df  : long-format triangle (accident_year, development_year, cumulative_claims)
    n_samples    : number of Monte Carlo posterior draws for ultimate distribution
    prior_tau    : prior std on log-development factors — controls how strongly the
                   prior pulls toward the volume-weighted MLE (0.15 ≈ ±15% on factor)
    random_state : numpy RNG seed

    Returns
    -------
    ult_mean : posterior mean ultimate per accident year (pd.Series)
    ult_std  : posterior predictive std (pd.Series)
    ult_q05  : 5th-percentile credible lower bound (pd.Series)
    ult_q95  : 95th-percentile credible upper bound (pd.Series)
    """
    rng = np.random.default_rng(random_state)
    mat, years, max_dev = _build_triangle_matrix(triangle_df)
    n_years = mat.shape[0]

    # ------------------------------------------------------------------ #
    # 1. Volume-weighted MLE development factors (classical chain ladder) #
    # ------------------------------------------------------------------ #
    mle_factors = np.ones(max_dev - 1)
    for k in range(max_dev - 1):
        num = den = 0.0
        for i in range(n_years):
            if np.isfinite(mat[i, k]) and np.isfinite(mat[i, k + 1]) and mat[i, k] > 0:
                num += mat[i, k + 1]
                den += mat[i, k]
        mle_factors[k] = (num / den) if den > 0 else 1.0

    # ------------------------------------------------------------------ #
    # 2. Conjugate Normal-Normal posterior for each mu_k = E[log f_k]    #
    #                                                                      #
    #    Prior  : mu_k ~ N(log(f_k_mle), prior_tau^2)                     #
    #    Likeli : x_{i,k} ~ N(mu_k, sigma_k^2)  i.i.d.                   #
    #    Posterior precision = prior_prec + n_k / sigma_k^2               #
    #    Posterior mean      = weighted average of prior mean and x-bar   #
    # ------------------------------------------------------------------ #
    mu_post = np.zeros(max_dev - 1)
    sigma_post = np.zeros(max_dev - 1)

    for k in range(max_dev - 1):
        log_obs: list[float] = []
        for i in range(n_years):
            ci, cj = mat[i, k], mat[i, k + 1]
            if np.isfinite(ci) and np.isfinite(cj) and ci > 0 and cj > 0:
                log_obs.append(float(np.log(cj / ci)))

        m0 = float(np.log(mle_factors[k]))

        if len(log_obs) == 0:
            mu_post[k] = m0
            sigma_post[k] = prior_tau
        elif len(log_obs) == 1:
            # Only one observation — sigma estimated conservatively
            sigma_data = max(abs(log_obs[0] - m0), 1e-4)
            prec_prior = 1.0 / prior_tau ** 2
            prec_data = 1.0 / sigma_data ** 2
            prec_post = prec_prior + prec_data
            mu_post[k] = (m0 * prec_prior + log_obs[0] * prec_data) / prec_post
            sigma_post[k] = np.sqrt(1.0 / prec_post)
        else:
            xbar = float(np.mean(log_obs))
            sigma_data = max(float(np.std(log_obs, ddof=1)), 1e-4)
            n = len(log_obs)
            prec_prior = 1.0 / prior_tau ** 2
            prec_data = n / sigma_data ** 2
            prec_post = prec_prior + prec_data
            mu_post[k] = (m0 * prec_prior + xbar * prec_data) / prec_post
            sigma_post[k] = np.sqrt(1.0 / prec_post)

    # ------------------------------------------------------------------ #
    # 3. Latest observed diagonal per accident year                       #
    # ------------------------------------------------------------------ #
    latest_val = np.full(n_years, np.nan)
    latest_dev = np.full(n_years, -1, dtype=int)  # 0-indexed column
    for i in range(n_years):
        for j in range(max_dev - 1, -1, -1):
            if np.isfinite(mat[i, j]) and mat[i, j] > 0:
                latest_val[i] = mat[i, j]
                latest_dev[i] = j
                break

    # ------------------------------------------------------------------ #
    # 4. Monte Carlo: sample mu_k ~ posterior, project to ultimate        #
    # ------------------------------------------------------------------ #
    # mu_draws : (n_samples, max_dev-1) — independent posterior samples
    mu_draws = rng.normal(mu_post, sigma_post, size=(n_samples, max_dev - 1))
    f_draws = np.exp(mu_draws)  # sampled development factors

    samples = np.full((n_years, n_samples), np.nan)
    for i in range(n_years):
        if latest_dev[i] < 0:
            continue
        if latest_dev[i] == max_dev - 1:
            # Fully mature — no tail uncertainty
            samples[i, :] = latest_val[i]
        else:
            # Product of remaining sampled factors
            tail = f_draws[:, latest_dev[i] : max_dev - 1]  # (S, remaining_periods)
            samples[i, :] = latest_val[i] * tail.prod(axis=1)

    ult_mean = pd.Series(np.nanmean(samples, axis=1), index=years, name="Bayesian CL")
    ult_std = pd.Series(np.nanstd(samples, axis=1), index=years, name="bayesian_cl_std")
    ult_q05 = pd.Series(np.nanpercentile(samples, 5, axis=1), index=years, name="bayesian_cl_q05")
    ult_q95 = pd.Series(np.nanpercentile(samples, 95, axis=1), index=years, name="bayesian_cl_q95")

    return ult_mean, ult_std, ult_q05, ult_q95
