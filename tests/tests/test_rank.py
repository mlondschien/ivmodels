import numpy as np
import pytest
import scipy

from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import inverse_anderson_rubin_test, rank_test


@pytest.mark.parametrize("n, mx, k, u", [(40, 2, 2, 1), (40, 2, 5, 2), (40, 1, 2, 2)])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_bounded_inverse_anderson_rubin_p_value(n, mx, k, u, fit_intercept):
    Z, X, y, _, _, _ = simulate_gaussian_iv(
        n=n, mx=mx, k=k, u=u, seed=0, include_intercept=fit_intercept
    )

    statistic = rank_test(Z, X, fit_intercept=fit_intercept)[0]
    # different degrees of freedom than used in the rank test
    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=k)
    assert 0.999 > p_value > 1e-12

    quad_below = inverse_anderson_rubin_test(
        Z, X, y, p_value * 0.999, fit_intercept=fit_intercept
    )
    quad_above = inverse_anderson_rubin_test(
        Z, X, y, p_value * 1.001, fit_intercept=fit_intercept
    )

    assert np.isinf(quad_below.volume())
    assert np.isfinite(quad_above.volume())


@pytest.mark.parametrize("n, k, m", [(2000, 2, 2), (2000, 2, 1), (2000, 8, 3)])
def test_rank_test(n, k, m):
    rng = np.random.default_rng(0)

    Pi = rng.normal(size=(k, m - 1)) @ rng.normal(size=(m - 1, m))

    n_seeds = 1000
    statistics = np.zeros(n_seeds)
    p_values = np.zeros(n_seeds)

    for idx in range(n_seeds):
        Z = rng.normal(size=(n, k))
        X = Z @ Pi + rng.normal(size=(n, m))

        statistics[idx], p_values[idx] = rank_test(Z, X)

    assert (
        scipy.stats.kstest(p_values, scipy.stats.uniform(loc=0.0, scale=1.0).cdf).pvalue
        > 0.05
    )
