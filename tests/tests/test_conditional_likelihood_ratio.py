import numpy as np
import pytest
import scipy

from ivmodels.tests.conditional_likelihood_ratio import (
    _conditional_likelihood_ratio_critical_value_function,
)


@pytest.mark.parametrize("p", [1, 5, 20])
@pytest.mark.parametrize("q", [0, 20])
@pytest.mark.parametrize("s_min", [0.01, 1, 1e3])
@pytest.mark.parametrize("z", [0.1, 1, 10])
@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6])
def test_conditional_likelihood_ratio_critical_value_function_tol(p, q, s_min, z, tol):
    approx = _conditional_likelihood_ratio_critical_value_function(
        p, q + p, s_min, z, tol=tol
    )
    exact = _conditional_likelihood_ratio_critical_value_function(
        p, q + p, s_min, z, tol=1e-8
    )
    assert np.isclose(approx, exact, atol=2 * tol)


@pytest.mark.parametrize("p", [1, 5, 20])
@pytest.mark.parametrize("q", [0, 20])
def test_conditional_likelihood_ratio_critical_value_function_equal_to_chi2(p, q):
    for z in np.linspace(0, 2 * (p + q), 10):
        assert np.isclose(
            _conditional_likelihood_ratio_critical_value_function(p, q + p, 1e-6, z),
            1 - scipy.stats.chi2(p + q).cdf(z),
            atol=1e-4,
        )

    # TODO: Implement something that works for p=1 and s_min rather large, s.t.
    # the difference below > tol but the approximation does take very long to converge.
    # for z in np.linspace(0, 2 * (p + q), 10):
    #     assert np.isclose(
    #         _conditional_likelihood_ratio_critical_value_function(p, q + p, 1e3, z),
    #         1 - scipy.stats.chi2(p).cdf(z),
    #         atol=1e-2
    #     )


@pytest.mark.parametrize("p", [1, 2, 5])
@pytest.mark.parametrize("q", [0, 1, 5, 20])
@pytest.mark.parametrize("s_min", [0.01, 1, 1e3])
@pytest.mark.parametrize("z", [0.1, 1, 10])
def test_conditional_likelihood_ratio_critical_value_function(p, q, s_min, z):
    chi2p = scipy.stats.chi2.rvs(size=20000, df=p, random_state=0)
    chi2q = scipy.stats.chi2.rvs(size=20000, df=q, random_state=1) if q > 0 else 0
    D = np.sqrt((chi2p + chi2q - s_min) ** 2 + 4 * chi2p * s_min)
    Q = 1 / 2 * (chi2p + chi2q - s_min + D)
    p_value = np.mean(Q > z)

    assert np.isclose(
        p_value,
        _conditional_likelihood_ratio_critical_value_function(p, q + p, s_min, z),
        atol=1e-2,
    )
