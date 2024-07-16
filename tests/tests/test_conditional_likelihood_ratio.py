import numpy as np
import pytest
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests.conditional_likelihood_ratio import (
    conditional_likelihood_ratio_critical_value_function,
    conditional_likelihood_ratio_test,
)


@pytest.mark.parametrize("p", [1, 5, 20])
@pytest.mark.parametrize("q", [0, 20])
@pytest.mark.parametrize("s_min", [0.01, 1, 1e3])
@pytest.mark.parametrize("z", [0.1, 10])
@pytest.mark.parametrize("tol", [1e-2, 1e-4])
@pytest.mark.parametrize("method", ["power_series", "numerical_integration"])
def test_conditional_likelihood_ratio_critical_value_function_tol(
    p, q, s_min, z, tol, method
):
    approx = conditional_likelihood_ratio_critical_value_function(
        p, q + p, s_min, z, tol=tol, method=method
    )
    exact = conditional_likelihood_ratio_critical_value_function(
        p, q + p, s_min, z, tol=1e-8, method=method
    )
    assert np.isclose(approx, exact, atol=3 * tol)


@pytest.mark.parametrize("p", [1, 5, 20])
@pytest.mark.parametrize("q", [0, 20])
@pytest.mark.parametrize("method", ["numerical_integration", "power_series"])
def test_conditional_likelihood_ratio_critical_value_function_equal_to_chi2(
    p, q, method
):
    for z in np.linspace(0, 2 * (p + q), 10):
        assert np.isclose(
            conditional_likelihood_ratio_critical_value_function(
                p, q + p, 1e-6, z, method
            ),
            1 - scipy.stats.chi2(p + q).cdf(z),
            atol=1e-4,
        )

    # The "power_series" method is very slow for a = (s_min + z) / s_min close to 1.
    if method == "numerical_integration":
        for z in np.linspace(0, 2 * (p + q), 10):
            assert np.isclose(
                conditional_likelihood_ratio_critical_value_function(p, q + p, 1e5, z),
                1 - scipy.stats.chi2(p).cdf(z),
                atol=1e-2,
            )


@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("q", [0, 5, 20])
@pytest.mark.parametrize("s_min", [0.01, 1, 1e3])
@pytest.mark.parametrize("z", [0.1, 1, 10])
@pytest.mark.parametrize("method", ["numerical_integration", "power_series"])
def test_conditional_likelihood_ratio_critical_value_function__(p, q, s_min, z, method):
    chi2p = scipy.stats.chi2.rvs(size=20000, df=p, random_state=0)
    chi2q = scipy.stats.chi2.rvs(size=20000, df=q, random_state=1) if q > 0 else 0
    D = np.sqrt((chi2p + chi2q - s_min) ** 2 + 4 * chi2p * s_min)
    Q = 1 / 2 * (chi2p + chi2q - s_min + D)
    p_value = np.mean(Q > z)

    assert np.isclose(
        p_value,
        conditional_likelihood_ratio_critical_value_function(
            p, q + p, s_min, z, method
        ),
        atol=1e-2,
    )


@pytest.mark.parametrize("p", [1, 20])
@pytest.mark.parametrize("q", [0, 20])
@pytest.mark.parametrize("s_min", [0.01, 1, 1e3])
@pytest.mark.parametrize("z", [0.1, 10])
@pytest.mark.parametrize("tol", [1e-2, 1e-6])
def test_conditional_likelihood_ratio_critical_value_function_some_by_method(
    p, q, s_min, z, tol
):
    assert np.isclose(
        conditional_likelihood_ratio_critical_value_function(
            p, q + p, s_min, z, "numerical_integration"
        ),
        conditional_likelihood_ratio_critical_value_function(
            p, q + p, s_min, z, "power_series"
        ),
        atol=2 * tol,
    )


@pytest.mark.parametrize(
    "n, k, mx, mw, mc, md, fit_intercept",
    [(100, 2, 1, 1, 0, 0, False), (100, 5, 2, 2, 1, 1, True)],
)
def test_conditional_likelihood_ratio_test_minimum_is_zero(
    n, k, mx, mw, mc, md, fit_intercept
):
    Z, X, y, C, W, D = simulate_gaussian_iv(n=n, mx=mx, mw=mw, k=k, mc=mc, md=md)

    liml = KClass(kappa="liml", fit_intercept=fit_intercept).fit(
        X=np.hstack([X, W]), y=y, Z=Z, C=np.hstack([D, C])
    )

    x0 = np.concatenate([liml.coef_[:mx], liml.coef_[(mx + mw) : (mx + mw + md)]])

    def f(x):
        return conditional_likelihood_ratio_test(Z, X, y, x, W, C, D, fit_intercept)[0]

    scipy.optimize.check_grad(func=f, grad=lambda _: 0, x0=x0)
    assert np.allclose(f(x0), 0, atol=1e-8)
