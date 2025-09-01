import numpy as np
import pytest
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests.conditional_likelihood_ratio import (
    _newton_minimal_root,
    conditional_likelihood_ratio_critical_value_function,
    conditional_likelihood_ratio_test,
)


@pytest.mark.parametrize(
    "q, lambdas", [([1.0, 1.0], [1.0]), ([5.0, 1.0, 2.0], [10.0, 100.0])]
)
def test__newton_minimal_root(q, lambdas):
    # make sure it actually finds a root.
    q_sum = np.sum(q)
    q = np.array(q)[1:]
    root = _newton_minimal_root(q_sum, q, np.array(lambdas), tol=1e-8, num_iter=100)
    g = (root - q_sum) - np.sum(
        [lambdas[i] * q[i] / (root - lambdas[i]) for i in range(len(lambdas))]
    )
    np.testing.assert_allclose(g, 0, atol=1e-8)


@pytest.mark.parametrize("p", [1, 5, 20])
@pytest.mark.parametrize("q", [0, 20])
@pytest.mark.parametrize("s_min", [0.01, 1, 1e3])
@pytest.mark.parametrize("z", [0.1, 10])
@pytest.mark.parametrize("tol", [1e-2, 1e-4])
def test_conditional_likelihood_ratio_critical_value_function_tol(p, q, s_min, z, tol):
    approx = conditional_likelihood_ratio_critical_value_function(
        k=q + p,
        mx=p,
        md=0,
        lambdas=np.array([s_min] * p),
        z=z,
        tol=tol,
        critical_values="moreira2003conditional",
    )
    exact = conditional_likelihood_ratio_critical_value_function(
        k=q + p,
        mx=p,
        md=0,
        lambdas=np.array([s_min] * p),
        z=z,
        tol=1e-8,
        critical_values="moreira2003conditional",
    )
    assert np.isclose(approx, exact, atol=3 * tol)


@pytest.mark.parametrize("p", [1, 5, 20])
@pytest.mark.parametrize("q", [0, 20])
@pytest.mark.parametrize(
    "critical_values", ["moreira2003conditional", "londschien2025exact"]
)
def test_conditional_likelihood_ratio_critical_value_function_equal_to_chi2(
    p, q, critical_values
):
    for z in np.linspace(0, 2 * (p + q), 10):
        pval = 1 - scipy.stats.chi2(p + q).cdf(z)
        assert np.isclose(
            conditional_likelihood_ratio_critical_value_function(
                k=q + p,
                mx=p,
                md=0,
                lambdas=1e-8 * np.ones(p),
                z=z,
                critical_values=critical_values,
                num_samples=10_000,
            ),
            pval,
            atol=max(1e-4, np.sqrt((1 - pval) * pval / 10_000) * 3),
        )


@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("q", [0, 5, 20])
@pytest.mark.parametrize("s_min", [0.01, 1, 1e3])
@pytest.mark.parametrize("z", [0.1, 1, 10])
def test_conditional_likelihood_ratio_critical_value_function__(p, q, s_min, z):
    chi2p = scipy.stats.chi2.rvs(size=20000, df=p, random_state=0)
    chi2q = scipy.stats.chi2.rvs(size=20000, df=q, random_state=1) if q > 0 else 0
    D = np.sqrt((chi2p + chi2q - s_min) ** 2 + 4 * chi2p * s_min)
    Q = 1 / 2 * (chi2p + chi2q - s_min + D)
    p_value = np.mean(Q > z)

    assert np.isclose(
        p_value,
        conditional_likelihood_ratio_critical_value_function(
            k=q + p,
            mx=p,
            md=0,
            lambdas=s_min * np.ones(p),
            z=z,
            critical_values="moreira2003conditional",
        ),
        atol=1e-2,
    )


@pytest.mark.parametrize("p", [1, 20])
@pytest.mark.parametrize("q", [0, 20])
@pytest.mark.parametrize("s_min", [0.01, 1, 1e3])
@pytest.mark.parametrize("z", [0.1, 10])
@pytest.mark.parametrize("tol", [1e-2, 1e-6])
def test_conditional_likelihood_ratio_critical_value_function_same_by_method(
    p, q, s_min, z, tol
):
    p1 = conditional_likelihood_ratio_critical_value_function(
        k=q + p,
        mx=p,
        md=0,
        lambdas=np.array([s_min] * p),
        z=z,
        critical_values="moreira2003conditional",
        tol=tol,
    )
    p2 = conditional_likelihood_ratio_critical_value_function(
        k=q + p,
        mx=p,
        md=0,
        lambdas=np.array([s_min] * p),
        z=z,
        critical_values="londschien2025exact",
        tol=tol,
    )
    np.testing.assert_allclose(p1, p2, atol=np.sqrt(p2 * (1 - p2) / 10_000) * 3 + tol)


@pytest.mark.parametrize(
    "n, k, mx, mw, mc, md, fit_intercept",
    [(100, 2, 1, 1, 0, 0, False), (100, 5, 2, 2, 1, 0, True)],
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
        return conditional_likelihood_ratio_test(
            Z, X, y, x, W, C, D, fit_intercept, critical_values="moreira2003conditional"
        )[0]

    scipy.optimize.check_grad(func=f, grad=lambda _: 0, x0=x0)
    assert np.allclose(f(x0), 0, atol=1e-8)
