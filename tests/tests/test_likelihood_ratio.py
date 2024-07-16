import numpy as np
import pytest
import scipy
import scipy.optimize

from ivmodels import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import inverse_likelihood_ratio_test, likelihood_ratio_test
from ivmodels.utils import proj


@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
@pytest.mark.parametrize("n, k, mx, u", [(100, 2, 1, 1), (100, 20, 5, 5)])
def test_inverse_likelihood_ratio_confidence_set_alternative_formulation(
    alpha, n, k, mx, u
):
    Z, X, y, _, _, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u)

    kappa_ = KClass(kappa="liml", fit_intercept=False).fit(X=X, y=y, Z=Z).kappa_

    inverse_ar = inverse_likelihood_ratio_test(
        Z, X, y, alpha=alpha, fit_intercept=False
    )
    kappa_alpha = kappa_ + scipy.stats.chi2(df=mx).ppf(1 - alpha) / (n - k)
    kclass_kappa_alpha = KClass(kappa=kappa_alpha, fit_intercept=False).fit(
        X=X, y=y, Z=Z
    )
    assert np.allclose(inverse_ar.center, kclass_kappa_alpha.coef_, rtol=1e-6)

    A = (kappa_alpha * proj(Z, X) + (1 - kappa_alpha) * X).T @ X

    assert np.allclose(
        A,
        inverse_ar.A,
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    "n, k, mx, mw, mc, md, fit_intercept",
    [(100, 2, 1, 1, 0, 0, False), (100, 5, 2, 2, 1, 1, True)],
)
def test_likelihood_ratio_test_minimum_is_zero(n, k, mx, mw, mc, md, fit_intercept):
    Z, X, y, C, W, D = simulate_gaussian_iv(n=n, mx=mx, mw=mw, k=k, mc=mc, md=md)

    liml = KClass(kappa="liml", fit_intercept=fit_intercept).fit(
        X=np.hstack([X, W]), y=y, Z=Z, C=np.hstack([D, C])
    )

    x0 = np.concatenate([liml.coef_[:mx], liml.coef_[(mx + mw) : (mx + mw + md)]])

    def f(x):
        return likelihood_ratio_test(Z, X, y, x, W, C, D, fit_intercept)[0]

    scipy.optimize.check_grad(func=f, grad=lambda _: 0, x0=x0)
    assert np.allclose(f(x0), 0, atol=1e-8)
