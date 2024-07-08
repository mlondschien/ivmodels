import pytest

from ivmodels.models.kclass import KClass
from ivmodels.simulate import simulate_gaussian_iv


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n, mx, k, mc", [(1000, 2, 2, 0), (1000, 4, 10, 1)])
def test_kclass_summary(n, mx, k, mc, fit_intercept):
    Z, X, y, C, _ = simulate_gaussian_iv(
        n=n, mx=mx, k=k, mc=mc, mw=0, include_intercept=fit_intercept
    )

    kclass = KClass(kappa="liml", fit_intercept=fit_intercept)
    kclass.fit(X, y, Z, C)

    _ = kclass.summary(X, y, Z, C, test="AR")
