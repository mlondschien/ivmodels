import pytest

from ivmodels.models.kclass import KClass
from ivmodels.simulate import simulate_gaussian_iv


@pytest.mark.parametrize(
    "test",
    [
        "anderson-rubin",
        "wald",
        "likelihood-ratio",
        "lagrange multiplier",
        "conditional likelihood-ratio",
    ],
)
@pytest.mark.parametrize(
    "n, mx, k, mc, fit_intercept", [(100, 2, 3, 0, False), (100, 4, 10, 2, True)]
)
def test_kclass_summary(test, n, mx, k, mc, fit_intercept):
    Z, X, y, C, _, _ = simulate_gaussian_iv(
        n=n, mx=mx, k=k, mc=mc, mw=0, include_intercept=fit_intercept
    )

    kclass = KClass(kappa="liml", fit_intercept=fit_intercept)
    kclass.fit(X, y, Z, C)

    _ = str(kclass.summary(X, y, Z, C, test=test))
