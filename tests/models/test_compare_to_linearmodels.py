import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, IVLIML

from ivmodels import KClass
from ivmodels.simulate import simulate_gaussian_iv


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n, mx, k, r, u", [(100, 2, 2, 1, 1), (100, 2, 5, 0, 2)])
@pytest.mark.parametrize("kappa", ["liml", "tsls"])
def test_compare_against_linearmodels(fit_intercept, n, mx, k, r, u, kappa):
    Z, X, y, C, _ = simulate_gaussian_iv(n, mx, k, u, r=r)

    kclass = KClass(kappa=kappa, fit_intercept=fit_intercept)
    kclass.fit(X, y, Z=Z, C=C)

    if fit_intercept:
        C = sm.add_constant(C)

    if kappa == "liml":
        linearmodel = IVLIML(y, C, X, Z)
    elif kappa == "tsls":
        linearmodel = IV2SLS(y, C, X, Z)
    else:
        raise ValueError

    results = linearmodel.fit(cov_type="unadjusted")

    np.testing.assert_allclose(kclass.coef_[:mx], results.params[-mx:], rtol=1e-5)
    np.testing.assert_allclose(
        kclass.coef_[mx:], results.params[-(mx + r) : -mx], rtol=1e-5
    )

    if fit_intercept:
        np.testing.assert_allclose(kclass.intercept_, results.params[0], rtol=1e-5)
