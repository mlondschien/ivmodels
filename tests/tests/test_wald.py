import numpy as np
import pytest
from linearmodels.iv import IV2SLS, IVLIML

from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import wald_test


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("estimator", ["liml", "tsls"])
@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_compare_wald_tests_with_linearmodels(n, mx, k, u, estimator, fit_intercept):
    Z, X, y, _, _ = simulate_gaussian_iv(n, mx, k, u)

    if fit_intercept:
        intercept = np.ones((n, 1))
    else:
        intercept = None

    if estimator == "liml":
        linearmodel = IVLIML(y, intercept, X, Z)
    elif estimator == "tsls":
        linearmodel = IV2SLS(y, intercept, X, Z)

    results = linearmodel.fit(cov_type="unadjusted", debiased=True)
    lm_wald_result = results.wald_test(
        np.eye(mx + int(fit_intercept))[int(fit_intercept) :, :], np.zeros(mx)
    )
    ivmodels_wald_result = wald_test(
        Z, X, y, beta=np.zeros(mx), estimator=estimator, fit_intercept=fit_intercept
    )

    np.testing.assert_allclose(
        lm_wald_result.stat,
        ivmodels_wald_result[0],
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        lm_wald_result.pval,
        ivmodels_wald_result[1],
        rtol=1e-6,
    )
