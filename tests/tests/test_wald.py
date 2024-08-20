import numpy as np
import pytest
from linearmodels.iv import IV2SLS, IVLIML

from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import wald_test


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("estimator", ["liml", "tsls"])
@pytest.mark.parametrize(
    "n, mx, mw, md, k, u",
    [(100, 2, 0, 0, 2, 1), (100, 2, 0, 1, 5, 2), (100, 1, 2, 2, 4, 1)],
)
@pytest.mark.parametrize("robust", [True, False])
def test_compare_wald_tests_with_linearmodels(
    n, mx, mw, md, k, u, estimator, fit_intercept, robust
):
    Z, X, y, _, W, D = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u, mw=mw, md=md)

    XW = np.hstack([X, W])

    if fit_intercept:
        D = np.hstack([np.ones((n, 1)), D])

    if estimator == "liml":
        linearmodel = IVLIML(y, D, XW, Z)
    elif estimator == "tsls":
        linearmodel = IV2SLS(y, D, XW, Z)

    cov_type = "robust" if robust else "unadjusted"
    results = linearmodel.fit(cov_type=cov_type, debiased=True)

    mat = np.eye(mx + mw + md + fit_intercept)[
        int(fit_intercept) : (mx + md + fit_intercept), :
    ]
    lm_wald_result = results.wald_test(mat, np.zeros(mx + md))

    ivmodels_wald_result = wald_test(
        Z,
        X,
        y,
        beta=np.zeros(mx + md),
        estimator=estimator,
        fit_intercept=fit_intercept,
        W=W,
        D=D[:, fit_intercept:],
        robust=robust,
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
