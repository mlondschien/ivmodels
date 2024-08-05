import numpy as np
import pytest

from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import anderson_rubin_test, j_test


@pytest.mark.parametrize(
    "n, k, mx, mc", [(100, 3, 1, 0), (100, 3, 2, 1), (100, 20, 5, 5)]
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_inverse_anderson_rubin_confidence_set_alternative_formulation(
    n, k, mx, mc, fit_intercept
):
    Z, X, y, C, _, _ = simulate_gaussian_iv(
        n=n, mx=mx, mc=mc, k=k, include_intercept=fit_intercept
    )
    j_stat, j_pvalue = j_test(Z, X, y, C, fit_intercept=fit_intercept)
    ar_stat, ar_pvalue = anderson_rubin_test(
        Z=Z,
        W=X,
        X=np.empty((n, 0)),
        y=y,
        C=C,
        beta=np.empty((0,)),
        fit_intercept=fit_intercept,
    )

    assert np.allclose(j_stat, ar_stat * (k - mx))
    assert np.allclose(j_pvalue, ar_pvalue)
