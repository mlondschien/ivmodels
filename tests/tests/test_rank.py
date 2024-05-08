import re

import numpy as np
import pytest

from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import inverse_anderson_rubin_test, rank_test


@pytest.mark.parametrize("n, mx, k, u", [(40, 2, 2, 1), (40, 2, 5, 2), (40, 1, 2, 2)])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_bounded_inverse_anderson_rubin_p_value(n, mx, k, u, fit_intercept):
    Z, X, y, _, _ = simulate_gaussian_iv(
        n, mx, k, u, seed=0, include_intercept=fit_intercept
    )

    p_value = rank_test(Z, X, fit_intercept=fit_intercept)[1]
    assert 0.999 > p_value > 1e-12

    quad_below = inverse_anderson_rubin_test(
        Z, X, y, p_value * 0.999, fit_intercept=fit_intercept
    )
    quad_above = inverse_anderson_rubin_test(
        Z, X, y, p_value * 1.001, fit_intercept=fit_intercept
    )

    assert np.isinf(quad_below.volume())
    assert np.isfinite(quad_above.volume())


def test_rank_test_raises():
    Z, X, y, _, _ = simulate_gaussian_iv(n=10, k=1, mx=2, u=0)

    with pytest.raises(ValueError, match=re.escape("Need `Z.shape[1] >= X.shape[1]`.")):
        _ = rank_test(Z, X)
