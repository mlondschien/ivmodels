import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from anchor_regression import LinearAnchorRegression
from anchor_regression.testing import simulate_iv
from anchor_regression.utils import anderson_rubin_test


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("dim_y", [1, 2])
def test_anderson_rubin_test_tsls(discrete, p, dim_y):
    X, Y, A = simulate_iv(discrete=discrete, p=p, seed=0, shift=0, dim_y=dim_y)
    Xhat = LinearRegression(fit_intercept=True).fit(A, X).predict(A)
    tsls = LinearRegression(fit_intercept=True).fit(Xhat, Y)
    residuals = Y - tsls.predict(X)
    p_value, _ = anderson_rubin_test(A, residuals)
    assert p_value < 1e-3


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("dim_y", [1, 2])
def test_anderson_rubin_rest_anchor(discrete, p, dim_y):
    X, Y, A = simulate_iv(discrete=discrete, p=p, seed=0, shift=0, dim_y=dim_y)
    gammas = [0, 1, 2, 4, 8, 16, 32, 64]
    ars = [LinearAnchorRegression(gamma=gamma).fit(X, Y, A) for gamma in gammas]

    statistics = [anderson_rubin_test(A, Y - ar.predict(X))[0] for ar in ars]
    assert np.all(statistics[:-1] >= statistics[1:])  # AR test should be monotonic
