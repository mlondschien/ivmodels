import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from anchor_regression import AnchorRegression
from anchor_regression.testing import simulate_iv
from anchor_regression.utils import anderson_rubin_test, pulse_test


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("p", [1, 5])
def test_pulse_test_tsls(discrete, p):
    X, Y, A = simulate_iv(discrete=discrete, p=p, seed=0, shift=0)
    Xhat = LinearRegression(fit_intercept=True).fit(A, X).predict(A)
    tsls = LinearRegression(fit_intercept=True).fit(Xhat, Y)
    residuals = Y - tsls.predict(X)
    _, p_value = pulse_test(A, residuals)
    assert p_value > 0.05


@pytest.mark.parametrize("test", [anderson_rubin_test, pulse_test])
@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("p", [1, 5])
def test_pulse_anchor(test, discrete, p):
    X, Y, A = simulate_iv(discrete=discrete, p=p, seed=0, shift=0)
    gammas = [0.1, 1, 2, 4, 8, 16, 32, 64]
    ars = [AnchorRegression(gamma=gamma).fit(X, Y, A) for gamma in gammas]
    statistics = [test(A, Y - ar.predict(X))[0] for ar in ars]
    assert np.all(statistics[:-1] >= statistics[1:])  # AR test should be monotonic
    p_values = [test(A, Y - ar.predict(X))[1] for ar in ars]
    assert np.all(p_values[:-1] <= p_values[1:])
