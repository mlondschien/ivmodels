import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from anchor_regression import AnchorRegression
from anchor_regression.simulate import simulate_gaussian_iv
from anchor_regression.tests import (
    anderson_rubin_test,
    bounded_inverse_anderson_rubin,
    inverse_anderson_rubin,
    pulse_test,
)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_pulse_test_tsls(n, p, q, u):
    A, X, Y = simulate_gaussian_iv(n, p, q, u)
    Xhat = LinearRegression(fit_intercept=True).fit(A, X).predict(A)
    tsls = LinearRegression(fit_intercept=True).fit(Xhat, Y)
    residuals = Y - tsls.predict(X)
    _, p_value = pulse_test(A, residuals)
    assert p_value > 0.05


@pytest.mark.parametrize("test", [anderson_rubin_test, pulse_test])
@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_pulse_anchor(test, n, p, q, u):
    A, X, Y = simulate_gaussian_iv(n, p, q, u)
    gammas = [0.1, 1, 2, 4, 8, 16, 32, 64]
    ars = [AnchorRegression(gamma=gamma).fit(X, Y, A) for gamma in gammas]
    statistics = [test(A, Y - ar.predict(X))[0] for ar in ars]
    p_values = [test(A, Y - ar.predict(X))[1] for ar in ars]

    assert np.all(statistics[:-1] <= statistics[1:])  # AR test should be monotonic
    assert np.all(p_values[:-1] >= p_values[1:])


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_inverse_anderson_rubin(n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u, seed=0)

    p_values = [0.5, 0.2, 0.1, 0.05]
    quadrics = [inverse_anderson_rubin(Z, X, y, p_value) for p_value in p_values]
    volumes = [quadric.volume() for quadric in quadrics]

    # Use <= instead of < as volumes can be infinite
    assert volumes[0] <= volumes[1] <= volumes[2] <= volumes[3]


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2), (100, 2, 1, 2)])
def test_bounded_inverse_anderson_rubin_p_value(n, p, q, u):
    Z, X, Y = simulate_gaussian_iv(n, p, q, u, seed=0)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    p_value = bounded_inverse_anderson_rubin(Z, X)

    if p > q:
        assert np.isclose(p_value, 1)
    else:
        quad_below = inverse_anderson_rubin(Z, X, Y, p_value * 0.999)
        quad_above = inverse_anderson_rubin(Z, X, Y, p_value * 1.001)

        assert np.isinf(quad_below.volume())
        assert np.isfinite(quad_above.volume())
