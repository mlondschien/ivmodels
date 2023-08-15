import numpy as np
import pytest
import scipy
from sklearn.linear_model import LinearRegression

from ivmodels import AnchorRegression
from ivmodels.linear_models import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import (
    anderson_rubin_test,
    asymptotic_confidence_interval,
    bounded_inverse_anderson_rubin,
    inverse_anderson_rubin,
    pulse_test,
)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_pulse_test_tsls(n, p, q, u):
    A, X, Y = simulate_gaussian_iv(n, p, q, u)
    Y = Y.flatten()
    Xhat = LinearRegression(fit_intercept=True).fit(A, X).predict(A)
    tsls = LinearRegression(fit_intercept=True).fit(Xhat, Y)
    residuals = Y - tsls.predict(X)
    _, p_value = pulse_test(A, residuals)
    assert p_value > 0.05


@pytest.mark.parametrize("test", [anderson_rubin_test, pulse_test])
@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_pulse_anchor(test, n, p, q, u):
    A, X, Y = simulate_gaussian_iv(n, p, q, u)
    Y = Y.flatten()
    gammas = [0.1, 1, 2, 4, 8, 16, 32, 64]
    ars = [AnchorRegression(gamma=gamma).fit(X, Y, A) for gamma in gammas]
    statistics = [test(A, Y.flatten() - ar.predict(X))[0] for ar in ars]
    p_values = [test(A, Y.flatten() - ar.predict(X))[1] for ar in ars]

    assert np.all(statistics[:-1] >= statistics[1:])  # AR test should be monotonic
    assert np.all(p_values[:-1] <= p_values[1:])


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_anderson_rubin_test(n, p, q, u, seed=0):
    rng = np.random.RandomState(0)

    delta = rng.normal(0, 1, (u, p))
    gamma = rng.normal(0, 1, (u, 1))
    beta = rng.normal(0, 0.1, (p, 1))
    Pi = rng.normal(0, 1, (q, p))

    n_seeds = 200
    vals = np.zeros(n_seeds)

    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        U = rng.normal(0, 1, (n, u))
        Z = rng.normal(0, 1, (n, q))
        X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
        y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

        Z = Z - Z.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y - y.mean()

        vals[s] = anderson_rubin_test(Z, y - X @ beta)[1]

    assert scipy.stats.kstest(vals, scipy.stats.uniform(loc=0.0, scale=1).cdf)[1] > 0.05


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_inverse_anderson_rubin_sorted(n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u, seed=0)

    p_values = [0.5, 0.2, 0.1, 0.05]
    quadrics = [inverse_anderson_rubin(Z, X, y, p_value) for p_value in p_values]
    volumes = [quadric.volume() for quadric in quadrics]

    # Use <= instead of < as volumes can be infinite
    assert volumes[0] <= volumes[1] <= volumes[2] <= volumes[3]


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("p_value", [0.5, 0.2, 0.1, 0.05])
def test_inverse_anderson_rubin_round_trip(n, p, q, u, p_value):
    Z, X, y = simulate_gaussian_iv(n, p, q, u, seed=0)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y.flatten() - y.mean()

    quadric = inverse_anderson_rubin(Z, X, y, p_value)
    boundary = quadric._boundary()

    assert np.allclose(quadric(boundary), 0, atol=1e-7)

    p_values = np.zeros(boundary.shape[0])
    for idx, row in enumerate(boundary):
        residuals = y - X @ row
        p_values[idx] = anderson_rubin_test(Z, residuals)[1]

    assert np.allclose(p_values, p_value, atol=1e-8)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("seed", [0, 1])
def test_inverse_anderson_rubin_below_above(n, p, q, u, seed):
    rng = np.random.RandomState(seed)

    delta = rng.normal(0, 1, (u, p))
    gamma = rng.normal(0, 1, (u, 1))

    beta = rng.normal(0, 0.1, (p, 1))
    Pi = rng.normal(0, 1, (q, p))

    U = rng.normal(0, 1, (n, u))

    Z = rng.normal(0, 1, (n, q))
    X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
    y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

    X = X - X.mean(axis=0)
    y = y - y.mean()

    beta_hat = rng.normal(0, 0.1, (p, 1))
    _, p_value = anderson_rubin_test(Z, y - X @ beta_hat)
    below = inverse_anderson_rubin(Z, X, y, p_value * 0.999)
    above = inverse_anderson_rubin(Z, X, y, p_value + (1 - p_value) * 0.001)
    assert below(beta_hat.reshape(1, -1)) < 0
    assert above(beta_hat.reshape(1, -1)) > 0


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2), (100, 2, 1, 2)])
def test_bounded_inverse_anderson_rubin_p_value(n, p, q, u):
    """
    Test `bounded_inverse_anderson_rubin` against `anderson_rubin_test`.

    `bounded_inverse_anderson_rubin` should return the largest p-value s.t. the
    corresponding confidence set is bounded. Test that this is the case by computing
    the volume of the confidence sets after increasing / decreasing the p-value by 0.1%.
    """
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


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 1, 1, 1)])
@pytest.mark.parametrize("alpha", [0.2, 0.05])
def test_asymptotic_confidence_set(alpha, n, p, q, u, seed=0):
    rng = np.random.RandomState(seed)

    delta = rng.normal(0, 1, (u, p))
    gamma = rng.normal(0, 1, (u, 1))
    beta = rng.normal(0, 0.1, (p, 1))
    Pi = rng.normal(0, 1, (q, p))

    n_seeds = 200
    vals = np.zeros(n_seeds)

    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        U = rng.normal(0, 1, (n, u))
        Z = rng.normal(0, 1, (n, q))
        X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
        y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

        Z = Z - Z.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y.flatten() - y.mean()

        beta_liml = KClass(kappa="liml").fit(X, y, Z).coef_.reshape(-1, 1)

        quadric = asymptotic_confidence_interval(Z, X, y, beta_liml, alpha)
        vals[s] = quadric(beta.flatten())

    assert np.abs(np.mean(vals > 0) - alpha) < 0.5 * alpha
