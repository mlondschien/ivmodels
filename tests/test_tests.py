import numpy as np
import pytest

from ivmodels.kclass import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import (
    anderson_rubin_test,
    bounded_inverse_anderson_rubin,
    inverse_anderson_rubin_test,
    inverse_likelihood_ratio_test,
    inverse_pulse_test,
    inverse_wald_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    pulse_test,
    wald_test,
)

TEST_PAIRS = [
    (anderson_rubin_test, inverse_anderson_rubin_test),
    (pulse_test, inverse_pulse_test),
    (lagrange_multiplier_test, None),
    (likelihood_ratio_test, inverse_likelihood_ratio_test),
    (wald_test, inverse_wald_test),
]

# def test_projected_test():
#     n=1000
#     p = 2
#     q = 3
#     u = 3

#     Z, X, y = simulate_gaussian_iv(n, p, q, u, seed=0)

#     Z = Z - Z.mean(axis=0)
#     X = X - X.mean(axis=0)
#     y = y.flatten() - y.mean()

#     W = np.zeros((n, 0))


#     res1=projected_anderson_rubin_test(Z, X, W, y, np.zeros((p, 1)))
#     res2=anderson_rubin_test(Z, X, y,  np.zeros((p, 1)))
#     breakpoint()


@pytest.mark.parametrize("test", [pair[0] for pair in TEST_PAIRS])
@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_test_size(test, n, p, q, u):
    """Test that the test size is close to the nominal level."""
    n_seeds = 200
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        delta = rng.normal(0, 1, (u, p))
        gamma = rng.normal(0, 1, (u, 1))

        beta = rng.normal(0, 0.1, (p, 1))
        Pi = rng.normal(0, 1, (q, p))

        U = rng.normal(0, 1, (n, u))

        Z = rng.normal(0, 1, (n, q))
        X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
        y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

        _, p_values[seed] = test(Z, X, y, beta)

    assert np.mean(p_values < 0.05) < 0.07  # 4 stds above 0.05 for n_seeds = 100


# The wald and likelihood ratio tests are not valid for weak instruments
@pytest.mark.parametrize("test", [pair[0] for pair in TEST_PAIRS[:-2]])
@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (1000, 2, 5, 2)])
def test_test_size_weak_ivs(test, n, p, q, u):
    """Test that the test size is close to the nominal level for weak instruments."""
    n_seeds = 1000
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        delta = rng.normal(0, 1, (u, p))
        gamma = rng.normal(0, 1, (u, 1))

        beta = rng.normal(0, 0.1, (p, 1))
        Pi = rng.normal(0, 1, (q, p)) / np.sqrt(n)

        U = rng.normal(0, 1, (n, u))

        Z = rng.normal(0, 1, (n, q))
        X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
        y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

        _, p_values[seed] = test(Z, X, y, beta)

    assert np.mean(p_values < 0.05) < 0.07  # 4 stds above 0.05 for n_seeds = 100


@pytest.mark.parametrize(
    "test, inverse_test", [(p[0], p[1]) for p in TEST_PAIRS if p[1] is not None]
)
@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("p_value", [0.1, 0.01])
def test_test_round_trip(test, inverse_test, n, p, q, u, p_value):
    """A test's p-value at the confidence set's boundary equals the nominal level."""
    Z, X, y = simulate_gaussian_iv(n, p, q, u, seed=0)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y.flatten() - y.mean()

    quadric = inverse_test(Z, X, y, p_value)
    boundary = quadric._boundary()

    assert np.allclose(quadric(boundary), 0, atol=1e-7)

    p_values = np.zeros(boundary.shape[0])
    for idx, row in enumerate(boundary):
        p_values[idx] = test(Z, X, y, beta=row)[1]

    assert np.allclose(p_values, p_value, atol=1e-8)


@pytest.mark.parametrize("test", [pair[0] for pair in TEST_PAIRS])
@pytest.mark.parametrize("kappa", ["liml", "tsls"])
@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_p_value_of_estimator(test, kappa, n, p, q, u):
    """The estimated coefficient should be in the confidence set with 95% coverage."""
    Z, X, y = simulate_gaussian_iv(n, p, q, u)
    estimator = KClass(kappa=kappa).fit(X, y.flatten(), Z=Z)
    p_value = test(Z, X, y, estimator.coef_)[1]
    assert p_value > 0.05


@pytest.mark.parametrize("test", [anderson_rubin_test, pulse_test])
@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_ar_test_monotonic_in_kappa(test, n, p, q, u):
    """AR(beta(kappa)) should be decreasing in kappa increasing towards 1."""
    A, X, Y = simulate_gaussian_iv(n, p, q, u)
    Y = Y.flatten()
    kappas = np.linspace(0, 1, 10)
    models = [KClass(kappa=kappa).fit(X, Y, Z=A) for kappa in kappas]
    statistics = [test(A, X, Y, model.coef_)[0] for model in models]
    p_values = [test(A, X, Y, model.coef_)[1] for model in models]

    assert np.all(statistics[:-1] >= statistics[1:])
    assert np.all(p_values[:-1] <= p_values[1:])


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize(
    "inverse_test", [pair[1] for pair in TEST_PAIRS if pair[1] is not None]
)
def test_inverse_test_sorted(inverse_test, n, p, q, u):
    """The volume of confidence sets should be increasing in the p-value."""
    Z, X, y = simulate_gaussian_iv(n, p, q, u, seed=0)

    p_values = [0.5, 0.2, 0.1, 0.05]
    quadrics = [inverse_test(Z, X, y, p_value) for p_value in p_values]
    volumes = [quadric.volume() for quadric in quadrics]

    # Use <= instead of < as volumes can be infinite
    assert volumes[0] <= volumes[1] <= volumes[2] <= volumes[3]


# @pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
# @pytest.mark.parametrize("seed", [0, 1])
# def test_inverse_anderson_rubin_below_above(n, p, q, u, seed):
#     rng = np.random.RandomState(seed)

#     delta = rng.normal(0, 1, (u, p))
#     gamma = rng.normal(0, 1, (u, 1))

#     beta = rng.normal(0, 0.1, (p, 1))
#     Pi = rng.normal(0, 1, (q, p))

#     U = rng.normal(0, 1, (n, u))

#     Z = rng.normal(0, 1, (n, q))
#     X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
#     y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

#     X = X - X.mean(axis=0)
#     y = y - y.mean()

#     beta_hat = rng.normal(0, 0.1, (p, 1))
#     _, p_value = anderson_rubin_test(Z, X, y.flatten(), beta_hat)
#     below = inverse_anderson_rubin_test(Z, X, y, p_value * 0.999)
#     above = inverse_anderson_rubin_test(Z, X, y, p_value + (1 - p_value) * 0.001)
#     assert below(beta_hat.reshape(1, -1)) < 0
#     assert above(beta_hat.reshape(1, -1)) > 0


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
        quad_below = inverse_anderson_rubin_test(Z, X, Y, p_value * 0.999)
        quad_above = inverse_anderson_rubin_test(Z, X, Y, p_value * 1.001)

        assert np.isinf(quad_below.volume())
        assert np.isfinite(quad_above.volume())
