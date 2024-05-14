from functools import partial

import numpy as np
import pytest
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    inverse_anderson_rubin_test,
    inverse_likelihood_ratio_test,
    inverse_pulse_test,
    inverse_wald_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    pulse_test,
    wald_test,
)

liml_wald_test = partial(wald_test, estimator="liml")
liml_inverse_wald_test = partial(inverse_wald_test, estimator="liml")
guggenberger_anderson_rubin_test = partial(
    anderson_rubin_test, critical_values="guggenberger2019more"
)
f_anderson_rubin_test = partial(anderson_rubin_test, critical_values="f")
inverse_f_anderson_rubin_test = partial(
    inverse_anderson_rubin_test, critical_values="f"
)

TEST_PAIRS = [
    (conditional_likelihood_ratio_test, None),
    (pulse_test, inverse_pulse_test),
    (lagrange_multiplier_test, None),
    (anderson_rubin_test, inverse_anderson_rubin_test),
    (f_anderson_rubin_test, inverse_f_anderson_rubin_test),
    (wald_test, inverse_wald_test),
    (likelihood_ratio_test, inverse_likelihood_ratio_test),
]


# The Pulse and the LM tests don't have subvector versions.
@pytest.mark.parametrize(
    "test",
    [
        anderson_rubin_test,
        guggenberger_anderson_rubin_test,
        f_anderson_rubin_test,
        lagrange_multiplier_test,
        wald_test,
        liml_wald_test,
        likelihood_ratio_test,
        conditional_likelihood_ratio_test,
    ],
)
@pytest.mark.parametrize(
    "n, mx, mw, k, u", [(100, 1, 1, 2, 1), (100, 1, 2, 5, 2), (300, 2, 5, 10, 2)]
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_subvector_test_size(test, n, mx, mw, k, u, fit_intercept):
    """Test that the test size is close to the nominal level."""
    n_seeds = 100
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        Z, X, y, _, W, beta = simulate_gaussian_iv(
            n=n,
            mx=mx,
            k=k,
            u=u,
            mw=mw,
            seed=seed,
            include_intercept=fit_intercept,
            return_beta=True,
        )

        _, p_values[seed] = test(Z, X, y, beta, W, fit_intercept=fit_intercept)

    assert np.mean(p_values < 0.1) <= 0.2


# The Pulse test does not have subvector a version.
@pytest.mark.parametrize(
    "test",
    [
        anderson_rubin_test,
        guggenberger_anderson_rubin_test,
        f_anderson_rubin_test,
        lagrange_multiplier_test,
        wald_test,
        liml_wald_test,
        likelihood_ratio_test,
        conditional_likelihood_ratio_test,
    ],
)
@pytest.mark.parametrize("n, p, r, q, u", [(100, 2, 5, 10, 2)])
def test_subvector_test_size_low_rank(test, n, p, r, q, u):
    """Test that the test size is close to the nominal level if Pi is low rank."""
    n_seeds = 200
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        delta_X = rng.normal(0, 1, (u, p))
        delta_W = rng.normal(0, 1, (u, r))
        delta_y = rng.normal(0, 1, (u, 1))

        beta = rng.normal(0, 0.1, (p, 1))
        gamma = rng.normal(0, 1, (r, 1))
        Pi = rng.normal(0, 1, (q, 1)) @ rng.normal(0, 1, (1, r + p))
        Pi_X = Pi[:, :p]
        Pi_W = Pi[:, p:]

        U = rng.normal(0, 1, (n, u))

        Z = rng.normal(0, 1, (n, q))
        X = Z @ Pi_X + U @ delta_X + rng.normal(0, 1, (n, p))
        W = Z @ Pi_W + U @ delta_W + rng.normal(0, 1, (n, r))
        y = X @ beta + W @ gamma + U @ delta_y + rng.normal(0, 1, (n, 1))

        _, p_values[seed] = test(Z, X, y, beta, W, fit_intercept=False)

    assert np.mean(p_values < 0.05) < 0.07  # 4 stds above 0.05 for n_seeds = 100


# The Pulse does not have a subvector version. The Wald and LR tests are
# not valid for weak instruments.
@pytest.mark.parametrize(
    "test",
    [
        anderson_rubin_test,
        guggenberger_anderson_rubin_test,
        f_anderson_rubin_test,
        conditional_likelihood_ratio_test,
        lagrange_multiplier_test,
    ],
)
@pytest.mark.parametrize("n, q", [(100, 5), (100, 30)])
def test_subvector_test_size_weak_instruments(test, n, q):
    """
    Test that the test size is close to the nominal level for weak instruments.

    This data generating process is proposed in :cite:p:`guggenberger2012asymptotic`.
    Here r = p = 1.
    """
    n_seeds = 200
    p_values = np.zeros(n_seeds)

    p = 1
    r = 1

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        # Make sure that sqrt(n) || Pi_W | ~ 1, sqrt(n) || Pi_X | ~ 100, and
        # sqrt(n) < Pi_W, Pi_X> ~ 0.95
        Pi_X = rng.normal(0, 1, (q, p))
        Pi_W = rng.normal(0, 1, (q, r))

        Pi_W = np.sqrt(0.05) * Pi_W + np.sqrt(0.95) * Pi_X
        Pi_W = Pi_W / np.linalg.norm(Pi_W) / np.sqrt(n)
        Pi_X = 100 * Pi_X / np.linalg.norm(Pi_X) / np.sqrt(n)

        # Equal to [eps, V_X, V_W]. Have Cov(eps, V_X) = 0, Cov(eps, V_w = 0.95), and
        # Cov(V_X, V_W) = 0.3.
        noise = scipy.stats.multivariate_normal.rvs(
            cov=np.array([[1, 0, 0.95], [0, 1, 0.3], [0.95, 0.3, 1]]),
            size=n,
            random_state=seed,
        )

        Z = rng.normal(0, 1, (n, q)) + 1

        X = Z @ Pi_X + noise[:, 1:2]
        W = Z @ Pi_W + noise[:, 2:3]
        y = X + W + noise[:, 0:1]

        # True beta
        beta = np.array([[1]])

        _, p_values[seed] = test(Z, X, y, beta, W)

    assert np.mean(p_values < 0.05) < 0.07  # 4 stds above 0.05 for n_seeds = 100


@pytest.mark.parametrize(
    "test",
    [
        pulse_test,
        lagrange_multiplier_test,
        anderson_rubin_test,
        f_anderson_rubin_test,
        wald_test,
        liml_wald_test,
        likelihood_ratio_test,
        conditional_likelihood_ratio_test,
    ],
)
@pytest.mark.parametrize(
    "n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2), (200, 5, 10, 2)]
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_test_size(test, n, mx, k, u, fit_intercept):
    """Test that the test size is close to the nominal level."""
    n_seeds = 100
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        Z, X, y, _, W, beta = simulate_gaussian_iv(
            n=n,
            mx=mx,
            k=k,
            u=u,
            r=0,
            seed=seed,
            include_intercept=fit_intercept,
            return_beta=True,
        )

        _, p_values[seed] = test(Z, X, y, beta, fit_intercept=fit_intercept)

    assert np.mean(p_values <= 0.1) <= 0.2


# The wald, and likelihood ratio tests are not valid for weak instruments
@pytest.mark.parametrize(
    "test",
    [
        lagrange_multiplier_test,
        anderson_rubin_test,
        f_anderson_rubin_test,
        pulse_test,
        conditional_likelihood_ratio_test,
    ],
)
@pytest.mark.parametrize(
    "n, mx, k, u", [(100, 2, 2, 1), (1000, 2, 5, 2), (100, 5, 10, 2)]
)
def test_test_size_weak_ivs(test, n, mx, k, u):
    """Test that the test size is close to the nominal level for weak instruments."""
    n_seeds = 200
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        delta = rng.normal(0, 1, (u, mx))
        gamma = rng.normal(0, 1, (u, 1))

        beta = rng.normal(0, 0.1, (mx, 1))
        Pi = rng.normal(0, 1, (k, mx)) / np.sqrt(n)

        U = rng.normal(0, 1, (n, u))

        Z = rng.normal(0, 1, (n, k))
        X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, mx))
        y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

        _, p_values[seed] = test(Z, X, y, beta)

    assert np.mean(p_values < 0.05) <= 0.075  # 4 stds above 0.05 for n_seeds = 100


@pytest.mark.parametrize(
    "test, inverse_test",
    [
        (pulse_test, inverse_pulse_test),
        (anderson_rubin_test, inverse_anderson_rubin_test),
        (f_anderson_rubin_test, inverse_f_anderson_rubin_test),
        (wald_test, inverse_wald_test),
        (liml_wald_test, liml_inverse_wald_test),
        (likelihood_ratio_test, inverse_likelihood_ratio_test),
    ],
)
@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("p_value", [0.1, 0.01])
def test_test_round_trip(test, inverse_test, n, mx, k, u, p_value):
    """A test's p-value at the confidence set's boundary equals the nominal level."""
    Z, X, y, _, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u, seed=0)

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


@pytest.mark.parametrize(
    "test, inverse_test",
    [
        (wald_test, inverse_wald_test),
        (liml_wald_test, liml_inverse_wald_test),
        (anderson_rubin_test, inverse_anderson_rubin_test),
        (f_anderson_rubin_test, inverse_f_anderson_rubin_test),
        (likelihood_ratio_test, inverse_likelihood_ratio_test),
    ],
)
@pytest.mark.parametrize("n, mx, k, mw, u", [(100, 2, 3, 1, 2), (100, 2, 5, 2, 3)])
@pytest.mark.parametrize("p_value", [0.1])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_subvector_round_trip(
    test, inverse_test, n, mx, k, u, mw, p_value, fit_intercept
):
    """
    A test's p-value at the confidence set's boundary equals the nominal level.

    This time for subvector tests.
    """
    Z, X, y, _, W = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u, mw=mw, seed=0)

    quadric = inverse_test(Z, X, y, p_value, W=W, fit_intercept=fit_intercept)
    boundary = quadric._boundary()

    assert np.allclose(quadric(boundary), 0, atol=1e-7)

    p_values = np.zeros(boundary.shape[0])
    for idx, row in enumerate(boundary):
        p_values[idx] = test(Z, X, y, beta=row, W=W, fit_intercept=fit_intercept)[1]

    assert np.allclose(p_values, p_value, atol=1e-8)


@pytest.mark.parametrize(
    "test",
    [
        pulse_test,
        lagrange_multiplier_test,
        anderson_rubin_test,
        f_anderson_rubin_test,
        wald_test,
        liml_wald_test,
        likelihood_ratio_test,
        conditional_likelihood_ratio_test,
    ],
)
@pytest.mark.parametrize("kappa", ["liml", "tsls"])
@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_p_value_of_estimator(test, kappa, n, mx, k, u):
    """The estimated coefficient should be in the confidence set with 95% coverage."""
    Z, X, y, _, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u)
    estimator = KClass(kappa=kappa).fit(X, y.flatten(), Z=Z)
    p_value = test(Z, X, y, estimator.coef_)[1]
    assert p_value > 0.05


@pytest.mark.parametrize("test", [anderson_rubin_test, pulse_test])
@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_ar_test_monotonic_in_kappa(test, n, mx, k, u):
    """AR(beta(kappa)) should be decreasing in kappa increasing towards kappa."""
    Z, X, Y, _, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u)
    Y = Y.flatten()
    kappas = np.linspace(0, KClass.ar_min(X, Y, Z) + 1, 10)
    models = [KClass(kappa=kappa).fit(X, Y, Z=Z) for kappa in kappas]
    statistics = [test(Z, X, Y, model.coef_)[0] for model in models]
    p_values = [test(Z, X, Y, model.coef_)[1] for model in models]

    assert np.all(statistics[:-1] >= statistics[1:])
    assert np.all(p_values[:-1] <= p_values[1:])


@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize(
    "inverse_test",
    [
        inverse_pulse_test,
        inverse_anderson_rubin_test,
        inverse_f_anderson_rubin_test,
        inverse_wald_test,
        liml_inverse_wald_test,
        inverse_likelihood_ratio_test,
    ],
)
def test_inverse_test_sorted(inverse_test, n, mx, k, u):
    """The volume of confidence sets should be increasing in the p-value."""
    Z, X, y, _, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u, seed=0)

    p_values = [0.5, 0.2, 0.1, 0.05]
    quadrics = [inverse_test(Z, X, y, p_value) for p_value in p_values]
    volumes = [quadric.volume() for quadric in quadrics]

    # Use <= instead of < as volumes can be infinite
    assert volumes[0] <= volumes[1] <= volumes[2] <= volumes[3]
