from functools import partial

import numpy as np
import pytest

from ivmodels.models.kclass import KClass
from ivmodels.quadric import Quadric
from ivmodels.simulate import simulate_gaussian_iv, simulate_guggenberger12
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    inverse_anderson_rubin_test,
    inverse_conditional_likelihood_ratio_test,
    inverse_lagrange_multiplier_test,
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
    (conditional_likelihood_ratio_test, inverse_conditional_likelihood_ratio_test),
    (pulse_test, inverse_pulse_test),
    (lagrange_multiplier_test, inverse_lagrange_multiplier_test),
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
    "n, mx, mw, k, u, mc",
    [(100, 1, 1, 2, 1, 3), (100, 1, 2, 5, 2, 0), (300, 2, 5, 10, 2, 2)],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_subvector_test_size(test, n, mx, mw, k, u, mc, fit_intercept):
    """Test that the test size is close to the nominal level."""
    n_seeds = 100
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        Z, X, y, C, W, beta = simulate_gaussian_iv(
            n=n,
            mx=mx,
            k=k,
            u=u,
            mw=mw,
            mc=mc,
            seed=seed,
            include_intercept=fit_intercept,
            return_beta=True,
        )

        _, p_values[seed] = test(Z, X, y, beta, W, C=C, fit_intercept=fit_intercept)

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
@pytest.mark.parametrize(
    "n, mx, mw, mc, k, u", [(100, 2, 5, 0, 10, 2), (100, 2, 2, 2, 5, 2)]
)
def test_subvector_test_size_low_rank(test, n, mx, mw, mc, k, u):
    """Test that the test size is close to the nominal level if Pi is low rank."""
    n_seeds = 200
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        delta_X = rng.normal(0, 1, (u, mx))
        delta_W = rng.normal(0, 1, (u, mw))
        delta_y = rng.normal(0, 1, (u, 1))

        beta_X = rng.normal(0, 0.1, (mx, 1))
        beta_C = rng.normal(0, 0.1, (mc, 1))

        gamma = rng.normal(0, 1, (mw, 1))
        Pi = rng.normal(0, 1, (k, 1)) @ rng.normal(0, 1, (1, mw + mx))
        Pi_X = Pi[:, :mx]
        Pi_W = Pi[:, mx:]

        U = rng.normal(0, 1, (n, u))

        C = rng.normal(0, 1, (n, mc))
        Z = rng.normal(0, 1, (n, k)) + C @ rng.normal(0, 0.1, (mc, k))

        X = Z @ Pi_X + U @ delta_X + rng.normal(0, 1, (n, mx))
        W = Z @ Pi_W + U @ delta_W + rng.normal(0, 1, (n, mw))
        y = X @ beta_X + C @ beta_C + W @ gamma + U @ delta_y + rng.normal(0, 1, (n, 1))

        _, p_values[seed] = test(Z, X, y, beta_X, W, fit_intercept=False)

    assert np.mean(p_values < 0.1) < 0.15


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
@pytest.mark.parametrize("n, k", [(100, 5), (10000, 30)])
def test_subvector_test_size_weak_instruments(test, n, k):
    """
    Test that the test size is close to the nominal level for weak instruments.

    This data generating process is proposed in :cite:p:`guggenberger2012asymptotic`.
    Here r = p = 1.
    """
    n_seeds = 200
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        Z, X, y, _, W, beta = simulate_guggenberger12(
            n, k=k, seed=seed, return_beta=True
        )
        _, p_values[seed] = test(Z, X, y, beta, W)

    assert np.mean(p_values < 0.1) < 0.15


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
    "n, mx, k, u, mc", [(100, 2, 2, 1, 3), (100, 2, 5, 2, 0), (200, 5, 10, 2, 2)]
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_test_size(test, n, mx, k, u, mc, fit_intercept):
    """Test that the test size is close to the nominal level."""
    n_seeds = 100
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        Z, X, y, C, _, beta = simulate_gaussian_iv(
            n=n,
            mx=mx,
            k=k,
            u=u,
            mc=mc,
            seed=seed,
            include_intercept=fit_intercept,
            return_beta=True,
        )

        _, p_values[seed] = test(Z, X, y, beta=beta, C=C, fit_intercept=fit_intercept)

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
    "n, mx, k, u, mc", [(100, 2, 2, 1, 3), (1000, 2, 5, 2, 0), (100, 5, 10, 2, 2)]
)
def test_test_size_weak_ivs(test, n, mx, k, u, mc):
    """Test that the test size is close to the nominal level for weak instruments."""
    n_seeds = 200
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        delta = rng.normal(0, 1, (u, mx))
        gamma = rng.normal(0, 1, (u, 1))

        beta = rng.normal(0, 0.1, (mx, 1))
        beta_C = rng.normal(0, 0.1, (mc, 1))

        Pi = rng.normal(0, 1, (k, mx)) / np.sqrt(n)

        U = rng.normal(0, 1, (n, u))

        Z = rng.normal(0, 1, (n, k))
        C = rng.normal(0, 1, (n, mc))

        X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, mx))
        y = X @ beta + C @ beta_C + U @ gamma + rng.normal(0, 1, (n, 1))

        _, p_values[seed] = test(Z, X, y, C=C, beta=beta)

    assert np.mean(p_values < 0.1) <= 0.2  # 4 stds above 0.05 for n_seeds = 100


@pytest.mark.parametrize(
    "test, inverse_test",
    [
        (pulse_test, inverse_pulse_test),
        (anderson_rubin_test, inverse_anderson_rubin_test),
        (f_anderson_rubin_test, inverse_f_anderson_rubin_test),
        (wald_test, inverse_wald_test),
        (lagrange_multiplier_test, inverse_lagrange_multiplier_test),
        (liml_wald_test, liml_inverse_wald_test),
        (likelihood_ratio_test, inverse_likelihood_ratio_test),
        (conditional_likelihood_ratio_test, inverse_conditional_likelihood_ratio_test),
    ],
)
@pytest.mark.parametrize(
    "data", [(100, 1, 2, 1, 3), (100, 2, 5, 2, 0), "guggenberger12"]
)
@pytest.mark.parametrize("p_value", [0.1, 0.01])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_test_round_trip(test, inverse_test, data, p_value, fit_intercept):
    """A test's p-value at the confidence set's boundary equals the nominal level."""
    if data == "guggenberger12":
        Z, X, y, C, _ = simulate_guggenberger12(n=10000, k=10, seed=0)
    else:
        n, mx, k, u, mc = data

        if test == lagrange_multiplier_test and mx > 1:
            pytest.skip("LM test inverse not implemented for mx > 1")
        if test == conditional_likelihood_ratio_test and mx > 1:
            pytest.skip("CLR test inverse not implemented for mx > 1")

        Z, X, y, C, _ = simulate_gaussian_iv(
            n=n, mx=mx, k=k, u=u, mc=mc, seed=0, include_intercept=fit_intercept
        )

    quadric = inverse_test(Z, X, y, C=C, alpha=p_value, fit_intercept=fit_intercept)
    boundary = quadric._boundary()

    if isinstance(quadric, Quadric):
        assert np.allclose(quadric(boundary), 0, atol=1e-7)

    p_values = np.zeros(boundary.shape[0])
    for idx, row in enumerate(boundary):
        p_values[idx] = test(Z, X, y, C=C, beta=row, fit_intercept=fit_intercept)[1]

    assert np.allclose(p_values, p_value, atol=1e-4)


@pytest.mark.parametrize(
    "test, inverse_test",
    [
        # (wald_test, inverse_wald_test),
        # (liml_wald_test, liml_inverse_wald_test),
        (anderson_rubin_test, inverse_anderson_rubin_test),
        # (lagrange_multiplier_test, inverse_lagrange_multiplier_test),
        # (f_anderson_rubin_test, inverse_f_anderson_rubin_test),
        # (likelihood_ratio_test, inverse_likelihood_ratio_test),
        # (conditional_likelihood_ratio_test, inverse_conditional_likelihood_ratio_test),
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        # (100, 1, 3, 1, 2, 3),
        # (100, 2, 5, 2, 3, 0),
        # (100, 1, 10, 5, None, 0),
        "guggenberger12",
    ],
)
@pytest.mark.parametrize("p_value", [0.1])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_subvector_round_trip(test, inverse_test, data, p_value, fit_intercept):
    """
    A test's p-value at the confidence set's boundary equals the nominal level.

    This time for subvector tests.
    """
    if data == "guggenberger12":
        Z, X, y, C, W = simulate_guggenberger12(n=10000, k=10, seed=0)
    else:
        n, mx, k, mw, u, mc = data

        if test == lagrange_multiplier_test and mx > 1:
            pytest.skip("LM test inverse not implemented for mx > 1")
        if test == conditional_likelihood_ratio_test and mx > 1:
            pytest.skip("CLR test inverse not implemented for mx > 1")

        Z, X, y, C, W = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u, mw=mw, mc=mc, seed=0)

    kwargs = {"Z": Z, "X": X, "y": y, "W": W, "C": C, "fit_intercept": fit_intercept}

    quadric = inverse_test(alpha=p_value, **kwargs)
    boundary = quadric._boundary()

    if isinstance(quadric, Quadric):
        assert np.allclose(quadric(boundary), 0, atol=1e-7)

    p_values = np.zeros(boundary.shape[0])
    for idx, row in enumerate(boundary):
        p_values[idx] = test(beta=row, **kwargs)[1]

    assert np.allclose(p_values, p_value, atol=1e-4)


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
@pytest.mark.parametrize("n, mx, k, u, mc", [(100, 2, 2, 1, 3), (100, 2, 5, 2, 0)])
def test_p_value_of_estimator(test, kappa, n, mx, k, u, mc):
    """The estimated coefficient should be in the confidence set with 95% coverage."""
    Z, X, y, C, _ = simulate_gaussian_iv(n=n, mx=mx, mc=mc, k=k, u=u)
    estimator = KClass(kappa=kappa).fit(X, y.flatten(), Z=Z, C=C)
    p_value = test(Z, X, y, beta=estimator.coef_[:mx], C=C)[1]
    assert p_value > 0.05


@pytest.mark.parametrize("test", [anderson_rubin_test, pulse_test])
@pytest.mark.parametrize("n, mx, k, u, mc", [(100, 2, 2, 1, 3), (100, 2, 5, 2, 0)])
def test_ar_test_monotonic_in_kappa(test, n, mx, k, u, mc):
    """AR(beta(kappa)) should be decreasing in kappa increasing towards kappa."""
    Z, X, Y, C, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u, mc=mc)
    Y = Y.flatten()
    kappas = np.linspace(0, KClass.ar_min(X, Y, Z) + 1, 10)
    models = [KClass(kappa=kappa).fit(X, Y, Z=Z) for kappa in kappas]
    statistics = [test(Z, X, Y, model.coef_)[0] for model in models]
    p_values = [test(Z, X, Y, model.coef_)[1] for model in models]

    assert np.all(statistics[:-1] >= statistics[1:])
    assert np.all(p_values[:-1] <= p_values[1:])


@pytest.mark.parametrize("n, mx, k, u, mc", [(100, 2, 2, 1, 3), (100, 2, 5, 2, 0)])
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
def test_inverse_test_sorted(inverse_test, n, mx, k, u, mc):
    """The volume of confidence sets should be increasing in the p-value."""
    Z, X, y, C, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u, mc=mc, seed=0)

    p_values = [0.5, 0.2, 0.1, 0.05]
    quadrics = [inverse_test(Z, X, y, C=C, alpha=p_value) for p_value in p_values]
    volumes = [quadric.volume() for quadric in quadrics]

    # Use <= instead of < as volumes can be infinite
    assert volumes[0] <= volumes[1] <= volumes[2] <= volumes[3]


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n, mx, mw, u, mc", [(100, 2, 0, 2, 2), (100, 2, 2, 2, 2)])
def test_ar_lm_clr_yield_same_result(n, mx, mw, u, mc, fit_intercept):
    """The AR, LM, and CLR tests should yield the same result if k = m."""
    Z, X, y, C, W = simulate_gaussian_iv(n=n, mx=mx, k=mx + mw, u=u, mw=mw, mc=mc)

    for seed in range(5):
        rng = np.random.RandomState(seed)
        beta = rng.normal(size=(mx, 1))

        ar = anderson_rubin_test(Z, X, y, beta, W, C=C, fit_intercept=fit_intercept)[1]
        lm = lagrange_multiplier_test(
            Z, X, y, beta, W, C=C, fit_intercept=fit_intercept
        )[1]
        clr = conditional_likelihood_ratio_test(
            Z, X, y, beta, W, C=C, fit_intercept=fit_intercept
        )[1]

        assert np.allclose(ar, lm)
        assert np.allclose(ar, clr)


# once with W=None, once with W!=None
@pytest.mark.parametrize(
    "test",
    [
        conditional_likelihood_ratio_test,
        pulse_test,
        lagrange_multiplier_test,
        anderson_rubin_test,
        wald_test,
        likelihood_ratio_test,
    ],
)
@pytest.mark.parametrize("n, mx, mw, u, mc", [(100, 2, 0, 2, 2), (100, 2, 2, 2, 2)])
def test_test_output_type(n, mx, mw, u, mc, test):
    if test == pulse_test and mw > 0:
        pytest.skip("Pulse test does not have a subvector version.")

    Z, X, y, C, W, beta = simulate_gaussian_iv(
        n=n, mx=mx, k=mx + mw, u=u, mw=mw, mc=mc, return_beta=True
    )

    statistic, p_values = test(Z, X, y, beta=beta, W=W, C=C)
    assert isinstance(statistic, float)
    assert isinstance(p_values, float)
