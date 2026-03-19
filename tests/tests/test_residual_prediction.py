from functools import partial

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import (
    inverse_weak_residual_prediction_test,
    residual_prediction_test,
    weak_residual_prediction_test,
)

rf = partial(RandomForestRegressor, n_estimators=50, max_depth=3)


@pytest.mark.parametrize(
    "test",
    [residual_prediction_test, weak_residual_prediction_test],
)
@pytest.mark.parametrize("n, mx, k, u, mc", [(200, 1, 2, 1, 1), (200, 2, 5, 2, 0)])
def test_test_output_type(test, n, mx, k, u, mc):
    """The tests should output a tuple of floats (statistic, p_value)."""
    Z, X, y, C, _, _, beta = simulate_gaussian_iv(
        n=n, mx=mx, k=k, u=u, mc=mc, return_beta=True
    )

    kwargs = {"nonlinear_model": rf(random_state=0)}
    if test == weak_residual_prediction_test:
        kwargs["beta"] = beta

    statistic, p_value = test(Z, X, y, C=C, **kwargs)

    assert isinstance(statistic, float)
    assert isinstance(p_value, float)


@pytest.mark.parametrize(
    "test",
    [residual_prediction_test, weak_residual_prediction_test],
)
@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("n, mx, k, u, mc", [(200, 1, 2, 1, 1), (300, 2, 5, 2, 0)])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_test_size(test, robust, n, mx, k, u, mc, fit_intercept):
    """Test that the test size is close to the nominal level under strong IVs."""
    n_seeds = 50
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        Z, X, y, C, _, _, beta = simulate_gaussian_iv(
            n=n,
            mx=mx,
            k=k,
            u=u,
            mc=mc,
            seed=seed,
            include_intercept=fit_intercept,
            return_beta=True,
        )

        kwargs = {
            "robust": robust,
            "fit_intercept": fit_intercept,
            "nonlinear_model": rf(random_state=seed),
            "seed": seed,
        }

        if test == weak_residual_prediction_test:
            kwargs["beta"] = beta

        _, p_values[seed] = test(Z, X, y, C=C, **kwargs)

    # Allow some finite-sample slack for ML-based tests
    assert np.mean(p_values <= 0.1) <= 0.2


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("instrument_strength", [1.0, 0.1, 0.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_weak_residual_prediction_test_size(instrument_strength, robust, fit_intercept):
    """Test that the weak test maintains nominal size even when IVs are completely weak."""
    rng = np.random.default_rng(42)
    n_seeds = 50
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        Z = rng.normal(size=(200, 3))
        U = rng.normal(size=(200, 1))

        X = (
            instrument_strength * (Z[:, 0:1] + Z[:, 1:2])
            + U
            + rng.normal(size=(200, 1))
        )

        beta_true_2d = np.array([[1.5]])

        noise = rng.normal(size=(200, 1))
        if robust:
            noise *= Z[:, 0:1] ** 2 + 0.1

        y = X @ beta_true_2d + U + noise

        _, p_values[seed] = weak_residual_prediction_test(
            Z=Z,
            X=X,
            y=y.flatten(),
            beta=np.array([1.5]),  # Pass the 1D version to the test
            robust=robust,
            nonlinear_model=rf(random_state=seed),
            fit_intercept=fit_intercept,
            seed=seed,
        )

    assert np.mean(p_values < 0.05) <= 0.15


def test_weak_residual_prediction_test_rejects_false_beta():
    """Test that the weak residual prediction test rejects an incorrect beta."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 3))
    U = rng.normal(size=(500, 1))
    X = Z[:, 0:1] - Z[:, 1:2] + U + rng.normal(size=(500, 1))

    beta_true_2d = np.array([[2.0]])
    y = X @ beta_true_2d + U + rng.normal(size=(500, 1))

    beta_false = np.array([0.0])

    _, p_value = weak_residual_prediction_test(
        Z=Z,
        X=X,
        y=y.flatten(),
        beta=beta_false,
        nonlinear_model=rf(random_state=0),
    )

    assert p_value < 0.05


def test_inverse_weak_residual_prediction_empty_cs():
    """If the structural model is fundamentally misspecified, the CS should be empty."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 2))
    X = Z[:, 0:1] + rng.normal(size=(300, 1))

    beta_true_2d = np.array([[1.0]])
    y = X @ beta_true_2d + 2.0 * (Z[:, 0:1] ** 2) + rng.normal(size=(300, 1))

    cs = inverse_weak_residual_prediction_test(
        Z=Z,
        X=X,
        y=y.flatten(),
        alpha=0.05,
        nonlinear_model=rf(random_state=0),
        tol=1e-2,
        max_eval=50,
    )

    assert len(cs.boundaries) == 0


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("n, k, mc", [(200, 2, 1), (250, 3, 0)])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_test_round_trip_1d(n, k, mc, fit_intercept, robust, alpha):
    """A test's p-value at the confidence set's boundary changes sign appropriately."""
    Z, X, y, C, _, _ = simulate_gaussian_iv(
        n=n, mx=1, k=k, mc=mc, seed=0, include_intercept=fit_intercept
    )

    kwargs = dict(
        Z=Z,
        X=X,
        y=y.flatten(),
        C=C,
        robust=robust,
        fit_intercept=fit_intercept,
        seed=0,
    )

    cs = inverse_weak_residual_prediction_test(
        **kwargs,
        nonlinear_model=rf(random_state=0),
        alpha=alpha,
        tol=1e-2,
        max_eval=50,
    )

    # Use a large step size because RF prediction surfaces are jagged step-functions
    step = 0.20

    for left, right in cs.boundaries:
        if not np.isinf(left):
            assert (
                weak_residual_prediction_test(
                    beta=np.array([left + step]),
                    nonlinear_model=rf(random_state=0),
                    **kwargs,
                )[1]
                > alpha
            )
            assert (
                weak_residual_prediction_test(
                    beta=np.array([left - step]),
                    nonlinear_model=rf(random_state=0),
                    **kwargs,
                )[1]
                < alpha
            )

        if not np.isinf(right):
            assert (
                weak_residual_prediction_test(
                    beta=np.array([right + step]),
                    nonlinear_model=rf(random_state=0),
                    **kwargs,
                )[1]
                < alpha
            )
            assert (
                weak_residual_prediction_test(
                    beta=np.array([right - step]),
                    nonlinear_model=rf(random_state=0),
                    **kwargs,
                )[1]
                > alpha
            )

        left_check = left if not np.isinf(left) else -1e6
        right_check = right if not np.isinf(right) else 1e6
        mid = (left_check + right_check) / 2
        assert (
            weak_residual_prediction_test(
                beta=np.array([mid]),
                nonlinear_model=rf(random_state=0),
                **kwargs,
            )[1]
            > alpha
        )
