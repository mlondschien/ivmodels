from functools import partial

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from ivmodels.simulate import simulate_gaussian_iv, simulate_guggenberger12
from ivmodels.tests import (
    inverse_weak_residual_prediction_test,
    residual_prediction_test,
    weak_residual_prediction_test,
)

rf = partial(RandomForestRegressor, n_estimators=50, max_depth=2)


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
def test_zero_predictions(test):
    Z, X, y, C, _, _, beta = simulate_gaussian_iv(
        n=100, mx=1, k=2, return_beta=True, seed=42
    )

    kwargs = {"nonlinear_model": DummyRegressor(strategy="constant", constant=0.0)}
    if test == weak_residual_prediction_test:
        kwargs["beta"] = beta

    stat, pval = test(Z, X, y, C=C, **kwargs)

    assert np.isclose(stat, 0.0)
    assert np.isclose(pval, 0.5)


@pytest.mark.parametrize(
    "test",
    [residual_prediction_test, weak_residual_prediction_test],
)
@pytest.mark.parametrize(
    "robust, n, mx, k, u, mc, fit_intercept",
    [
        (False, 200, 1, 2, 1, 1, True),
        (True, 300, 2, 5, 2, 0, False),
        (False, 200, 1, 2, 1, 1, False),
    ],
)
def test_test_size(test, robust, n, mx, k, u, mc, fit_intercept):
    """Test that the test size is close to the nominal level under strong IVs."""
    n_seeds = 25
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
def test_weak_residual_prediction_test_size_weak_iv(robust):
    """Test that the weak test maintains nominal size even when IVs are weak."""
    n_seeds = 50
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        # h11=0 forces the first-stage to be 0 (completely weak instruments)
        Z, X, y, _, _, _, beta = simulate_guggenberger12(
            n=200, k=3, h11=0.0, seed=seed, return_beta=True
        )

        _, p_values[seed] = weak_residual_prediction_test(
            Z=Z,
            X=X,
            y=y,
            beta=beta,
            robust=robust,
            nonlinear_model=rf(random_state=seed),
            seed=seed,
        )

    assert np.mean(p_values < 0.05) <= 0.15


def test_weak_residual_prediction_test_rejects_false_beta():
    """Test that the weak residual prediction test rejects an incorrect beta."""
    Z, X, y, _, _, _, beta = simulate_gaussian_iv(
        n=500, mx=1, k=3, return_beta=True, seed=0
    )

    # Test against a wildly incorrect beta
    beta_false = beta + 10.0

    _, p_value = weak_residual_prediction_test(
        Z=Z,
        X=X,
        y=y,
        beta=beta_false,
        nonlinear_model=rf(random_state=0),
    )

    assert p_value < 0.05


def test_inverse_weak_residual_prediction_empty_cs():
    """If the structural model is fundamentally misspecified, the CS should be empty."""
    Z, X, y, _, _, _ = simulate_gaussian_iv(n=300, mx=1, k=2, seed=0)

    # Fundamental misspecification: Y depends directly on Z^2 (violates exclusion restriction)
    y += 5.0 * (Z[:, 0] ** 2)

    cs = inverse_weak_residual_prediction_test(
        Z=Z,
        X=X,
        y=y,
        alpha=0.05,
        nonlinear_model=rf(random_state=0),
        tol=1e-2,
        max_eval=50,
    )

    # Evaluates the "if f(res.x) >= 0: return ConfidenceSet([])" branch
    assert len(cs.boundaries) == 0


@pytest.mark.parametrize(
    "robust, n, k, mc, fit_intercept, alpha",
    [
        (False, 200, 2, 1, True, 0.1),
        (True, 300, 3, 0, False, 0.05),
        (False, 200, 2, 1, True, 0.2),
    ],
)
def test_test_round_trip_1d(n, k, mc, fit_intercept, robust, alpha):
    """A test's p-value at the confidence set's boundary changes sign appropriately."""
    Z, X, y, C, _, _ = simulate_gaussian_iv(
        n=n, mx=1, k=k, mc=mc, seed=0, include_intercept=fit_intercept
    )

    kwargs = dict(
        Z=Z,
        X=X,
        y=y,
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
        max_eval=25,
    )

    for left, right in cs.boundaries:
        if not np.isinf(left):
            assert (
                weak_residual_prediction_test(
                    beta=np.array([left + 0.01]),
                    nonlinear_model=rf(random_state=0),
                    **kwargs,
                )[1]
                > alpha
            )
            assert (
                weak_residual_prediction_test(
                    beta=np.array([left - 0.01]),
                    nonlinear_model=rf(random_state=0),
                    **kwargs,
                )[1]
                < alpha
            )

        if not np.isinf(right):
            assert (
                weak_residual_prediction_test(
                    beta=np.array([right + 0.01]),
                    nonlinear_model=rf(random_state=0),
                    **kwargs,
                )[1]
                < alpha
            )
            assert (
                weak_residual_prediction_test(
                    beta=np.array([right - 0.01]),
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
