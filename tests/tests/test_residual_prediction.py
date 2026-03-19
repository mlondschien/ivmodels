import numpy as np
import pytest
import scipy.stats
from sklearn.ensemble import RandomForestRegressor

from ivmodels.tests import (
    inverse_weak_residual_prediction_test,
    residual_prediction_test,
    weak_residual_prediction_test,
)


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize(
    "n, k, mx, mc, fit_intercept",
    [(500, 3, 3, 1, True), (500, 3, 1, 1, False), (500, 15, 5, 5, True)],
)
def test_residual_prediction_test(n, k, mx, mc, fit_intercept, robust):
    rng = np.random.default_rng(0)

    Pi = rng.normal(size=(k, mx))
    beta = rng.normal(size=(mx, 1))
    gamma = rng.normal(size=(mx + 1, mx))
    Pi_CZ = rng.normal(size=(mc, k))
    Pi_CX = rng.normal(size=(mc, mx))
    Pi_Cy = rng.normal(size=(mc, 1))

    n_seeds = 20
    statistics = np.zeros(n_seeds)
    p_values = np.zeros(n_seeds)

    for idx in range(n_seeds):
        C = rng.normal(size=(n, mc))
        Z = rng.normal(size=(n, k)) + C @ Pi_CZ
        U = rng.normal(size=(n, mx + 1))
        X = Z @ Pi + U @ gamma + C @ Pi_CX + rng.normal(size=(n, mx))
        X[:, 0] += Z[:, 0] ** 2  # allow for nonlinearity Z -> X
        noise = rng.normal(size=(n, 1))
        if robust:
            noise *= Z[:, 0:1] ** 2
        y = X @ beta + U[:, 0:1] + np.sin(U[:, 0:1]) + C @ Pi_Cy + noise

        statistics[idx], p_values[idx] = residual_prediction_test(
            Z=Z,
            X=X,
            y=y,
            C=C,
            robust=robust,
            nonlinear_model=RandomForestRegressor(n_estimators=50, random_state=0),
            fit_intercept=fit_intercept,
            train_fraction=0.4,
            seed=0,
        )

    assert (
        scipy.stats.kstest(p_values, scipy.stats.uniform(loc=0.0, scale=1.0).cdf).pvalue
        > 0.05
    )


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize(
    "n, k, mc, fit_intercept",
    [(200, 3, 1, True), (200, 3, 0, False)],
)
@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_weak_residual_prediction_round_trip(n, k, mc, fit_intercept, robust, alpha):
    """The p-value of weak_residual_prediction_test changes sign across CI boundaries."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(n, k))
    X = Z[:, :1] + rng.normal(size=(n, 1))
    C = rng.normal(size=(n, mc)) if mc > 0 else None
    y = X[:, 0] + rng.normal(size=n)
    if mc > 0:
        y += C[:, 0]

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
        nonlinear_model=RandomForestRegressor(n_estimators=100, random_state=0),
        alpha=alpha,
    )

    for left, right in cs.boundaries:
        if not np.isinf(left):
            assert (
                weak_residual_prediction_test(
                    beta=np.array([left + 0.01]),
                    nonlinear_model=RandomForestRegressor(
                        n_estimators=50, random_state=0
                    ),
                    **kwargs,
                )[1]
                > alpha
            )
            assert (
                weak_residual_prediction_test(
                    beta=np.array([left - 0.01]),
                    nonlinear_model=RandomForestRegressor(
                        n_estimators=50, random_state=0
                    ),
                    **kwargs,
                )[1]
                < alpha
            )

        if not np.isinf(right):
            assert (
                weak_residual_prediction_test(
                    beta=np.array([right + 0.01]),
                    nonlinear_model=RandomForestRegressor(
                        n_estimators=50, random_state=0
                    ),
                    **kwargs,
                )[1]
                < alpha
            )
            assert (
                weak_residual_prediction_test(
                    beta=np.array([right - 0.01]),
                    nonlinear_model=RandomForestRegressor(
                        n_estimators=50, random_state=0
                    ),
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
                nonlinear_model=RandomForestRegressor(n_estimators=50, random_state=0),
                **kwargs,
            )[1]
            > alpha
        )


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("pi_scale", [1.0, 0.1, 0.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_weak_residual_prediction_test_size(pi_scale, robust, fit_intercept):
    rng = np.random.default_rng(42)
    n_seeds = 50
    p_values = np.zeros(n_seeds)

    for seed in range(n_seeds):
        Z = rng.normal(size=(200, 3))
        U = rng.normal(size=(200, 1))

        X = pi_scale * (Z[:, 0:1] + Z[:, 1:2]) + U + rng.normal(size=(200, 1))

        beta_true_2d = np.array([[1.5]])

        noise = rng.normal(size=(200, 1))
        if robust:
            noise *= Z[:, 0:1] ** 2 + 0.1

        y = X @ beta_true_2d + U + noise

        _, p_values[seed] = weak_residual_prediction_test(
            Z=Z,
            X=X,
            y=y.flatten(),
            beta=np.array([1.5]),  # Pass the 1D version to the function
            robust=robust,
            nonlinear_model=RandomForestRegressor(
                n_estimators=15, max_depth=3, random_state=seed
            ),
            fit_intercept=fit_intercept,
            seed=seed,
        )

    assert np.mean(p_values < 0.05) <= 0.15


def test_weak_residual_prediction_test_rejects_false_beta():
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
        nonlinear_model=RandomForestRegressor(n_estimators=20, random_state=0),
    )

    assert p_value < 0.05


def test_inverse_weak_residual_prediction_empty_cs():
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
        nonlinear_model=RandomForestRegressor(
            n_estimators=15, max_depth=3, random_state=0
        ),
        tol=1e-2,
        max_eval=50,
    )

    assert len(cs.boundaries) == 0
