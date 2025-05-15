import numpy as np
import pytest
import scipy.stats
from sklearn.ensemble import RandomForestRegressor

from ivmodels.tests import residual_prediction_test


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
            nonlinear_model=RandomForestRegressor(n_estimators=20, random_state=0),
            fit_intercept=fit_intercept,
            train_fraction=0.4,
            seed=0,
        )

    assert (
        scipy.stats.kstest(p_values, scipy.stats.uniform(loc=0.0, scale=1.0).cdf).pvalue
        > 0.05
    )


def test_residual_prediction_test_rejects():
    rng = np.random.default_rng(0)

    Pi = rng.normal(size=(5, 2))
    beta = rng.normal(size=(2, 1))

    Z = rng.normal(size=(200, 5))
    X = Z @ Pi + rng.normal(size=(200, 2))
    y = X @ beta + Z[:, 0:1] ** 2 + rng.normal(size=(200, 1))

    _, p_value = residual_prediction_test(
        Z=Z,
        X=X,
        y=y,
        nonlinear_model=RandomForestRegressor(),
        fit_intercept=False,
        train_fraction=None,
        seed=0,
    )
    assert p_value < 0.05
