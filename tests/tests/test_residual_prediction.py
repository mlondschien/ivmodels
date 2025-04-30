import numpy as np
import pytest
import scipy
from sklearn.ensemble import RandomForestRegressor

from ivmodels.tests import residual_prediction_test


@pytest.mark.parametrize(
    "n, k, mx, fit_intercept",
    [(200, 3, 3, True), (200, 3, 1, False), (200, 15, 5, False)],
)
def test_residual_prediction_test(n, k, mx, fit_intercept):
    rng = np.random.default_rng(0)

    Pi = rng.normal(size=(k, mx))
    beta = rng.normal(size=(mx, 1))
    gamma = rng.normal(size=(mx + 1, mx))

    n_seeds = 50
    statistics = np.zeros(n_seeds)
    p_values = np.zeros(n_seeds)

    for idx in range(n_seeds):
        Z = rng.normal(size=(n, k))
        U = rng.normal(size=(n, mx + 1))
        X = Z @ Pi + U @ gamma + rng.normal(size=(n, mx))
        X[:, 0] += Z[:, 0] ** 2  # allow for nonlinearity Z -> X
        y = X @ beta + U[:, 0:1] + U[:, 0:1] ** 3 + rng.normal(size=(n, 1))

        statistics[idx], p_values[idx] = residual_prediction_test(
            Z=Z,
            X=X,
            y=y,
            nonlinear_model=RandomForestRegressor(n_estimators=20),
            kappa="tsls",
            fit_intercept=fit_intercept,
            train_fraction=0.6,
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
        kappa="tsls",
        fit_intercept=False,
        train_fraction=None,
        seed=0,
    )
    assert p_value < 0.05
