import numpy as np
import pytest
import scipy
from sklearn.ensemble import RandomForestRegressor

from ivmodels.tests import scheidegger_test


@pytest.mark.parametrize("n, k, mx", [(100, 3, 3), (100, 3, 1), (100, 20, 5)])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_scheidegger_test(n, k, mx, fit_intercept):
    rng = np.random.default_rng(0)

    Pi = rng.normal(size=(k, mx))
    beta = rng.normal(size=(mx, 1))

    n_seeds = 100
    statistics = np.zeros(n_seeds)
    p_values = np.zeros(n_seeds)

    for idx in range(n_seeds):
        Z = rng.normal(size=(n, k))
        X = Z @ Pi + rng.normal(size=(n, mx))
        y = X @ beta + rng.normal(size=(n, 1))

        statistics[idx], p_values[idx] = scheidegger_test(
            Z=Z,
            X=X,
            y=y,
            nonlinear_model=RandomForestRegressor(),
            kappa="tsls",
            fit_intercept=fit_intercept,
            train_fraction=None,
            clipping_quantile=0.8,
            seed=0,
        )

    assert (
        scipy.stats.kstest(p_values, scipy.stats.uniform(loc=0.0, scale=1.0).cdf).pvalue
        > 0.05
    )
