import numpy as np
import pytest
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.tests.anderson_rubin import (
    anderson_rubin_test,
    inverse_anderson_rubin_test,
    more_powerful_subvector_anderson_rubin_critical_value_function,
)
from ivmodels.utils import proj


@pytest.mark.parametrize(
    "k, alpha, hat_kappa_1_cvs",
    [
        (1, 0.1, [[1.1, 0.7], [2.1, 1.2], [3.3, 1.6], [5.0, 2.0], [8.8, 2.4]]),
        (
            1,
            0.01,
            [[1.0, 0.9], [2.0, 1.8], [3.0, 2.6], [5.0, 3.9], [10.0, 5.6], [16.6, 6.2]],
        ),
        (
            5,
            0.1,
            [
                [1.0, 0.9],
                [2.0, 1.8],
                [3.0, 2.6],
                [6.0, 4.8],
                [11.0, 7.2],
                [15.6, 8.2],
                [34.8, 9.0],
            ],
        ),
    ],
)
def testmore_powerful_subvector_anderson_rubin_critical_value_function(
    k, alpha, hat_kappa_1_cvs
):
    """Compare to tables 3, 7 in Guggenberger (2019)."""
    for hat_kappa_1, cv in hat_kappa_1_cvs:
        # Through rounding to one decimal place
        assert (
            more_powerful_subvector_anderson_rubin_critical_value_function(
                cv, hat_kappa_1, k, mW=0
            )
            <= alpha
        )
        assert (
            more_powerful_subvector_anderson_rubin_critical_value_function(
                cv - 0.1, hat_kappa_1, k, mW=0
            )
            >= alpha
        )


@pytest.mark.parametrize("k", [1, 5, 20])
@pytest.mark.parametrize("hat_kappa_1", [0.1, 1, 5, 100])
def test_more_powerful_sAR_critical_value_function_integrates_to_one(k, hat_kappa_1):
    assert np.isclose(
        more_powerful_subvector_anderson_rubin_critical_value_function(
            hat_kappa_1, hat_kappa_1, k, mW=0
        ),
        0,
        atol=2e-4,
    )


@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
@pytest.mark.parametrize("n, k, mx, u", [(100, 1, 1, 1), (100, 20, 5, 5)])
def test_inverse_anderson_rubin_confidence_set_alternative_formulation(
    alpha, n, k, mx, u
):
    rng = np.random.RandomState(0)

    delta_X = rng.normal(0, 1, (u, mx))
    delta_y = rng.normal(0, 1, (u, 1))

    beta = rng.normal(0, 0.1, (mx, 1))
    Pi_X = rng.normal(0, 1, (k, mx))

    U = rng.normal(0, 1, (n, u))

    Z = rng.normal(0, 1, (n, k))
    X = Z @ Pi_X + U @ delta_X + rng.normal(0, 1, (n, mx))
    y = X @ beta + U @ delta_y + rng.normal(0, 1, (n, 1))

    X = X - X.mean(axis=0)
    y = y - y.mean()

    inverse_ar = inverse_anderson_rubin_test(Z, X, y, alpha=alpha)
    kappa_alpha = 1 + scipy.stats.f(dfn=k, dfd=n - k).ppf(1 - alpha) * k / (n - k)
    kclass_kappa_alpha = KClass(kappa=kappa_alpha).fit(X=X, y=y, Z=Z)

    assert np.allclose(inverse_ar.center, kclass_kappa_alpha.coef_, rtol=1e-8)

    residuals = y.flatten() - X @ kclass_kappa_alpha.coef_
    residuals_orth = residuals - proj(Z, residuals)
    sigma_hat_sq = (residuals_orth.T @ residuals_orth) / (n - k)
    ar = anderson_rubin_test(Z, X, y, beta=kclass_kappa_alpha.coef_)[0]
    A = (kappa_alpha * proj(Z, X) + (1 - kappa_alpha) * X).T @ X

    assert np.allclose(
        A
        / (
            -sigma_hat_sq
            * (k * scipy.stats.f(dfn=k, dfd=n - k).ppf(1 - alpha) * k - ar)
        ),
        inverse_ar.A / inverse_ar.c_standardized,
    )
