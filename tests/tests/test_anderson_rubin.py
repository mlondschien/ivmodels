import numpy as np
import pytest
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests.anderson_rubin import (
    anderson_rubin_test,
    inverse_anderson_rubin_test,
    more_powerful_subvector_anderson_rubin_critical_value_function,
)
from ivmodels.utils import oproj, proj


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
def test_more_powerful_subvector_anderson_rubin_critical_value_function(
    k, alpha, hat_kappa_1_cvs
):
    """Compare to tables 3, 7 in Guggenberger (2019)."""
    for hat_kappa_1, cv in hat_kappa_1_cvs:
        # Through rounding to one decimal place
        assert (
            more_powerful_subvector_anderson_rubin_critical_value_function(
                cv, hat_kappa_1, k, mw=0
            )
            <= alpha
        )
        assert (
            more_powerful_subvector_anderson_rubin_critical_value_function(
                cv - 0.1, hat_kappa_1, k, mw=0
            )
            >= alpha
        )


@pytest.mark.parametrize("k", [1, 5, 20])
@pytest.mark.parametrize("hat_kappa_1", [0.1, 1, 5, 100])
def test_more_powerful_sAR_critical_value_function_integrates_to_one(k, hat_kappa_1):
    assert np.isclose(
        more_powerful_subvector_anderson_rubin_critical_value_function(
            hat_kappa_1, hat_kappa_1, k, mw=0
        ),
        0,
        atol=2e-4,
    )


@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
@pytest.mark.parametrize(
    "n, k, mx, mw", [(100, 2, 1, 0), (100, 2, 2, 1), (100, 20, 5, 5)]
)
def test_inverse_anderson_rubin_confidence_set_alternative_formulation(
    alpha, n, k, mx, mw
):
    Z, X, y, _, W, _ = simulate_gaussian_iv(n=n, mx=mx, mw=mw, k=k)
    S = np.hstack([X, W])

    inverse_ar = inverse_anderson_rubin_test(
        Z, X, y, W=W, alpha=alpha, fit_intercept=False
    )
    kappa_alpha = 1 + scipy.stats.chi2(df=k - mw).ppf(1 - alpha) / (n - k)
    kclass_kappa_alpha = KClass(kappa=kappa_alpha, fit_intercept=False).fit(
        X=S, y=y, Z=Z
    )
    assert np.allclose(inverse_ar.center, kclass_kappa_alpha.coef_[:mx], rtol=1e-6)

    A = (kappa_alpha * proj(Z, S) + (1 - kappa_alpha) * S).T @ S
    if mw > 0:
        A = np.linalg.inv(np.linalg.inv(A)[:mx, :mx])

    residuals = y - kclass_kappa_alpha.predict(S)
    sigma_hat_sq = np.linalg.norm(oproj(Z, residuals)) ** 2 / (n - k)
    ar = anderson_rubin_test(
        Z, S, y, beta=kclass_kappa_alpha.coef_, fit_intercept=False
    )[0]
    c = sigma_hat_sq * (scipy.stats.chi2(df=k - mw).ppf(1 - alpha) - ar * k)

    assert np.allclose(
        -A / c,
        inverse_ar.A / inverse_ar.c_standardized,
        rtol=1e-8,
    )
