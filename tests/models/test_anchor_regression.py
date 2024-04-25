import numpy as np
import pandas as pd
import pytest
import scipy
from glum import GeneralizedLinearRegressor

from ivmodels.models.anchor_regression import AnchorRegression
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.utils import proj


@pytest.mark.parametrize("alpha, l1_ratio", [(0, 0), (1, 0), (1, 0.5), (1, 1)])
@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_linear_anchor_regression_equal_to_ols(alpha, l1_ratio, n, mx, k, u):
    n = 100

    A, X, y, _, _ = simulate_gaussian_iv(n, mx, k, u)
    df = pd.DataFrame(
        np.hstack([X, A]),
        columns=[f"X{i}" for i in range(mx)] + [f"anchor{i}" for i in range(k)],
    )

    lar = AnchorRegression(
        gamma=1,
        alpha=alpha,
        l1_ratio=l1_ratio,
        instrument_regex="anchor",
        fit_intercept=True,
    ).fit(X=df, y=y.flatten())
    ols = GeneralizedLinearRegressor(
        family="gaussian", alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True
    ).fit(X, y.flatten())

    assert np.allclose(lar.predict(df), ols.predict(X))
    assert np.allclose(lar.coef_, ols.coef_)
    assert np.allclose(lar.intercept_, ols.intercept_)


@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("gamma", [0.01, 1, 5])
def test_linear_anchor_regression_different_inputs(gamma, n, mx, k, u):
    A, X, y, _, _ = simulate_gaussian_iv(n, mx, k, u)

    anchors = [f"A{i}" for i in range(k)]
    df = pd.DataFrame(np.hstack([X, A]), columns=[f"X{i}" for i in range(mx)] + anchors)

    ar_1 = AnchorRegression(gamma=gamma, instrument_names=anchors).fit(df, y)
    ar_2 = AnchorRegression(gamma=gamma, instrument_regex="A").fit(df, y)
    ar_3 = AnchorRegression(gamma=gamma).fit(X, y, A)

    assert np.allclose(ar_1.coef_, ar_2.coef_)
    assert np.allclose(ar_1.coef_, ar_3.coef_)

    assert np.allclose(ar_1.intercept_, ar_2.intercept_)
    assert np.allclose(ar_1.intercept_, ar_3.intercept_)


def test_score():
    A, X, y, _, _ = simulate_gaussian_iv(100, 2, 2, 1)
    model = AnchorRegression(gamma=1).fit(X, y, A)
    assert model.score(X, y) > 0.5


@pytest.mark.parametrize("gamma", [0.1, 1, 5])
@pytest.mark.parametrize(
    "n, mx, r, k, u", [(100, 2, 1, 2, 1), (100, 2, 0, 5, 2), (100, 0, 2, 0, 2)]
)
def test_anchor_solution_minimizes_loss(n, mx, k, u, r, gamma):
    """
    Test that the anchor solution minimizes the loss function.

    This indirectly checks the mapping kappa <-> gamma for validity.
    """
    Z, X, y, C, _ = simulate_gaussian_iv(n, mx, k, u, r=r)

    ar = AnchorRegression(gamma=gamma, fit_intercept=False).fit(X, y, Z, C=C)

    def loss(beta):
        return np.mean((y - np.hstack([X, C]) @ beta - ar.intercept_) ** 2) + (
            gamma - 1
        ) * np.mean(
            proj(np.hstack([Z, C]), y - np.hstack([X, C]) @ beta - ar.intercept_) ** 2
        )

    assert np.allclose(scipy.optimize.approx_fprime(ar.coef_, loss), 0, atol=1e-6)
