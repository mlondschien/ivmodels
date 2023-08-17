import numpy as np
import pandas as pd
import pytest
from glum import GeneralizedLinearRegressor

from ivmodels.anchor_regression import AnchorRegression
from ivmodels.simulate import simulate_gaussian_iv


@pytest.mark.parametrize("alpha, l1_ratio", [(0, 0), (1, 0), (1, 0.5), (1, 1)])
@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_linear_anchor_regression_equal_to_ols(alpha, l1_ratio, n, p, q, u):
    n = 100

    A, X, y = simulate_gaussian_iv(n, p, q, u)
    df = pd.DataFrame(
        np.hstack([X, A]),
        columns=[f"X{k}" for k in range(p)] + [f"anchor{k}" for k in range(q)],
    )

    lar = AnchorRegression(
        gamma=1, alpha=alpha, l1_ratio=l1_ratio, instrument_regex="anchor"
    ).fit(X=df, y=y.flatten())
    ols = GeneralizedLinearRegressor(
        family="gaussian", alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True
    ).fit(X, y.flatten())

    assert np.allclose(lar.predict(df), ols.predict(X))
    assert np.allclose(lar.coef_, ols.coef_)
    assert np.allclose(lar.intercept_, ols.intercept_)


# @pytest.mark.parametrize("shift", [0, 1, 2, 5])
# @pytest.mark.parametrize("p", [1, 5])
# @pytest.mark.parametrize("discrete", [False])
# def test_linear_anchor_regression_extrapolation(shift, p, discrete):

#     X, Y, A = simulate_gaussian_iv(discrete=discrete, p=p, shift=0, seed=0)
#     X_test, Y_test, _ = simulate_gaussian_iv(discrete=discrete, p=p, shift=shift, seed=1)

#     gammas = [0, 0.5,  1, 2, 3, 5, 9, 16, 32, 64]
#     losses = {}
#     models = {}
#     for gamma in gammas:
#         models[gamma] = LinearAnchorRegression(gamma=gamma).fit(X, Y, A)
#         losses[gamma] = np.mean((models[gamma].predict(X_test) - Y_test) ** 2)
#     breakpoint()


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("gamma", [0.01, 1, 5])
def test_linear_anchor_regression_different_inputs(gamma, n, p, q, u):
    A, X, Y = simulate_gaussian_iv(n, p, q, u)

    anchors = [f"A{k}" for k in range(q)]
    df = pd.DataFrame(np.hstack([X, A]), columns=[f"X{k}" for k in range(p)] + anchors)

    ar_1 = AnchorRegression(gamma=gamma, instrument_names=anchors).fit(df, Y.flatten())
    ar_2 = AnchorRegression(gamma=gamma, instrument_regex="A").fit(df, Y.flatten())
    ar_3 = AnchorRegression(gamma=gamma).fit(X, Y.flatten(), A)

    assert np.allclose(ar_1.coef_, ar_2.coef_)
    assert np.allclose(ar_1.coef_, ar_3.coef_)

    assert np.allclose(ar_1.intercept_, ar_2.intercept_)
    assert np.allclose(ar_1.intercept_, ar_3.intercept_)


# We fit on df with feature names, but predict on X without feature names
# @pytest.mark.filterwarnings("ignore:X does not have valid feature names, but LinearAnc")
def test_linear_anchor_regression_raise():
    A, X, Y = simulate_gaussian_iv(10, 3, 2, 1)
    Y = Y.flatten()

    df = pd.DataFrame(np.hstack([X, A]), columns=["X1", "X2", "X3", "A1", "A2"])

    ar_1 = AnchorRegression(gamma=1, instrument_names=["A1", "A2"])
    with pytest.raises(ValueError, match="must be None"):
        ar_1.fit(df, Y, A)

    with pytest.raises(ValueError, match="not found in X: {'A1'}"):
        ar_1.fit(df.drop(columns=["A1"]), Y)

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        ar_1.fit(X, Y)

    ar_1.fit(df, Y)
    _ = ar_1.predict(df)
    _ = ar_1.predict(X)
    _ = ar_1.predict(df.drop(columns=["A1", "A2"]))

    ar_2 = AnchorRegression(gamma=1, instrument_regex="A")
    with pytest.raises(ValueError, match="must be None"):
        ar_2.fit(df, Y, A)

    with pytest.raises(ValueError, match="No columns in X matched the regex A"):
        ar_2.fit(df.drop(columns=["A1", "A2"]), Y)

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        ar_2.fit(X, Y)

    ar_2.fit(df, Y)
    _ = ar_2.predict(df)
    _ = ar_2.predict(X)
    _ = ar_2.predict(df.drop(columns=["A1", "A2"]))

    ar_3 = AnchorRegression(gamma=1)
    with pytest.raises(ValueError, match="`Z` must be specified"):
        ar_3.fit(X, Y)

    ar_3.fit(X, Y, A)
    _ = ar_3.predict(X)


def test_score():
    A, X, Y = simulate_gaussian_iv(100, 2, 2, 1)
    model = AnchorRegression(gamma=1).fit(X, Y.flatten(), A)
    assert model.score(X, Y.flatten()) > 0.5
