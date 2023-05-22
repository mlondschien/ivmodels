import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from anchor_regression.linear_model import (
    AnchorElasticNet,
    AnchorRidge,
    LinearAnchorRegression,
)
from anchor_regression.testing import simulate_iv


# You are not supposed to use Ridge and ElasticNet with alpha=0
@pytest.mark.filterwarnings("ignore:Objective did not converge")
@pytest.mark.filterwarnings("ignore:With alpha=0, this algorithm does not converge")
@pytest.mark.filterwarnings("ignore:Coordinate descent with no regularization may lead")
@pytest.mark.parametrize("gamma", [1, 5])
@pytest.mark.parametrize("alpha", [0, 1])
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("dim_y", [1, 2])
def test_linear_anchor_regression_equal_to_ols(gamma, alpha, p, dim_y):
    n = 100

    X, Y, A = simulate_iv(n=n, discrete=False, p=p, seed=0, shift=0, dim_y=dim_y)
    df = pd.DataFrame(
        np.hstack([X, A]), columns=[f"X{k}" for k in range(p)] + ["anchor"]
    )

    lar = LinearAnchorRegression(gamma=gamma, exogenous_names=["anchor"]).fit(df, Y)
    aridge = AnchorRidge(gamma=gamma, alpha=alpha, exogenous_names=["anchor"]).fit(
        df, Y
    )
    aelastic = AnchorElasticNet(
        gamma=gamma, alpha=alpha / n, l1_ratio=0, exogenous_names=["anchor"]
    ).fit(df, Y)
    ols = LinearRegression(fit_intercept=True).fit(X, Y)

    if gamma == 1 and alpha == 0:
        assert np.allclose(aelastic.predict(df), ols.predict(X))
        assert np.allclose(aelastic.coef_, ols.coef_)
        assert np.allclose(aelastic.intercept_, ols.intercept_)
    if alpha == 0:
        assert np.allclose(aelastic.predict(df), lar.predict(df))
        assert np.allclose(aelastic.coef_, lar.coef_)
        assert np.allclose(aelastic.intercept_, lar.intercept_)

    assert np.allclose(aelastic.predict(df), aridge.predict(df))
    assert np.allclose(aelastic.coef_, aridge.coef_)
    assert np.allclose(aelastic.intercept_, aridge.intercept_)


# @pytest.mark.parametrize("shift", [0, 1, 2, 5])
# @pytest.mark.parametrize("p", [1, 5])
# @pytest.mark.parametrize("discrete", [False])
# def test_linear_anchor_regression_extrapolation(shift, p, discrete):

#     X, Y, A = simulate_iv(discrete=discrete, p=p, shift=0, seed=0)
#     X_test, Y_test, _ = simulate_iv(discrete=discrete, p=p, shift=shift, seed=1)

#     gammas = [0, 0.5,  1, 2, 3, 5, 9, 16, 32, 64]
#     losses = {}
#     models = {}
#     for gamma in gammas:
#         models[gamma] = LinearAnchorRegression(gamma=gamma).fit(X, Y, A)
#         losses[gamma] = np.mean((models[gamma].predict(X_test) - Y_test) ** 2)
#     breakpoint()


@pytest.mark.parametrize(
    "model", [LinearAnchorRegression, AnchorRidge, AnchorElasticNet]
)
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("dim_y", [1, 2])
@pytest.mark.parametrize("gamma", [0, 1, 5])
def test_linear_anchor_regression_different_inputs(model, p, dim_y, gamma):
    X, Y, A = simulate_iv(discrete=False, p=p, seed=0, shift=0, dim_y=dim_y)

    df = pd.DataFrame(np.hstack([X, A]), columns=[f"X{k}" for k in range(p)] + ["A1"])

    ar_1 = model(gamma=gamma, exogenous_names=["A1"]).fit(df, Y)
    ar_2 = model(gamma=gamma, exogenous_regex="A").fit(df, Y)
    ar_3 = model(gamma=gamma).fit(X, Y, A)

    assert np.allclose(ar_1.coef_, ar_2.coef_)
    assert np.allclose(ar_1.coef_, ar_3.coef_)

    assert np.allclose(ar_1.intercept_, ar_2.intercept_)
    assert np.allclose(ar_1.intercept_, ar_3.intercept_)


# We fit on df with feature names, but predict on X without feature names
@pytest.mark.filterwarnings("ignore:X does not have valid feature names, but LinearAnc")
def test_linear_anchor_regression_raises():
    X, Y, A = simulate_iv(discrete=False, p=5, seed=0, shift=0, dim_y=1)

    df = pd.DataFrame(
        np.hstack([X, A, A]), columns=[f"X{k}" for k in range(5)] + ["A1", "A2"]
    )

    ar_1 = LinearAnchorRegression(gamma=1, exogenous_names=["A1", "A2"])
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

    ar_2 = LinearAnchorRegression(gamma=1, exogenous_regex="A")
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

    ar_3 = LinearAnchorRegression(gamma=1)
    with pytest.raises(ValueError, match="`a` must be specified"):
        ar_3.fit(X, Y)

    ar_3.fit(X, Y, A)
    _ = ar_3.predict(X)
