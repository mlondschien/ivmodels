import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from anchor_regression.linear_model import LinearAnchorRegression


def simulate_iv(discrete=False, p=1, dim_y=1, seed=0, shift=0):
    rng = np.random.RandomState(seed)

    if discrete and shift == 0:
        A = rng.binomial(1, 0.5, size=(100, 1))
    elif discrete and shift != 0:
        A = np.ones((100, 1)) * shift
    else:
        A = rng.normal(size=(100, 1)) + shift

    H = rng.normal(size=(100, 1))
    X = rng.normal(size=(100, p)) + H + A
    Y = X[:, 0:dim_y] + 2 * H + A

    return (
        pd.DataFrame(
            np.hstack([X, A]), columns=[f"X{k}" for k in range(p)] + ["anchor"]
        ),
        Y,
    )


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("dim_y", [1, 2])
def test_linear_anchor_regression_equal_to_ols(seed, p, dim_y):
    X, Y = simulate_iv(discrete=False, p=p, seed=seed, shift=0, dim_y=dim_y)

    ar = LinearAnchorRegression(gamma=1, anchor_column_names=["anchor"]).fit(X, Y)
    ols = LinearRegression(fit_intercept=True).fit(X.drop(columns=["anchor"]), Y)

    assert np.allclose(ar.predict(X), ols.predict(X.drop(columns=["anchor"])))
    assert np.allclose(ar.coef_, ols.coef_)

    assert np.allclose(ar.intercept_, ols.intercept_)
