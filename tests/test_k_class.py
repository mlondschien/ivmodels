import numpy as np
import pytest
import scipy
from sklearn.linear_model import LinearRegression

from anchor_regression.linear_model import KClass, KClassMixin
from anchor_regression.utils import anderson_rubin_test, proj


def data(n, p, q, u):
    rng = np.random.RandomState(0)
    delta = rng.normal(0, 1, (u, p))
    gamma = rng.normal(0, 1, (u, 1))

    beta = rng.normal(0, 1, (p, 1))
    Pi = rng.normal(0, 1, (q, p))

    U = rng.normal(0, 1, (n, u))
    Z = rng.normal(0, 1, (n, q))
    X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
    y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

    return Z, X, y


@pytest.mark.parametrize(
    "kappa, expected",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        ("fuller(1)", 1),
        ("fuller(0.2)", 0.2),
        ("FULLER(4)", 4),
        ("fuller", 1),
        ("FulLeR", 1),
        ("liml", 0),
        ("LIML", 0),
    ],
)
def test__fuller_alpha(kappa, expected):
    assert KClassMixin()._fuller_alpha(kappa) == expected


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test__lambda_liml_same_with_shortcut(n, p, q, u):
    Z, X, y = data(n, p, q, u)

    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)

    k_class = KClass(kappa="liml")

    lbda = k_class.fit(X, y, Z).lambda_liml_
    assert np.allclose(lbda, k_class._lambda_liml(X, y, X_proj=X_proj, y_proj=y_proj))
    assert np.allclose(lbda, k_class._lambda_liml(X, y, Z=Z, X_proj=X_proj))
    assert np.allclose(lbda, k_class._lambda_liml(X, y, Z=Z, y_proj=y_proj))
    assert np.allclose(lbda, k_class._lambda_liml(X, y, Z=Z))


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test__lambda_liml_positive(n, p, q, u):
    Z, X, y = data(n, p, q, u)

    k_class = KClass(kappa="liml")
    k_class.fit(X, y, Z)

    if Z.shape[1] > X.shape[1]:
        assert k_class.lambda_liml_ > 0
    else:
        assert np.allclose(k_class.lambda_liml_, 0)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_liml_minimizes_anderson_rubin(n, p, q, u):
    Z, X, y = data(n, p, q, u)
    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)

    k_class = KClass(kappa="liml")
    k_class.fit(X, y, Z)

    def ar(beta):
        return anderson_rubin_test(Z, y - X @ beta.reshape(-1, 1))[0]

    grad = scipy.optimize.approx_fprime(k_class.coef_.flatten(), ar, 1e-8)
    np.testing.assert_allclose(grad, 0, atol=1e-4)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("kappa", [0.2, 0.8])
def test_k_class_normal_equations(kappa, n, p, q, u):
    Z, X, y = data(n, p, q, u)

    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)

    k_class = KClass(kappa=kappa)
    k_class.fit(X, y, Z)

    assert np.allclose(
        k_class.coef_, k_class._solve_normal_equations(X, y, X_proj, y_proj)
    )


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 4, 4, 2)])
def test_liml_equal_to_tsls_in_just_identified_setting(n, p, q, u):
    Z, X, y = data(n, p, q, u)

    liml = KClass(kappa="liml")
    liml.fit(X, y, Z)

    Xhat = LinearRegression().fit(Z, X).predict(Z)
    tsls = LinearRegression().fit(Xhat, y)

    assert np.allclose(liml.coef_, tsls.coef_)
    assert np.allclose(liml.intercept_, tsls.intercept_)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 4, 4, 2)])
def test_anderson_rubin_at_liml_is_equal_to_lambda_liml(n, p, q, u):
    Z, X, y = data(n, p, q, u)

    liml = KClass(kappa="liml")
    liml.fit(X, y, Z)

    assert np.allclose(
        anderson_rubin_test(Z, y - X @ liml.coef_.reshape(-1, 1))[0],
        (n - q) / q * liml.lambda_liml_ / (1 - liml.lambda_liml_),
    )
