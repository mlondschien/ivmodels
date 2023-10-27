import numpy as np
import pytest
import scipy
from sklearn.linear_model import LinearRegression

from ivmodels.kclass import KClass, KClassMixin
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import anderson_rubin_test
from ivmodels.utils import proj


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
def test__kappa_liml_same_with_shortcut(n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u)

    X = X - X.mean(axis=0)
    y = y.flatten() - y.mean(axis=0)

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)

    k_class = KClass(kappa="liml")

    lbda = k_class.fit(X, y, Z).kappa_liml_
    assert np.allclose(lbda, k_class._kappa_liml(X, y, X_proj=X_proj, y_proj=y_proj))
    assert np.allclose(lbda, k_class._kappa_liml(X, y, Z=Z, X_proj=X_proj))
    assert np.allclose(lbda, k_class._kappa_liml(X, y, Z=Z, y_proj=y_proj))
    assert np.allclose(lbda, k_class._kappa_liml(X, y, Z=Z))


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test__kappa_liml_positive(n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u)

    k_class = KClass(kappa="liml")
    k_class.fit(X, y.flatten(), Z)

    if Z.shape[1] > X.shape[1]:
        assert k_class.kappa_liml_ > 1
    else:
        assert np.allclose(k_class.kappa_liml_, 1)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_liml_minimizes_anderson_rubin(n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u)
    X = X - X.mean(axis=0)
    y = y.flatten() - y.mean(axis=0)

    k_class = KClass(kappa="liml")
    k_class.fit(X, y, Z)

    def ar(beta):
        return anderson_rubin_test(Z, X, y, beta)[0]

    grad = scipy.optimize.approx_fprime(k_class.coef_.flatten(), ar, 1e-8)
    np.testing.assert_allclose(grad, 0, atol=1e-4)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("kappa", [0.2, 0.8])
def test_k_class_normal_equations(kappa, n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u)

    X = X - X.mean(axis=0)
    y = y.flatten() - y.mean(axis=0)

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)

    k_class = KClass(kappa=kappa)
    k_class.fit(X, y, Z)

    assert np.allclose(
        k_class.coef_, k_class._solve_normal_equations(X, y, X_proj, y_proj)
    )


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 4, 4, 2)])
def test_liml_equal_to_tsls_in_just_identified_setting(n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u)
    y = y.flatten()

    liml = KClass(kappa="liml")
    liml.fit(X, y, Z)

    Xhat = LinearRegression().fit(Z, X).predict(Z)
    tsls = LinearRegression().fit(Xhat, y)

    assert np.allclose(liml.coef_, tsls.coef_)
    assert np.allclose(liml.intercept_, tsls.intercept_)


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 4, 4, 2)])
def test_anderson_rubin_at_liml_is_equal_to_kappa_liml(n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u)
    y = y.flatten()

    liml = KClass(kappa="liml")
    liml.fit(X, y, Z)

    assert np.allclose(
        anderson_rubin_test(Z, X, y, liml.coef_)[0],
        (n - q) / q * (liml.kappa_liml_ - 1),
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "n, beta, Pi, gamma, delta",
    [
        (
            50,
            np.array([0.2]),
            np.array([[1.0], [-1.0]]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
        (
            50,
            np.array([0.2, -0.2]),
            np.array([[1.0, -1.0], [-1.0, 0.0]]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
        (
            200,
            np.array([0.2, -0.2]),
            np.array([[1.0, -1.0], [0.2, 0.8], [2.0, 1.2], [-1.0, 0.0]]),
            np.array([1.0]),
            np.array([[1.0]]),
        ),
    ],
)
def test_fuller_bias_and_mse(n, beta, Pi, gamma, delta):
    n_iterations = 100

    q, p = Pi.shape
    u = delta.shape[0]

    kappas = ["liml", "fuller(1)", "fuller(4)", 0]  # 0 is for OLS
    results = {kappa: np.zeros(shape=(n_iterations, p)) for kappa in kappas}

    for seed in range(n_iterations):
        rng = np.random.RandomState(seed)
        U = rng.normal(0, 1, (n, u))
        Z = rng.normal(0, 1, (n, q))
        X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
        y = X @ beta + U @ gamma + rng.normal(0, 1, (n,))

        for kappa in kappas:
            results[kappa][seed, :] = KClass(kappa=kappa).fit(X, y, Z).coef_

    mses = {k: np.mean((v - beta.flatten()) ** 2) for k, v in results.items()}
    # biases = {k: np.mean(v - beta.flatten(), axis=0) for k, v in results.items()}

    # Fuller(4) has the lowest MSE, but Fuller(1) has the lowest bias
    assert mses["fuller(4)"] == min(mses.values())
    # assert all(
    #     np.abs(biases["fuller(1)"]) == np.min(np.abs(list(biases.values())), axis=0)
    # )
