import numpy as np
import pytest
import scipy

from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests.lagrange_multiplier import _LM
from ivmodels.utils import proj


@pytest.mark.parametrize(
    "n, mx, mw",
    [(100, 1, 1)],
)
def test__LM_raises(n, mx, mw):
    rng = np.random.RandomState(0)
    X, W, y = rng.randn(n, mx), rng.randn(n, mw), rng.randn(n)

    with pytest.raises(ValueError, match="Z must"):
        _LM(X=X, y=y, W=W, dof=7)


@pytest.mark.parametrize(
    "n, mx, mw, k",
    [(100, 1, 1, 2), (100, 1, 2, 5), (1000, 2, 5, 10), (1000, 5, 2, 10)],
)
def test__LM__init__(n, mx, mw, k):
    rng = np.random.RandomState(0)
    X, W, y, Z = rng.randn(n, mx), rng.randn(n, mw), rng.randn(n), rng.randn(n, k)

    X_proj, W_proj, y_proj = proj(Z, X, W, y)

    lm1 = _LM(X=X, y=y, W=W, dof=7, X_proj=X_proj, y_proj=y_proj, W_proj=W_proj)
    lm2 = _LM(X=X, y=y, W=W, dof=7, Z=Z)
    assert lm1.__dict__.keys() == lm2.__dict__.keys()
    assert all(
        [
            np.all(np.isclose(lm1.__dict__[k], lm2.__dict__[k]))
            for k in lm1.__dict__.keys()
        ]
    )


@pytest.mark.parametrize(
    "n, mx, mw, k",
    [(100, 1, 1, 2), (100, 1, 2, 5), (1000, 2, 5, 10), (1000, 5, 2, 10)],
)
def test_lm_gradient(n, mx, mw, k):
    Z, X, y, _, W = simulate_gaussian_iv(
        n=n, mx=mx, k=k, mw=mw, include_intercept=False
    )
    lm = _LM(X=X, y=y, W=W, dof=7, Z=Z)

    rng = np.random.RandomState(0)
    for _ in range(5):
        beta = rng.normal(0, 1, mx)
        gamma = rng.normal(0, 1, mw)

        grad_approx = scipy.optimize.approx_fprime(
            gamma,
            lambda g: lm.derivative(beta=beta, gamma=g)[0],
            1e-6,
        )
        grad = lm.derivative(beta, gamma)[1]

        assert np.allclose(grad, grad_approx, rtol=5e-5, atol=5e-5)

        hess_approx = scipy.optimize.approx_fprime(
            gamma,
            lambda g: lm.derivative(beta=beta, gamma=g)[1],
            1e-6,
        )
        hess = lm.derivative(beta, gamma)[2]

        assert np.allclose(hess, hess_approx, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize(
    "n, mx, mw, k",
    [(100, 2, 5, 10), (100, 5, 2, 10)],
)
def test_lm_gradient_beta_gamma(n, mx, mw, k):
    Z, X, y, _, W = simulate_gaussian_iv(
        n=n, mx=mx, k=k, mw=mw, include_intercept=False
    )
    lm = _LM(X=X, y=y, W=W, dof=7, Z=Z)

    rng = np.random.RandomState(0)
    beta = rng.normal(0, 1, mx)
    gamma = rng.normal(0, 1, mw)

    assert np.allclose(
        lm.derivative(np.concatenate([beta, gamma]))[0],
        lm.derivative(beta, gamma)[0],
    )

    assert np.allclose(
        lm.derivative(np.concatenate([beta, gamma]))[1],
        lm.derivative(beta, gamma)[1],
    )
