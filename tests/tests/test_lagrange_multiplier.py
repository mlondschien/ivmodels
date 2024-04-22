import numpy as np
import pytest
import scipy

from ivmodels.tests.lagrange_multiplier import _LM
from ivmodels.utils import proj


@pytest.mark.parametrize(
    "n, p, r, q, u",
    [(100, 1, 1, 2, 1), (100, 1, 2, 5, 2), (1000, 2, 5, 10, 2), (1000, 5, 2, 10, 2)],
)
def test_lm_gradient(n, p, r, q, u):
    rng = np.random.RandomState(0)

    delta_X = rng.normal(0, 1, (u, p))
    delta_W = rng.normal(0, 1, (u, r))
    delta_y = rng.normal(0, 1, (u, 1))

    beta = rng.normal(0, 0.1, (p, 1))
    gamma = 10 * rng.normal(0, 0.1, (r, 1))

    Pi_X = rng.normal(0, 1, (q, p))
    Pi_W = rng.normal(0, 1, (q, r))

    U = rng.normal(0, 1, (n, u))

    Z = rng.normal(0, 1, (n, q))
    X = Z @ Pi_X + U @ delta_X + rng.normal(0, 1, (n, p))
    W = Z @ Pi_W + U @ delta_W + rng.normal(0, 1, (n, r))
    y = X @ beta + W @ gamma + U @ delta_y + rng.normal(0, 1, (n, 1))

    X_proj = proj(Z, X)
    W_proj = proj(Z, W)
    y_proj = proj(Z, y)

    for _ in range(5):
        beta_test = rng.normal(0, 0.1, (p, 1))

        grad_approx = scipy.optimize.approx_fprime(
            beta_test.flatten(),
            lambda b: _LM(X, X_proj, y, y_proj, W, W_proj, b.reshape(-1, 1))[0],
            1e-6,
        )
        grad = _LM(X, X_proj, y, y_proj, W, W_proj, beta_test.reshape(-1, 1))[1]

        assert np.allclose(grad.flatten(), grad_approx, rtol=1e-4, atol=1e-4)
