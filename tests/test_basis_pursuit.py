import numpy as np

from ivmodels.anderson_rubin_basis_pursuit import anderson_rubin_basis_pursuit
from ivmodels.tests import anderson_rubin_test, inverse_anderson_rubin


def test_anderson_rubin_basis_pursuit():
    rng = np.random.default_rng(0)

    n = 1000
    p = 20
    q = 40
    u = 10

    nonzero_coefficients = 5

    beta = rng.normal(size=p)
    beta[nonzero_coefficients:] = 0

    Pi = rng.normal(size=(q, p))
    delta = rng.normal(size=(u, p))
    gamma = rng.normal(size=u)

    Z = rng.normal(size=(n, q))
    U = rng.normal(size=(n, u))
    X = Z @ Pi + U @ delta + rng.normal(size=(n, p))
    y = X @ beta + U @ gamma + rng.normal(size=n)

    beta_hat = anderson_rubin_basis_pursuit(Z, X, y)

    assert anderson_rubin_test(Z, y - X @ beta_hat)[1] > 0.05

    if p == 2:
        boundary = inverse_anderson_rubin(Z, X, y, alpha=0.05)._boundary()
        assert np.linalg.norm(beta_hat, ord=1) <= np.max(
            np.linalg.norm(boundary, ord=1, axis=1)
        )
