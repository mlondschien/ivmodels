import numpy as np
import pytest
import scipy

from ivmodels.quadric import Quadric


@pytest.mark.parametrize("p", [2, 5])
@pytest.mark.parametrize("seed", [0, 1])
def test_quadric(p, seed):
    rng = np.random.RandomState(seed)

    A = rng.normal(0, 1, (p, p))
    A = A / 2 + A.T / 2
    b = rng.normal(0, 1, p)
    c = rng.normal(0, 1)

    quadric = Quadric(A, b, c)
    x = rng.normal(0, 1, (100, p))

    assert np.allclose(x, quadric.inverse_map(quadric.forward_map(x)))
    assert np.allclose(x, quadric.forward_map(quadric.inverse_map(x)))

    assert np.allclose(
        ((quadric.inverse_map(x) * quadric.D) * quadric.inverse_map(x)).sum(axis=1)
        + quadric.c_standardized,
        (x @ quadric.A * x).sum(axis=1) + quadric.b @ x.T + quadric.c,
    )

    assert np.allclose(
        (quadric.forward_map(x) @ quadric.A * quadric.forward_map(x)).sum(axis=1)
        + quadric.forward_map(x) @ quadric.b
        + quadric.c,
        (x * quadric.D * x).sum(axis=1) + quadric.c_standardized,
    )


@pytest.mark.parametrize(
    "D, c_standardized",
    [
        (np.array([1, 2]), -1),
        (np.array([-2, -2]), 1),
        (np.array([1, -2]), -1),
        (np.array([-1, 4]), -1),
    ],
)
def test_quadric_boundary(D, c_standardized):
    quadric = Quadric(np.diag(D), np.zeros(2), c_standardized)
    zeros = quadric._boundary()
    assert np.allclose(quadric(zeros), 0)


@pytest.mark.parametrize(
    "D, c, expected",
    [
        ([1], -1, 2),
        ([1, 1], -1, np.pi),
        ([1, 4], -1, np.pi / 2),
        ([1], 1, 0),
        ([-1], 1, np.inf),
    ],
)
def test_quadric_volume(D, c, expected):
    quadric = Quadric(np.diag(D), np.zeros_like(D), c)
    assert quadric.volume() == expected


def test_quadric_from_constraints():
    rng = np.random.RandomState(0)

    n = 100
    p = 2

    X = rng.normal(0, 1, (n, p))
    beta = rng.normal(0, 1, p)

    A = X.T @ X
    b = 2 * -A @ beta
    c = beta.T @ A @ beta

    quadric = Quadric(A, b, c)

    assert np.allclose(quadric.center, beta)

    beta_hat = rng.normal(0, 1, p)
    assert np.allclose(
        quadric(beta_hat.reshape(1, -1)), (beta_hat - beta).T @ A @ (beta_hat - beta)
    )


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("p", [2, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_quadric_projection(n, p, seed):
    rng = np.random.RandomState(seed)

    X = rng.normal(0, 1, (n, p))
    beta = rng.normal(0, 1, p)

    A = X.T @ X
    b = 2 * -A @ beta
    c = beta.T @ A @ beta - 10

    quadric = Quadric(A, b, c)

    for j in range(p):
        sol = quadric._projection(j)
        assert np.allclose(quadric(np.array(sol)), 0)

        # The gradient of the quadric at the solution should be zero in each direction
        # other than j. In direction j, the gradient should be negative for the min,
        # (first component) and positive for the max (second component).
        grad0 = scipy.optimize.approx_fprime(sol[0], lambda x: quadric(x), 1e-8)
        grad1 = scipy.optimize.approx_fprime(sol[1], lambda x: quadric(x), 1e-8)

        assert grad0[j] < 0
        assert grad1[j] > 0

        grad0[j] = 0
        grad1[j] = 0

        assert np.allclose(grad0, 0, atol=1e-3)
        assert np.allclose(grad1, 0, atol=1e-3)
