import numpy as np
import pytest

from anchor_regression.quadric import Quadric


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
