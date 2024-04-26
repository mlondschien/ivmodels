import numpy as np

from ivmodels.utils import oproj, proj


def test_proj():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 2))
    Z = rng.normal(0, 1, (100, 5))

    assert np.allclose(proj(Z, X), Z @ np.linalg.inv(Z.T @ Z) @ Z.T @ X)


def test_oproj():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 2))
    Z = rng.normal(0, 1, (100, 5))

    assert np.allclose(X - proj(Z, X), oproj(Z, X))
