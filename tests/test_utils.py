import numpy as np
import pytest

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


def test_proj_multiple_args():
    rng = np.random.RandomState(0)

    X = rng.normal(0, 1, (100, 2))
    z1 = rng.normal(0, 1, (100, 5))
    z2 = rng.normal(0, 1, (100, 0))
    z3 = rng.normal(0, 1, (100,))

    z1_proj, z2_proj, z3_proj = proj(X, z1, z2, z3)
    assert (z1_proj.shape, z2_proj.shape, z3_proj.shape) == (
        z1.shape,
        z2.shape,
        z3.shape,
    )
    assert np.allclose(proj(X, z1), X @ np.linalg.inv(X.T @ X) @ X.T @ z1)
    assert np.allclose(proj(X, z2), X @ np.linalg.inv(X.T @ X) @ X.T @ z2)
    assert np.allclose(
        proj(X, z2, z2),
        X @ np.linalg.inv(X.T @ X) @ X.T @ z2,
        X @ np.linalg.inv(X.T @ X) @ X.T @ z2,
    )
    assert np.allclose(proj(X, z3), X @ np.linalg.inv(X.T @ X) @ X.T @ z3)


def test_oproj_multiple_args():
    rng = np.random.RandomState(0)

    X = rng.normal(0, 1, (100, 2))
    z1 = rng.normal(0, 1, (100, 5))
    z2 = rng.normal(0, 1, (100, 0))
    z3 = rng.normal(0, 1, (100,))

    z1_proj, z2_proj, z3_proj = proj(X, z1, z2, z3)
    assert (z1_proj.shape, z2_proj.shape, z3_proj.shape) == (
        z1.shape,
        z2.shape,
        z3.shape,
    )
    assert np.allclose(oproj(X, z1), z1 - X @ np.linalg.inv(X.T @ X) @ X.T @ z1)
    assert np.allclose(oproj(X, z2), z2 - X @ np.linalg.inv(X.T @ X) @ X.T @ z2)
    assert np.allclose(oproj(X, z3), z3 - X @ np.linalg.inv(X.T @ X) @ X.T @ z3)


def test_proj_raises():
    rng = np.random.RandomState(0)

    X = rng.normal(0, 1, (100, 2))
    z = rng.normal(0, 1, (100, 5))

    with pytest.raises(ValueError, match="Shape mismatch:"):
        proj(X, z[1:, :], z)

    with pytest.raises(ValueError, match="Shape mismatch:"):
        proj(X, z[1:, 0], z)

    with pytest.raises(ValueError, match="args should have shapes"):
        proj(X, z.reshape(10, 10, 5), z)


def test_oproj_raises():
    rng = np.random.RandomState(0)

    X = rng.normal(0, 1, (100, 2))
    z = rng.normal(0, 1, (100, 5))

    with pytest.raises(ValueError, match="Shape mismatch:"):
        oproj(X, z[1:, :], z)

    with pytest.raises(ValueError, match="Shape mismatch:"):
        oproj(X, z[1:, 0], z)

    with pytest.raises(ValueError, match="args should have shapes"):
        oproj(X, z.reshape(10, 10, 5), z)
