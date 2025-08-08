import numpy as np
import pandas as pd
import pytest
import scipy

from ivmodels.utils import _characteristic_roots, _find_roots, oproj, proj, to_numpy


def test_proj():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 2))
    Z = rng.normal(0, 1, (100, 4)) @ rng.normal(0, 1, (4, 6))

    assert np.allclose(proj(Z, X), Z @ np.linalg.pinv(Z.T @ Z) @ Z.T @ X)
    assert np.allclose(proj(Z, X), proj(Z, proj(Z, X)))
    assert np.allclose(proj(proj(Z, X), X), proj(Z, X))


def test_oproj():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 2))
    Z = rng.normal(0, 1, (100, 4)) @ rng.normal(0, 1, (4, 6))

    assert np.allclose(X - proj(Z, X), oproj(Z, X))
    assert np.allclose(oproj(Z, X), oproj(Z, oproj(Z, X)))


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


def test_to_numpy():
    rng = np.random.RandomState(0)
    s1 = pd.Series(rng.normal(0, 1, 100))
    df2 = pd.DataFrame(rng.normal(0, 1, (100, 1)))
    df3 = pd.DataFrame(rng.normal(0, 1, (100, 2)))

    x1, x2, x3 = to_numpy(s1, df2, df3)
    assert np.allclose(x1, s1)
    assert np.allclose(x2, df2)
    assert np.allclose(x3, df3)

    assert np.allclose(to_numpy(s1), s1)
    assert np.allclose(to_numpy(df2), df2)
    assert np.allclose(to_numpy(df3), df3)


@pytest.mark.parametrize("dim, rank", [(10, 5), (10, 10), (10, 1)])
def test_characteristic_roots_random(dim, rank):
    rng = np.random.RandomState(0)
    A = rng.normal(0, 1, (dim, dim))
    A = A.T @ A
    B = rng.normal(0, 1, (dim, rank))
    B = B @ B.T

    roots = _characteristic_roots(A, B)
    assert len(roots) == rank

    for root in roots:
        assert np.allclose(np.min(np.abs(scipy.linalg.eigvalsh(A - root * B))), 0)

    assert np.allclose(
        _characteristic_roots(A, B, subset_by_index=[0, 0]), np.min(roots)
    )


@pytest.mark.parametrize(
    "A, B, roots",
    [
        (np.diag([1.0, 1.0]), np.diag([1.0, 0.0]), [1.0]),  # [1.0, 0.0] is a root
        # This is an important test case. If we just drop the second row & column of A, B,
        # corresponding to the zero eigenvalue of B, we get the root 1.0 > 0.5. This holds
        # even though A is symmetric positive definite.
        (
            np.array([[1.0, -0.5], [-0.5, 0.5]]),
            np.diag([1.0, 0.0]),
            [0.5],
        ),  # [1.0, 1.0]
        (np.array([[1.0, 1.0], [1.0, -1.0]]), np.diag([1.0, 0.0]), [2.0]),  # [1.0, 1.0]
    ],
)
def test_characteristic_roots_fixed(A, B, roots):
    roots_found = _characteristic_roots(A, B)
    assert np.allclose(roots_found, roots)


@pytest.mark.parametrize(
    "f, a, b, expected",
    [
        (np.sin, -1, 8, [0, np.pi, 2 * np.pi]),
        (lambda x: -np.sin(x), 8, -1, [0, np.pi, 2 * np.pi]),
        (lambda x: x**2 - 1, 0, 2, [1]),
        (lambda x: x**2 - 1, 0, -np.inf, [-1]),
        (lambda x: x**3 - x, -2, 2, [-1, 0, 1]),
        (lambda x: x**3 - x, 0.5, -np.inf, [-np.inf, -1, 0]),
    ],
)
@pytest.mark.parametrize("tol", [1e-3, 1e-6])
def test_find_roots(f, a, b, expected, tol):
    roots = _find_roots(f, a, b, max_value=1e6, max_eval=1e4, tol=tol)
    assert np.allclose(sorted(roots), expected, atol=tol)


@pytest.mark.parametrize("fun", [proj, oproj])
@pytest.mark.parametrize(
    "Z", [np.array([[1, 0], [0, 1]]), pd.DataFrame({"a": [1, 0], "b": [0, 1]}), None]
)
@pytest.mark.parametrize(
    "args",
    [
        (np.array([[1, 2], [3, 4]]),),
        (np.array([1, 2]),),
        (pd.Series([1, 2], index=["a", "b"]),),
        (pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=["a", "b"]),),
        (pd.DataFrame(index=["a", "b"]),),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([1, 2]),
            pd.Series([1, 2], index=["a", "b"]),
            pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=["a", "b"]),
            pd.DataFrame(index=["a", "b"]),
        ),
    ],
)
def test_proj_keeps_attributes(fun, Z, args):
    new = fun(Z, *args)

    if not isinstance(new, tuple):
        new = (new,)

    for o, n in zip(args, new):
        if isinstance(o, np.ndarray):
            assert o.shape == n.shape
        elif isinstance(o, pd.DataFrame):
            assert o.index.equals(n.index)
            assert o.columns.equals(n.columns)
        elif isinstance(o, pd.Series):
            assert o.index.equals(n.index)
            assert o.name == n.name
        else:
            raise ValueError(f"Invalid type: {type(o)}")
