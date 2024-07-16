import numpy as np


def _check_test_inputs(Z, X, y, W=None, C=None, D=None, beta=None):
    """
    Test dimensions of inputs to tests.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    W: np.ndarray of dimension (n, mw), optional, default=None
        Regressors to control for.
    C: np.ndarray of dimension (n, r), optional, default=None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md), optional, default=None
        Exogenous regressors of interest.
    beta: np.ndarray of dimension (mx,), optional, default=None
        Coefficients.

    Returns
    -------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    W: np.ndarray of dimension (n, mw)
        Regressors to control for. If input was None, returns an empty matrix of
        shape (n, 0).
    C: np.ndarray of dimension (n, mc)
        Exogenous regressors not of interest. If input was None, returns an empty matrix
        of shape (n, 0).
    D: np.ndarray of dimension (n, md)
        Exogenous regressors of interest. If input was None, returns an empty matrix of
        shape (n, 0).
    beta: np.ndarray of dimension (mx + md,) or None
        Coefficients.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    """
    if X is None:
        X = np.empty((Z.shape[0], 0))

    if W is None:
        W = np.empty((Z.shape[0], 0))

    if C is None:
        C = np.empty((Z.shape[0], 0))

    if D is None:
        D = np.empty((Z.shape[0], 0))

    if Z.ndim != 2:
        raise ValueError(f"Z must be a matrix. Got shape {Z.shape}.")
    if X.ndim != 2:
        raise ValueError(f"X must be a matrix. Got shape {X.shape}.")
    if y is None or y.ndim != 1:
        if y is None:
            y = np.empty(Z.shape[0])
        elif y.shape[1] != 1:
            raise ValueError(f"y must be a vector. Got shape {y.shape}.")
        else:
            y = y.flatten()
    if W.ndim != 2:
        raise ValueError(f"W must be a matrix. Got shape {W.shape}.")
    if C.ndim != 2:
        raise ValueError(f"C must be a matrix. Got shape {C.shape}.")
    if D.ndim != 2:
        raise ValueError(f"D must be a matrix. Got shape {D.shape}.")

    if (
        not Z.shape[0]
        == X.shape[0]
        == y.shape[0]
        == W.shape[0]
        == C.shape[0]
        == D.shape[0]
    ):
        raise ValueError(
            f"Z, X, y, W, C, and D must have the same number of rows. Got shapes "
            f"{Z.shape}, {X.shape}, {y.shape}, {W.shape}, {C.shape}, and {D.shape}."
        )

    if beta is not None and beta.ndim != 1:
        if beta.shape[1] != 1:
            raise ValueError(f"beta must be a vector. Got shape {beta.shape}.")
        else:
            beta = beta.flatten()

    if beta is not None:
        if beta.shape[0] != X.shape[1] + D.shape[1]:
            raise ValueError(
                "beta must have the same length or number of rows as X and D have "
                f"columns. Got shapes {beta.shape} and {X.shape}, {D.shape}."
            )

    return Z, X, y, W, C, D, beta


def _find_roots(f, a, b, tol, max_value, max_eval, n_points=50):
    """
    Find the root of function ``f`` between ``a`` and ``b`` closest to ``b``.

    Assumes ``f(a) < 0`` and ``f(b) > 0``. Finds root by building a grid between ``a``
    and ``b`` with ``n_points``, evaluating ``f`` at each point, and finding the last
    point where ``f`` is negative. If ``b`` is infinite, uses a logarithmic grid between
    ``a`` and ``a + sign(b - a) * max_value``. The function is then called recursively
    on the new interval until the size of the interval is less than ``tol`` or the
    maximum number of evaluations ``max_eval`` of ``f`` is reached.

    There is no scipy root finding algorithm that ensures that the root found is the
    closest to ``b``. Note that this is also not strictly ensured by this function.
    """
    if np.abs(b - a) < tol or max_eval < 0:
        return b  # conservative
    if np.isinf(a):
        return a

    sgn = np.sign(b - a)
    if np.isinf(b):
        grid = np.ones(n_points) * a
        grid[1:] += sgn * np.logspace(0, np.log10(max_value), n_points - 1)
    else:
        grid = np.linspace(a, b, n_points)

    y = np.zeros(n_points)
    y[-1] = f(grid[-1])
    if y[-1] < 0:
        return sgn * np.inf

    y[0] = f(grid[0])
    if y[0] >= 0:
        raise ValueError("f(a) must be negative.")

    for i, x in enumerate(grid[:-1]):
        y[i] = f(x)

    last_positive = np.where(y < 0)[0][-1]

    # f(a_new) < 0 < f(b_new) -> repeat
    return _find_roots(
        f,
        grid[last_positive],
        grid[last_positive + 1],
        tol=tol,
        n_points=n_points,
        max_value=None,
        max_eval=max_eval - n_points,
    )
