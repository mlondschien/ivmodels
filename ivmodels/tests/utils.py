import numpy as np


def _check_test_inputs(Z, X, y, W=None, beta=None):
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
    beta: np.ndarray of dimension (mx,) or None
        Coefficients.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    """
    if W is None:
        W = np.empty((Z.shape[0], 0))

    if Z.ndim != 2:
        raise ValueError(f"Z must be a matrix. Got shape {Z.shape}.")
    if X.ndim != 2:
        raise ValueError(f"X must be a matrix. Got shape {X.shape}.")
    if y.ndim != 1:
        if y.shape[1] != 1:
            raise ValueError(f"y must be a vector. Got shape {y.shape}.")
        else:
            y = y.flatten()
    if W.ndim != 2:
        raise ValueError(f"W must be a matrix. Got shape {W.shape}.")

    if not Z.shape[0] == X.shape[0] == y.shape[0] == W.shape[0]:
        raise ValueError(
            f"Z, X, y, and W must have the same number of rows. Got shapes {Z.shape}, "
            f"{X.shape}, {y.shape} and {W.shape}."
        )

    if beta is not None and beta.ndim != 1:
        if beta.shape[1] != 1:
            raise ValueError(f"beta must be a vector. Got shape {beta.shape}.")
        else:
            beta = beta.flatten()

        if beta.shape[0] != X.shape[1]:
            raise ValueError(
                "beta must have the same length or number of rows as X has columns. "
                f"Got shapes {beta.shape} and {X.shape}."
            )

    return Z, X, y, W, beta
