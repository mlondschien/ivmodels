def _check_test_inputs(Z, X, y, W=None, beta=None):
    """
    Test dimensions of inputs to tests.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    W: np.ndarray of dimension (n, r), optional, default=None
        Regressors to control for.
    beta: np.ndarray of dimension (p,), optional, default=None
        Coefficients.

    Returns
    -------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    W: np.ndarray of dimension (n, r) or None
        Regressors to control for.
    beta: np.ndarray of dimension (p,) or None
        Coefficients.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be a matrix. Got shape {Z.shape}.")
    if X.ndim != 2:
        raise ValueError(f"X must be a matrix. Got shape {X.shape}.")
    if y.ndim != 1:
        if y.shape[1] != 1:
            raise ValueError(f"y must be a vector. Got shape {y.shape}.")
        else:
            y = y.flatten()

    if not Z.shape[0] == X.shape[0] == y.shape[0]:
        raise ValueError(
            f"Z, X, and y must have the same number of rows. Got shapes {Z.shape}, {X.shape}, and {y.shape}."
        )

    if beta is not None and beta.ndim != 1:
        if beta.shape[1] != 1:
            raise ValueError(f"beta must be a vector. Got shape {beta.shape}.")
        else:
            beta = beta.flatten()

        if beta.shape[0] != X.shape[1]:
            raise ValueError(
                f"beta must have the same length or number of rows as X has columns. Got shapes {beta.shape} and {X.shape}."
            )

    if W is not None:
        if W.ndim != 2:
            raise ValueError(f"W must be a matrix. Got shape {W.shape}.")
        if not W.shape[0] == X.shape[0]:
            raise ValueError(
                f"W and X must have the same number of rows. Got shapes {W.shape} and {X.shape}."
            )

    return Z, X, y, W, beta
