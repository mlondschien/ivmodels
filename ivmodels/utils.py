import numpy as np

try:
    import pandas as pd

    _PANDAS_INSTALLED = True
except ImportError:
    _PANDAS_INSTALLED = False


def proj(Z, f):
    """Project f onto the subspace spanned by Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z)
        The Z matrix. If None, returns np.zeros_like(f).
    f: np.ndarray of dimension (n, d_f) or (n,)
        The vector to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of f onto the subspace spanned by Z. Same dimension as f.
    """
    if Z is None:
        return np.zeros_like(f)

    return np.dot(Z, np.linalg.lstsq(Z, f, rcond=None)[0])


def oproj(Z, f):
    """Project f onto the subspace orthogonal to the subspace spanned by Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z) or None
        The Z matrix. If None, returns f.
    f: np.ndarray of dimension (n, d_f) or (n,)
        The vector to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of f onto the subspace spanned by Z. Same dimension as f.
    """
    if Z is None:
        return f

    return f - np.dot(Z, np.linalg.lstsq(Z, f, rcond=None)[0])


def to_numpy(x):
    """Convert x to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    elif _PANDAS_INSTALLED and isinstance(x, pd.DataFrame):
        return x.to_numpy()
    else:
        raise ValueError(f"Invalid type: {type(x)}")
