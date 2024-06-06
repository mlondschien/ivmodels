import numpy as np

try:
    import pandas as pd

    _PANDAS_INSTALLED = True
except ImportError:
    _PANDAS_INSTALLED = False


def proj(Z, *args):
    """Project f onto the subspace spanned by Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z)
        The Z matrix. If None, returns np.zeros_like(f).
    *args: np.ndarrays of dimension (n, d_f) or (n,)
        vector or matrices to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of args onto the subspace spanned by Z. Same number of
        outputs as args. Same dimension as args
    """
    if Z is None:
        return *(np.zeros_like(f) for f in args),

    for f in args:
        if len(f.shape) > 2:
            raise ValueError(
                f"*args should have shapes (n, d_f) or (n,). Got {f.shape}."
            )
        if f.shape[0] != Z.shape[0]:
            raise ValueError(
                f"Shape mismatch: Z.shape={Z.shape}, f.shape={f.shape}."
            )

    if len(args) == 1:
        return  np.dot(Z, np.linalg.lstsq(Z, args[0], rcond=None)[0])
    
    csum = np.cumsum([f.shape[1] if len(f.shape) == 2 else 1 for f in args])
    csum = [0] + csum.tolist()

    fs = np.hstack([f.reshape(Z.shape[0], -1) for f in args])
    fs = np.dot(Z, np.linalg.lstsq(Z, fs, rcond=None)[0])

    return *(fs[:, i : j].reshape(f.shape) for i, j, f in zip(csum[:-1], csum[1:], args)),

def oproj(Z, *args):
    """Project f onto the subspace orthogonal to Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z)
        The Z matrix. If None, returns f.
    *args: np.ndarrays of dimension (n, d_f) or (n,)
        vector or matrices to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of args onto the subspace spanned by Z. Same number of
        outputs as args. Same dimension as args
    """
    if Z is None:
        return *args,

    for f in args:
        if len(f.shape) > 2:
            raise ValueError(
                f"*args should have shapes (n, d_f) or (n,). Got {f.shape}."
            )
        if f.shape[0] != Z.shape[0]:
            raise ValueError(
                f"Shape mismatch: Z.shape={Z.shape}, f.shape={f.shape}."
            )
    
    if len(args) == 1:
        return args[0] - np.dot(Z, np.linalg.lstsq(Z, args[0], rcond=None)[0])

    csum = np.cumsum([f.shape[1] if len(f.shape) == 2 else 1 for f in args])
    csum = [0] + csum.tolist()

    fs = np.hstack([f.reshape(Z.shape[0], -1) for f in args])
    fs = fs - np.dot(Z, np.linalg.lstsq(Z, fs, rcond=None)[0])

    return *(fs[:, i : j].reshape(f.shape) for i, j, f in zip(csum[:-1], csum[1:], args)),


def to_numpy(x):
    """Convert x to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    elif _PANDAS_INSTALLED and isinstance(x, pd.DataFrame):
        return x.to_numpy()
    else:
        raise ValueError(f"Invalid type: {type(x)}")
