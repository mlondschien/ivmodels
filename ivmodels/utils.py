import numpy as np


def proj(Z, f):
    """Project f onto the subspace spanned by Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z)
        The Z matrix.
    f: np.ndarray of dimension (n, d_f) or (n,)
        The vector to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of f onto the subspace spanned by Z. Same dimension as f.
    """
    Z = Z - Z.mean(axis=0)
    # f = f - f.mean(axis=0)

    return np.dot(Z, np.linalg.lstsq(Z, f, rcond=None)[0])
