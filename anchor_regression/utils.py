import numpy as np
import scipy


def proj(anchor, f):
    """Project f onto the subspace spanned by anchor.

    Parameters
    ----------
    anchor: np.ndarray of dimension (n, d_anchor).
        The anchor matrix.
    f: np.ndarray of dimension (n, d_f) or (n,).
        The vector to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of f onto the subspace spanned by anchor. Same dimension as f.
    """
    anchor = anchor - anchor.mean(axis=0)
    f = f - f.mean(axis=0)

    return np.dot(anchor, np.linalg.lstsq(anchor, f, rcond=None)[0])


def anderson_rubin_test(anchor, residuals):
    """Perform the Anderson-Rubin test."""
    proj_residuals = proj(anchor, residuals)
    chi_squared = np.square(proj_residuals).sum() / np.square(residuals).sum()
    chi_squared = chi_squared * (residuals.shape[0] - anchor.shape[1])
    p_value = 1 - scipy.stats.chi2.cdf(chi_squared, df=anchor.shape[1])
    return chi_squared, p_value
