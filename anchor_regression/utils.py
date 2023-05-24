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


def pulse_test(anchor, residuals):
    """
    Test proposed in [1]_ with H0: anchor and residuals are uncorrelated.

    See [1]_ Section 3.2 for details.

    References
    ----------
    .. [1] https://arxiv.org/abs/2005.03353
    """
    proj_residuals = proj(anchor, residuals)
    statistic = np.square(proj_residuals).sum() / np.square(residuals).sum()
    statistic *= anchor.shape[0]
    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=anchor.shape[1])
    return statistic, p_value