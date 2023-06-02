import numpy as np
import scipy


def proj(Z, f):
    """Project f onto the subspace spanned by Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z).
        The Z matrix.
    f: np.ndarray of dimension (n, d_f) or (n,).
        The vector to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of f onto the subspace spanned by Z. Same dimension as f.
    """
    Z = Z - Z.mean(axis=0)
    # f = f - f.mean(axis=0)

    return np.dot(Z, np.linalg.lstsq(Z, f, rcond=None)[0])


def pulse_test(Z, residuals):
    """
    Test proposed in [1]_ with H0: Z and residuals are uncorrelated.

    See [1]_ Section 3.2 for details.

    References
    ----------
    .. [1] https://arxiv.org/abs/2005.03353
    """
    proj_residuals = proj(Z, residuals)
    statistic = np.square(proj_residuals).sum() / np.square(residuals).sum()
    statistic *= Z.shape[0]
    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=Z.shape[1])
    return statistic, p_value


def anderson_rubin_test(Z, residuals):
    """
    Perform the Anderson Rubin test.

    Test the null hypothesis that the residuals are uncorrelated with the instruments.
    Under the null, the test statistic is distributed as `F_{q, n - q}`, where `q` is
    the number of instruments and `n` is the number of observations. Requires normally
    distributed errors for exactness.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d).
        Instruments.
    residuals: np.ndarray of dimension (n,).
        The residuals to test.

    Returns
    -------
    statistic: float
        The test statistic.
    p_value: float
        The p-value of the test.

    References
    ----------
    .. [1] Anderson, T.W. and Rubin, H. (1949), Estimation of the parameters of a single
           equation in a complete system of stochastic equations, Annals of Mathematical
           Statistics, 20, 46-63.
    """
    n, q = Z.shape
    proj_residuals = proj(Z, residuals)
    statistic = (
        np.square(proj_residuals).sum() / np.square(residuals - proj_residuals).sum()
    )
    statistic *= (n - q) / q
    p_value = 1 - scipy.stats.f.cdf(statistic, dfn=n - q, dfd=q)
    return statistic, p_value
