import numpy as np
import scipy

from ivmodels.utils import proj


def f_test(Z, X, fit_intercept=True):
    """
    Perform Anderson's test for reduced rank :cite:p:`anderson1951estimating`.

    This is the likelihood ratio test with null hypothesis

    .. math: H_0 := \\mathrm{rank}(\\Pi) < m_X

    where :math:`X = Z \\Pi + V` with :math:`V` consisting i.i.d. copies of a centered
    Gaussian, uncorrelated with :math:`Z`. The test statistic is

    .. math: F := \\frac{n-k}{k} \\lambda_\\mathrm{min}((X^T M_Z X)^{-1} X^T P_Z X)

    where :math:`P_Z = Z (Z^T Z)^{-1} Z^T` is the orthogonal projection onto the
    column space of :math:`Z`, :math:`M_Z = I - P_Z` is the orthogonal projection onto
    the orthogonal complement of the column space of :math:`Z`, and
    :math:`\\lambda_\\mathrm{min}` is the smallest eigenvalue.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    fit_intercept: bool
        Whether to fit an intercept.

    Returns
    -------
    statistic: float
        The test statistic :math:`F`.
    p_value: float
        The p-value of the test.
    """
    n, k = Z.shape

    if fit_intercept:
        X = X - X.mean(axis=0)
        Z = Z - Z.mean(axis=0)

    X_proj = proj(Z, X)

    W = np.linalg.solve(X.T @ (X - X_proj), X.T @ X_proj)
    statistic = (n - k - fit_intercept) / k * min(np.real(np.linalg.eigvals(W)))

    cdf = scipy.stats.f.cdf(statistic, dfn=k, dfd=n - k - fit_intercept)

    return statistic, 1 - cdf
