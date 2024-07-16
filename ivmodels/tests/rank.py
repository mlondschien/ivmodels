import numpy as np
import scipy

from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import oproj, proj


def rank_test(Z, X, C=None, fit_intercept=True):
    """
    Perform Anderson's test for reduced rank :cite:p:`anderson1951estimating`.

    This is the likelihood ratio test with null hypothesis

    .. math: H_0 := \\mathrm{rank}(\\Pi) < m_X

    where :math:`X = Z \\Pi + V` with :math:`V` consisting i.i.d. copies of a centered
    Gaussian, uncorrelated with :math:`Z`. The test statistic is

    .. math: \\lambda := (n-k) \\lambda_\\mathrm{min}((X^T M_Z X)^{-1} X^T P_Z X)

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
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    fit_intercept: bool
        Whether to fit an intercept.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\lambda`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(k - m_X + 1)}(\\lambda)`,
        where :math:`F_{\\chi^2(k - m_X + 1)}` is the cumulative distribution function
        of the :math:`\\chi^2(k - m_X + 1)` distribution.

    """
    Z, X, _, _, C, D, _ = _check_test_inputs(Z, X, y=None, C=C)

    n, k = Z.shape
    m = X.shape[1]

    if k < m:
        raise ValueError("Need `Z.shape[1] >= X.shape[1]`.")

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        # TODO: How to handle D here?
        X, Z, D = oproj(C, X, Z, D)

    X_proj = proj(Z, X)

    statistic = (n - k - C.shape[1]) * np.real(
        scipy.linalg.eigvalsh(
            a=X.T @ X_proj, b=X.T @ (X - X_proj), subset_by_index=[0, 0]
        )[0]
    )
    cdf = scipy.stats.chi2.cdf(statistic, df=(k - m + 1))

    return statistic, 1 - cdf
