import numpy as np
import scipy

from ivmodels.utils import _characteristic_roots, _check_inputs, oproj, proj


def rank_test(Z, X, C=None, fit_intercept=True):
    """
    Perform the Cragg-Donald test for reduced rank :cite:p:`cragg1997inferring`.

    Let :math:`X = Z \\Pi + V` with :math:`\\Pi \\in \\mathbb{R}^{k \\times m_X}`. The
    Cragg-Donald test is the Wald test with the null hypothesis

    .. math:: H_0 := \\mathrm{rank}(\\Pi) < m_X,

    The test statistic is

    .. math:: \\mathrm{CD} := \\lambda := (n-k) \\lambda_\\mathrm{min}((X^T M_Z X)^{-1} X^T P_Z X)

    where :math:`P_Z = Z (Z^T Z)^{-1} Z^T` is the orthogonal projection onto the
    column space of :math:`Z`, :math:`M_Z = I - P_Z` is the orthogonal projection onto
    the orthogonal complement of the column space of :math:`Z`, and
    :math:`\\lambda_\\mathrm{min}` is the smallest eigenvalue. Under the null
    hypothesis, :math:`\\mathrm{CD}` is asymptotically distributed as
    :math:`\\chi^2(k - m_X + 1)`.

    The Cragg-Donald test is asymptotically equivalent to
    :cite:t:`anderson1951estimating`'s likelihood ratio test for reduced rank of
    :math:`\\Pi`.

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
        The test statistic :math:`\\mathrm{CD}`.
    p_value: float
        The p-value of the test. Equal to
        :math:`1 - F_{\\chi^2(k - m_X + 1)}(\\mathrm{CD})`, where
        :math:`F_{\\chi^2(k - m_X + 1)}` is the cumulative distribution function of the
        :math:`\\chi^2(k - m_X + 1)` distribution.

    References
    ----------
    .. bibliography::
       :filter: False

       cragg1997inferring
       anderson1951estimating
    """
    Z, X, _, _, C, _, _ = _check_inputs(Z, X, y=None, C=C)

    n, k = Z.shape
    m = X.shape[1]

    if k < m:
        return np.nan, np.nan

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, Z = oproj(C, X, Z)

    X_proj = proj(Z, X)

    statistic = (n - k - C.shape[1]) * np.real(
        _characteristic_roots(
            a=X.T @ X_proj, b=X.T @ (X - X_proj), subset_by_index=[0, 0]
        )[0]
    )
    cdf = scipy.stats.chi2.cdf(statistic, df=(k - m + 1))

    return statistic, 1 - cdf
