import numpy as np
import scipy

from ivmodels.quadric import Quadric
from ivmodels.utils import _check_inputs, oproj, proj


def pulse_test(Z, X, y, beta, W=None, C=None, D=None, fit_intercept=True):
    """
    Test proposed by :cite:t:`jakobsen2022distributional` with null hypothesis: :math:`Z` and :math:`y - X \\beta` are uncorrelated.

    The test statistic is defined as

    .. math:: T := n \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| (y - X \\beta) \\|_2^2}.

    Under the null, :math:`T` is asymptotically distributed as :math:`\\chi^2(k)`.
    See Section 3.2 of :cite:p:`jakobsen2022distributional` for details.

    If ``D`` is not ``None``, it is added to both the instruments :math:`Z` and the
    regressors :math:`X`, such that :math:`T` is asymptotically distributed as
    :math:`\\chi^2(k + m_D)` under the null.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    beta: np.ndarray of dimension (mx + md,)
        Coefficients to test.
    W: np.ndarray of dimension (n, mw) or None, optional, default=None
        Must be None or `mw` must be 0. No subvector variant of the test is implemented.
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md) or None, optional, default=None
        Exogenous regressors of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

    Returns
    -------
    statistic: float
        The test statistic :math:`T`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(k + m_D)}(T)`, where
        :math:`F_{\\chi^2(k + m_D)}` is the cumulative distribution function of the
        :math:`\\chi^2(k + m_D)` distribution.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    References
    ----------
    .. bibliography::
       :filter: False

       jakobsen2022distributional
    """
    Z, X, y, W, C, D, beta = _check_inputs(Z, X, y, C=C, W=W, D=D, beta=beta)

    if W.shape[1] > 0:
        raise ValueError("No subvector variant of the pulse test is implemented.")

    if D.shape[1] > 0:
        return pulse_test(
            np.hstack([Z, D]),
            np.hstack([X, D]),
            y,
            beta,
            C=C,
            fit_intercept=fit_intercept,
        )

    n, k = Z.shape

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z = oproj(C, X, y, Z)

    residuals = y - X @ beta
    proj_residuals = proj(Z, residuals)
    statistic = np.square(proj_residuals).sum() / np.square(residuals).sum()
    statistic *= n - k - C.shape[1]

    p_value = scipy.stats.chi2.sf(statistic, df=k)
    return statistic, p_value


def inverse_pulse_test(Z, X, y, alpha=0.05, W=None, C=None, D=None, fit_intercept=True):
    """
    Return the quadric for the inverse pulse test's acceptance region.

    The quadric satisfies ``quadric(x) <= 0`` if and only if
    ``pulse_test(Z, X, y, beta=x, C=C, D=D)[1] > alpha``. It is thus a confidence
    region for the causal parameter corresponding to the regressors ``X`` and ``D``.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    alpha: float, optional, default=0.05
        Significance level.
    W: np.ndarray of dimension (n, mw) or None, optional, default=None
        Must be None or `mw` must be 0. No subvector variant of the test is implemented.
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md) or None, optional, default=None
        Exogenous regressors of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

    Returns
    -------
    Quadric
        The quadric for the acceptance region.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, C, D, _ = _check_inputs(Z, X, y, W=W, C=C, D=D)

    if W.shape[1] > 0:
        raise ValueError("No subvector variant of the pulse test is implemented.")

    if D.shape[1] > 0:
        return inverse_pulse_test(
            np.hstack([Z, D]),
            np.hstack([X, D]),
            y,
            alpha=alpha,
            C=C,
            fit_intercept=fit_intercept,
        )

    n, k = Z.shape

    quantile = scipy.stats.chi2.ppf(1 - alpha, df=k)

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z = oproj(C, X, y, Z)

    X_proj, y_proj = proj(Z, X, y)

    A = X.T @ (X_proj - 1 / (n - k - C.shape[1]) * quantile * X)
    b = -2 * (X_proj - 1 / (n - k - C.shape[1]) * quantile * X).T @ y
    c = y.T @ (y_proj - 1 / (n - k - C.shape[1]) * quantile * y)

    return Quadric(A, b, c)
