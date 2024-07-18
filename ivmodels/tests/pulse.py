import numpy as np
import scipy

from ivmodels.quadric import Quadric
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import oproj, proj


def pulse_test(Z, X, y, beta, C=None, W=None, D=None, fit_intercept=True):
    """
    Test proposed by :cite:t:`jakobsen2022distributional` with null hypothesis: :math:`Z` and :math:`y - X \\beta` are uncorrelated.

    The test statistic is defined as

    .. math:: T := n \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| (y - X \\beta) \\|_2^2}.

    Under the null, :math:`T` is asymptotically distributed as :math:`\\chi^2(k)`.
    See Section 3.2 of :cite:p:`jakobsen2022distributional` for details.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    beta: np.ndarray of dimension (mx,)
        Coefficients to test.
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    W: np.ndarray of dimension (n, mw) or None, optional, default=None
        Must be None or `mw` must be 0. No subvector variant of the test is implemented.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

    Returns
    -------
    statistic: float
        The test statistic :math:`T`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(k)}(T)`, where
        :math:`F_{\\chi^2(k)}` is the cumulative distribution function of the
        :math:`\\chi^2(k)` distribution.

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
    Z, X, y, W, C, D, beta = _check_test_inputs(Z, X, y, C=C, W=W, D=D, beta=beta)

    if W.shape[1] > 0:
        raise ValueError("No subvector variant of the pulse test is implemented.")

    if D.shape[1] > 0:
        return pulse_test(
            np.hstack([Z, D]),
            np.hstack([Z, X]),
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

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=k)
    return statistic, p_value


def inverse_pulse_test(Z, X, y, C=None, D=None, alpha=0.05, fit_intercept=True):
    """Return the quadric for the inverse pulse test's acceptance region."""
    Z, X, y, _, C, D, _ = _check_test_inputs(Z, X, y, C=C, D=D)

    if D.shape[1] > 0:
        return inverse_pulse_test(
            np.hstack([Z, D]),
            np.hstack([Z, X]),
            y,
            C=C,
            alpha=alpha,
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

    if isinstance(c, np.ndarray):
        c = c.item()

    return Quadric(A, b, c)
