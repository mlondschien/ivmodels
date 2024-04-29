import numpy as np
import scipy

from ivmodels.quadric import Quadric
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


def pulse_test(Z, X, y, beta, fit_intercept=True):
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
    Z, X, y, _, beta = _check_test_inputs(Z, X, y, beta=beta)

    n, k = Z.shape

    if fit_intercept:
        Z = Z - Z.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y - y.mean()

    residuals = y - X @ beta
    proj_residuals = proj(Z, residuals)
    statistic = np.square(proj_residuals).sum() / np.square(residuals).sum()
    statistic *= n - k - fit_intercept

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=k)
    return statistic, p_value


def inverse_pulse_test(Z, X, y, alpha=0.05, fit_intercept=True):
    """Return the quadric for the inverse pulse test's acceptance region."""
    Z, X, y, _, _ = _check_test_inputs(Z, X, y)

    n, k = Z.shape

    quantile = scipy.stats.chi2.ppf(1 - alpha, df=k)

    if fit_intercept:
        Z = Z - Z.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y - y.mean()

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)

    A = X.T @ (X_proj - 1 / (n - k - fit_intercept) * quantile * X)
    b = -2 * (X_proj - 1 / (n - k - fit_intercept) * quantile * X).T @ y
    c = y.T @ (y_proj - 1 / (n - k - fit_intercept) * quantile * y)

    if isinstance(c, np.ndarray):
        c = c.item()

    return Quadric(A, b, c)
