import scipy

from ivmodels.models.kclass import KClass
from ivmodels.utils import _check_inputs, proj


def j_test(Z, X, y, beta, C=None, fit_intercept=True, estimator="liml"):
    """
    Perform the J-test for overidentifying restrictions.

    This is also called Sargan–Hansen test or Sargan’s J test.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Endogenous regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    C: np.ndarray of dimension (n, mc) or None, optional, default = None
        Exogenous regressors.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.
    estimator: str, optional, default = 'liml'
        Estimator to use. Passed to :py:class:`~ivmodels.KClass`.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\mathrm{AR}(\\beta)`.
    p_value: float
        The p-value of the test.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    """
    Z, X, y, _, C, _, _ = _check_inputs(Z, X, y, C=C)

    n, k = Z.shape
    mx, mc = X.shape[1], C.shape[1]

    estimator = KClass(estimator, fit_intercept=fit_intercept)
    residuals = y - X @ beta
    residuals_proj = proj(Z, residuals)
    residuals_orth = residuals - residuals_proj

    statistic = residuals_proj.T @ residuals_proj
    statistic /= residuals_orth.T @ residuals_orth / (n - k - mc - fit_intercept)

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=k - mx)

    return statistic, p_value
