import numpy as np
import scipy

from ivmodels.quadric import Quadric
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


def likelihood_ratio_test(Z, X, y, beta, W=None, fit_intercept=True):
    """
    Perform the likelihood ratio test for ``beta``.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       \\mathrm{LR(\\beta)} &:= (n - k) \\frac{ \\| P_Z (y - X \\beta) \\|_2^2}{ \\| M_Z (y - X \\beta) \\|_2^2} - (n - k) \\frac{ \\| P_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 } \\\\
       &= k \\ \\mathrm{AR}(\\beta)) - k \\ \\mathrm{AR}(\\hat\\beta_\\mathrm{LIML}),

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z`,
    :math:`M_Z = \\mathrm{Id} - P_Z`, and :math:`\\hat\\beta_\\mathrm{LIML}` is the LIML
    estimator of :math:`\\beta`, minimizing the Anderson-Rubin test statistic
    :math:`\\mathrm{AR}(\\beta)` (see :py:func:`~ivmodels.tests.anderson_rubin_test`) at
    :math:`\\mathrm{AR}(\\hat\\beta_\\mathrm{LIML}) = \\frac{n - k}{k} (\\hat\\kappa_\\mathrm{LIML} - 1)`.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::

       \\mathrm{LR(\\beta)} := (n - k) \\frac{ \\|P_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|^2_2 }{\\| M_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2 } - (n - k) \\frac{\\| P_Z (y - (X \\ W) \\hat\\delta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - (X \\ W) \\hat\\delta_\\mathrm{LIML}) \\|_2^2}

    where :math:`\\gamma_\\mathrm{LIML}` is the LIML estimator (see
    :py:class:`~ivmodels.KClass`) using instruments :math:`Z`, endogenous
    covariates :math:`W`, and outcomes :math:`y - X \\beta` and
    :math:`\\hat\\delta_\\mathrm{LIML}` is the LIML estimator
    using instruments :math:`Z`, endogenous covariates :math:`X \\ W`, and outcomes :math:`y`.

    Under the null and given strong instruments, the test statistic is asymptotically
    distributed as :math:`\\chi^2(m_X)`, where :math:`m_X` is the number of regressors.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    beta: np.ndarray of dimension (mw,)
        Coefficients to test.
    W: np.ndarray of dimension (n, mw) or None, optional, default=None
        Endogenous regressors not of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

    Returns
    -------
    statistic: float
        The test statistic :math:`LR`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(m_X)}(LR)`, where
        :math:`F_{\\chi^2(m_X)}` is the cumulative distribution function of the
        :math:`\\chi^2(m_X)` distribution.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.
    """
    Z, X, y, W, beta = _check_test_inputs(Z, X, y, W=W, beta=beta)

    if fit_intercept:
        Z = Z - Z.mean(axis=0)
        W = W - W.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y - y.mean()

    n, mx = X.shape
    k = Z.shape[1]

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)
    W_proj = proj(Z, W)

    XWy = np.concatenate([X, W, y.reshape(-1, 1)], axis=1)
    XWy_proj = np.concatenate([X_proj, W_proj, y_proj.reshape(-1, 1)], axis=1)

    matrix = np.linalg.solve(XWy.T @ (XWy - XWy_proj), XWy_proj.T @ XWy)
    ar_min = (n - k - fit_intercept) * min(np.abs(scipy.linalg.eigvals(matrix)))

    if W.shape[1] == 0:
        statistic = (n - k - fit_intercept) * np.linalg.norm(
            y_proj - X_proj @ beta
        ) ** 2 / np.linalg.norm((y - y_proj) - (X - X_proj) @ beta) ** 2 - ar_min
    else:
        Wy = np.concatenate([W, (y - X @ beta).reshape(-1, 1)], axis=1)
        Wy_proj = np.concatenate(
            [W_proj, (y_proj - X_proj @ beta).reshape(-1, 1)], axis=1
        )
        matrix = np.linalg.solve(Wy.T @ (Wy - Wy_proj), Wy_proj.T @ Wy)
        statistic = (n - k - fit_intercept) * min(
            np.abs(scipy.linalg.eigvals(matrix))
        ) - ar_min

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx)

    return statistic, p_value


def inverse_likelihood_ratio_test(Z, X, y, alpha=0.05, W=None, fit_intercept=True):
    """
    Return the quadric for the inverse likelihood ratio test's acceptance region.

    If ``W`` is ``None``, the quadric is defined as

    .. math::

       LR(\\beta) = (n - k) \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2} \\leq \\frac{1}{n} \\| y - X \\hat \\beta \\|^2_2 \\leq F_{\\chi^2(m_X)}(1 - \\alpha).

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
        Endogenous regressors not of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, _ = _check_test_inputs(Z, X, y, W=W)

    n, mx = X.shape
    k = Z.shape[1]

    if fit_intercept:
        Z = Z - Z.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y - y.mean()
        W = W - W.mean(axis=0)

    X = np.concatenate([X, W], axis=1)

    X_proj = proj(Z, X)
    X_orth = X - X_proj
    y_proj = proj(Z, y)
    y_orth = y - y_proj

    Xy_proj = np.concatenate([X_proj, y_proj.reshape(-1, 1)], axis=1)
    Xy = np.concatenate([X, y.reshape(-1, 1)], axis=1)

    matrix = np.linalg.solve(Xy.T @ (Xy - Xy_proj), Xy.T @ Xy_proj)
    kappa_liml = min(np.abs(np.linalg.eigvals(matrix)))

    dfd = n - k - fit_intercept
    quantile = scipy.stats.chi2.ppf(1 - alpha, df=mx) + dfd * kappa_liml

    A = X.T @ (X_proj - 1 / dfd * quantile * X_orth)
    b = -2 * (X_proj - 1 / dfd * quantile * X_orth).T @ y
    c = y.T @ (y_proj - 1 / dfd * quantile * y_orth)

    if isinstance(c, np.ndarray):
        c = c.item()

    if W is not None:
        return Quadric(A, b, c).project(range(X.shape[1] - W.shape[1]))
    else:
        return Quadric(A, b, c)
