import numpy as np
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.quadric import Quadric
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


def wald_test(Z, X, y, beta, W=None, estimator="tsls", fit_intercept=True):
    """
    Test based on asymptotic normality of the TSLS (or LIML) estimator.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       \\mathrm{Wald}(\\beta) := (\\beta - \\hat{\\beta})^T \\widehat{\\mathrm{Cov}}(\\hat\\beta)^{-1} (\\beta - \\hat{\\beta}) / \\hat{\\sigma}^2,

    where :math:`\\hat \\beta = \\hat \\beta(\\kappa)` is a k-class estimator with
    :math:`\\sqrt{n} (1 - \\kappa) \\to 0`,
    :math:`\\widehat{\\mathrm{Cov}}(\\hat\\beta)^{-1} = \\frac{1}{n} (X^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X)^{-1}`,
    :math:`\\hat \\sigma^2 = \\frac{1}{n - m_X} \\| y - X \\hat \\beta \\|^2_2` is
    an estimate of the variance of the errors, :math:`P_Z` is the projection matrix
    onto the column space of :math:`Z`, and :math:`M_Z = \\mathrm{Id} - P_Z`.
    Under strong instruments, the test statistic is asymptotically distributed as
    :math:`\\chi^2(m_X)` under the null.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::

        \\mathrm{Wald}(\\beta) := (\\beta - B \\hat{\\beta})^T (B ( (X W)^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) (X W) )^{-1} B)^{-1} (\\beta - B \\hat{\\beta}) / \\hat{\\sigma}^2,

    where :math:`B \\in \\mathbb{R}^{m_X \\times (m_X + m_W)}` is a diagonal matrix with
    1 on the diagonal and 0 elsewhere.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    W: np.ndarray of dimension (n, mw) or None
        Endogenous regressors not of interest.
    beta: np.ndarray of dimension (mx,)
        Coefficients to test.
    estimator: str, optional, default = "tsls"
        Estimator to use. Must be one of ``"tsls"`` or ``"liml"``.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. The intercept will be included both in the
        complete and the (restricted) model. Including an intercept is equivalent to
        centering the columns of all design matrices.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\mathrm{Wald}`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(m_X)}(Wald)`, where
        :math:`F_{\\chi^2(m_X)}` is the cumulative distribution function of the
        :math:`\\chi^2(m_X)` distribution.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    """
    Z, X, y, W, beta = _check_test_inputs(Z, X, y, W=W, beta=beta)

    if fit_intercept:
        X = X - X.mean(axis=0)
        y = y - y.mean()
        Z = Z - Z.mean(axis=0)
        W = W - W.mean(axis=0)

    n, mx = X.shape

    XW = np.concatenate([X, W], axis=1)

    estimator = KClass(kappa=estimator, fit_intercept=False).fit(XW, y, Z)

    beta_gamma_hat = estimator.coef_

    residuals = y - estimator.predict(XW)
    sigma_hat_sq = np.sum(residuals**2) / (n - mx - W.shape[1] - fit_intercept)

    XW_proj = proj(Z, XW)

    kappa = estimator.kappa_
    cov_hat = (kappa * XW_proj + (1 - kappa) * XW).T @ XW

    if W.shape[1] == 0:
        statistic = (beta_gamma_hat - beta).T @ cov_hat @ (beta_gamma_hat - beta)
    else:
        beta_hat = beta_gamma_hat[:mx]
        statistic = (
            (beta_hat - beta).T
            @ np.linalg.inv(np.linalg.inv(cov_hat)[:mx, :mx])
            @ (beta_hat - beta)
        )

    statistic /= sigma_hat_sq

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx)

    return statistic, p_value


def inverse_wald_test(
    Z, X, y, alpha=0.05, W=None, estimator="tsls", fit_intercept=True
):
    """
    Return the quadric for the acceptance region based on asymptotic normality.

    If ``W`` is ``None``, the quadric is defined as

    .. math::

       (\\beta - \\hat{\\beta})^T (X^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X)^{-1} (\\beta - \\hat{\\beta}) \\leq \\hat{\\sigma}^2 F_{\\chi^2(m_X)}(1 - \\alpha),

    where :math:`\\hat \\beta` is an estimate of the causal parameter :math:`\\beta_0`
    (controlled by the parameter ``estimator``),
    :math:`\\hat \\sigma^2 = \\frac{1}{n - m_X} \\| y - X \\hat \\beta \\|^2_2`,
    :math:`P_Z` is the projection matrix onto the column space of :math:`Z`, :math:`M_Z`
    is the projection matrix onto the orthogonal complement of the column space of
    :math:`Z`, and :math:`F_{\\chi^2(m_X)}` is the cumulative distribution function of the
    :math:`\\chi^2(m_X)` distribution.

    If ``W`` is not ``None``, the quadric is defined as

    .. math::

       (\\beta - B \\hat{\\beta})^T (B ((X W)^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) (X W))^{-1} B^T)^{-1} (\\beta - B \\hat{\\beta}) \\leq \\hat{\\sigma}^2 F_{\\chi^2(m_X)}(1 - \\alpha),

    where :math:`B \\in \\mathbb{R}^{m_X \\times (m_X + m_W)}` is a diagonal matrix with
    1 on the diagonal and 0 elsewhere.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    alpha: float
        Significance level.
    W: np.ndarray of dimension (n, mw) or None, optional, default = None
        Endogenous regressors not of interest.
    estimator: float or str, optional, default = "tsls"
        Estimator to use. Passed as ``kappa`` parameter to ``KClass``.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. The intercept will be included both in the
        complete and the (restricted) model. Including an intercept is equivalent to
        centering the columns of all design matrices.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    n = Z.shape[0]

    Z, X, y, W, _ = _check_test_inputs(Z, X, y, W)

    z_alpha = scipy.stats.chi2.ppf(1 - alpha, df=X.shape[1])

    if fit_intercept:
        Z = Z - Z.mean(axis=0)
        X = X - X.mean(axis=0)
        W = W - W.mean(axis=0)
        y = y - y.mean()

    XW = np.concatenate([X, W], axis=1)

    kclass = KClass(kappa=estimator, fit_intercept=False).fit(XW, y, Z)
    beta = kclass.coef_

    residuals = y - kclass.predict(XW)
    hat_sigma_sq = np.sum(residuals**2) / (n - XW.shape[1] - fit_intercept)

    Xkappa = kclass.kappa_ * proj(Z, X) + (1 - kclass.kappa_) * X

    A = X.T @ Xkappa
    if W.shape[1] > 0:
        Wkappa = kclass.kappa_ * proj(Z, W) + (1 - kclass.kappa_) * X
        A = A - Xkappa.T @ W @ np.linalg.lstsq(Wkappa.T @ Wkappa, Wkappa.T @ Xkappa)

    b = -2 * A @ beta[: X.shape[1]]
    c = beta[: X.shape[1]].T @ A @ beta[: X.shape[1]] - hat_sigma_sq * z_alpha

    if isinstance(c, np.ndarray):
        c = c.item()

    if W is not None:
        return Quadric(A, b, c).project(range(XW.shape[1] - W.shape[1]))
    else:
        return Quadric(A, b, c)
