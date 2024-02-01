import numpy as np
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.quadric import Quadric
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


def wald_test(Z, X, y, beta, W=None, estimator="tsls"):
    """
    Test based on asymptotic normality of the TSLS (or LIML) estimator.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       Wald := (\\beta - \\hat{\\beta})^T \\hat\\Cov(\\hat\\beta)^{-1} (\\beta - \\hat{\\beta}) / \\hat{\\sigma}^2,

    where :math:`\\hat \\beta = \\hat \\beta(\\kappa)` is a k-class estimator with
    :math:`\\sqrt{n} (1 - \\kappa) \\to 0`,
    :math:`\\hat\\Cov(\\hat\\beta)^{-1} = \\frac{1}{n} (X^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X)^{-1}`,
    :math:`\\hat \\sigma^2 = \\frac{1}{n - p} \\| y - X \\hat \\beta \\|^2_2` is an
    estimate of the variance of the errors, and :math:`P_Z` is the projection matrix
    onto the column space of :math:`Z`.
    Under strong instruments, the test statistic is asymptotically distributed as
    :math:`\\chi^2(p)` under the null.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::

        Wald := (\\beta - \\hat{\\beta})^T (D ( (X W)^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) (X W) )^{-1} D)^{-1} (\\beta - \\hat{\\beta}) / \\hat{\\sigma}^2,

    where :math:`D \\in \\mathbb{R}^{(p + r) \\times (p + r)}` is diagonal with
    :math:`D_{ii} = 1` if :math:`i \\leq p` and :math:`D_{ii} = 0` otherwise.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    W: np.ndarray of dimension (n, r) or None
        Endogenous regressors not of interest.
    beta: np.ndarray of dimension (p,)
        Coefficients to test.
    estimator: str
        Estimator to use. Must be one of ``"tsls"`` or ``"liml"``.

    Returns
    -------
    statistic: float
        The test statistic :math:`Wald`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(p)}(Wald)`, where
        :math:`F_{\\chi^2(p)}` is the cumulative distribution function of the
        :math:`\\chi^2(p)` distribution.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    """
    Z, X, y, W, beta = _check_test_inputs(Z, X, y, W=W, beta=beta)

    p = X.shape[1]

    if W is None:
        W = np.zeros((X.shape[0], 0))

    XW = np.concatenate([X, W], axis=1)

    estimator = KClass(kappa=estimator).fit(XW, y, Z)
    beta_gamma_hat = estimator.coef_

    sigma_hat_sq = np.mean(np.square(y - XW @ beta_gamma_hat))

    XW_proj = proj(Z, XW)

    kappa = estimator.kappa_
    cov_hat = (kappa * XW_proj + (1 - kappa) * XW).T @ XW

    if W.shape[1] == 0:
        statistic = (beta_gamma_hat - beta).T @ cov_hat @ (beta_gamma_hat - beta)
    else:
        beta_hat = beta_gamma_hat[:p]
        statistic = (
            (beta_hat - beta).T
            @ np.linalg.inv(np.linalg.inv(cov_hat)[:p, :p])
            @ (beta_hat - beta)
        )

    statistic /= sigma_hat_sq

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=X.shape[1])

    return statistic, p_value


def inverse_wald_test(Z, X, y, alpha=0.05, W=None, estimator="tsls"):
    """
    Return the quadric for the acceptance region based on asymptotic normality.

    If ``W`` is ``None``, the quadric is defined as

    .. math::

       (\\beta - \\hat{\\beta})^T X^T P_Z X (\\beta - \\hat{\\beta}) \\leq \\hat{\\sigma}^2 F_{\\chi^2(p)}(1 - \\alpha),

    where :math:`\\hat \\beta` is an estimate of the causal parameter :math:`\\beta_0`
    (controlled by the parameter ``estimator``),
    :math:`\\hat \\sigma^2 = \\frac{1}{n} \\| y - X \\hat \\beta \\|^2_2`,
    :math:`P_Z` is the projection matrix onto the column space of :math:`Z`,
    and :math:`F_{\\chi^2(p)}` is the cumulative distribution function of the
    :math:`\\chi^2(p)` distribution.

    If ``W`` is not ``None``, the quadric is defined as

    .. math::

       (\\beta - B \\hat{\\beta})^T (B ((X W)^T P_Z (X W))^{-1} B^T)^{-1} (\\beta - B \\hat{\\beta}) \\leq \\hat{\\sigma}^2 F_{\\chi^2(p)}(1 - \\alpha).

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    alpha: float
        Significance level.
    W: np.ndarray of dimension (n, r) or None
        Endogenous regressors not of interest.
    estimator: float or str, optional, default = "tsls"
        Estimator to use. Passed as ``kappa`` parameter to ``KClass``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, _ = _check_test_inputs(Z, X, y, W)

    z_alpha = scipy.stats.chi2.ppf(1 - alpha, df=X.shape[1])

    if W is not None:
        X = np.concatenate([X, W], axis=1)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    X_proj = proj(Z, X)

    kclass = KClass(kappa=estimator).fit(X, y, Z)
    beta = kclass.coef_

    # Avoid settings where (X @ beta).shape = (n, 1) and y.shape = (n,), resulting in
    # predictions.shape = (n, n) and residuals.shape = (n, n).
    predictions = X @ beta
    residuals = y.reshape(predictions.shape) - predictions
    hat_sigma_sq = np.mean(np.square(residuals))

    A = X.T @ (kclass.kappa_ * X_proj + (1 - kclass.kappa_) * X)
    b = -2 * A @ beta
    c = beta.T @ A @ beta - hat_sigma_sq * z_alpha

    if isinstance(c, np.ndarray):
        c = c.item()

    if W is not None:
        return Quadric(A, b, c).project(range(X.shape[1] - W.shape[1]))
    else:
        return Quadric(A, b, c)
