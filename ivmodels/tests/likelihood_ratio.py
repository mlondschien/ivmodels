import numpy as np
import scipy

from ivmodels.quadric import Quadric
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


def likelihood_ratio_test(Z, X, y, beta, W=None):
    """
    Perform the likelihood ratio test for ``beta``.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       \\mathrm{LR(\\beta)} &:= (n - q) \\frac{ \\| P_Z (y - X \\beta) \\|_2^2}{ \\| M_Z (y - X \\beta) \\|_2^2} - (n - q) \\frac{ \\| P_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 } \\\\
       &= q \\ \\mathrm{AR}(\\beta)) - q \\ \\mathrm{AR}(\\hat\\beta_\\mathrm{LIML}),

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z`,
    :math:`M_Z = \\mathrm{Id} - P_Z`, and :math:`\\hat\\beta_\\mathrm{LIML}` is the LIML
    estimator of :math:`\\beta`, minimizing the Anderson-Rubin test statistic
    :math:`\\mathrm{AR}(\\beta)` (see :py:func:`~ivmodels.tests.anderson_rubin_test`) at
    :math:`\\mathrm{AR}(\\hat\\beta_\\mathrm{LIML}) = \\frac{n - q}{q} (\\hat\\kappa_\\mathrm{LIML} - 1)`.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::

       \\mathrm{LR(\\beta)} := (n - q) \\frac{ \\|P_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|^2_2 }{\\| M_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2 } - (n - q) \\frac{\\| P_Z (y - (X \\ W) \\hat\\delta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - (X \\ W) \\hat\\delta_\\mathrm{LIML}) \\|_2^2}

    where :math:`\\gamma_\\mathrm{LIML}` is the LIML estimator (see
    :py:class:`~ivmodels.KClass`) using instruments :math:`Z`, endogenous
    covariates :math:`W`, and outcomes :math:`y - X \\beta` and
    :math:`\\hat\\delta_\\mathrm{LIML}` is the LIML estimator
    using instruments :math:`Z`, endogenous covariates :math:`X \\ W`, and outcomes :math:`y`.

    Under the null and given strong instruments, the test statistic is asymptotically
    distributed as :math:`\\chi^2(p)`, where :math:`p` is the number of regressors.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    beta: np.ndarray of dimension (p,)
        Coefficients to test.
    W: np.ndarray of dimension (n, r) or None
        Endogenous regressors not of interest.

    Returns
    -------
    statistic: float
        The test statistic :math:`LR`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(p)}(LR)`, where
        :math:`F_{\\chi^2(p)}` is the cumulative distribution function of the
        :math:`\\chi^2(p)` distribution.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.
    """
    Z, X, y, W, beta = _check_test_inputs(Z, X, y, W=W, beta=beta)

    n, p = X.shape
    q = Z.shape[1]

    if W is None:
        W = np.zeros((n, 0))

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)
    W_proj = proj(Z, W)

    XWy = np.concatenate([X, W, y.reshape(-1, 1)], axis=1)
    XWy_proj = np.concatenate([X_proj, W_proj, y_proj.reshape(-1, 1)], axis=1)

    matrix = np.linalg.solve(XWy.T @ (XWy - XWy_proj), XWy_proj.T @ XWy)
    ar_min = (n - q) * min(np.abs(scipy.linalg.eigvals(matrix)))

    if W.shape[1] == 0:
        statistic = (n - q) * np.linalg.norm(
            y_proj - X_proj @ beta
        ) ** 2 / np.linalg.norm((y - y_proj) - (X - X_proj) @ beta) ** 2 - ar_min
    else:
        Wy = np.concatenate([W, (y - X @ beta).reshape(-1, 1)], axis=1)
        Wy_proj = np.concatenate(
            [W_proj, (y_proj - X_proj @ beta).reshape(-1, 1)], axis=1
        )
        matrix = np.linalg.solve(Wy.T @ (Wy - Wy_proj), Wy_proj.T @ Wy)
        statistic = (n - q) * min(np.abs(scipy.linalg.eigvals(matrix))) - ar_min

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=p)

    return statistic, p_value


def inverse_likelihood_ratio_test(Z, X, y, alpha=0.05, W=None):
    """
    Return the quadric for the inverse likelihood ratio test's acceptance region.

    If ``W`` is ``None``, the quadric is defined as

    .. math::

       LR(\\beta) = (n - q) \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2} \\leq \\frac{1}{n} \\| y - X \\hat \\beta \\|^2_2 \\leq F_{\\chi^2(p)}(1 - \\alpha).

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

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, _ = _check_test_inputs(Z, X, y, W=W)

    n, p = X.shape
    q = Z.shape[1]

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    if W is not None:
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

    quantile = scipy.stats.chi2.ppf(1 - alpha, df=p) + (n - q) * kappa_liml

    A = X.T @ (X_proj - 1 / (n - q) * quantile * X_orth)
    b = -2 * (X_proj - 1 / (n - q) * quantile * X_orth).T @ y
    c = y.T @ (y_proj - 1 / (n - q) * quantile * y_orth)

    if isinstance(c, np.ndarray):
        c = c.item()

    if W is not None:
        return Quadric(A, b, c).project(range(X.shape[1] - W.shape[1]))
    else:
        return Quadric(A, b, c)
