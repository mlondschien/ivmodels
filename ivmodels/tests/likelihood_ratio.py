import numpy as np
import scipy

from ivmodels.quadric import Quadric
from ivmodels.utils import _characteristic_roots, _check_inputs, oproj, proj


def likelihood_ratio_test(Z, X, y, beta, W=None, C=None, D=None, fit_intercept=True):
    """
    Perform the likelihood ratio test for ``beta``.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       \\mathrm{LR}(\\beta) &:= (n - k) \\frac{ \\| P_Z (y - X \\beta) \\|_2^2}{ \\| M_Z (y - X \\beta) \\|_2^2} - (n - k) \\frac{ \\| P_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 } \\\\
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
    beta: np.ndarray of dimension (mx + md,)
        Coefficients to test.
    W: np.ndarray of dimension (n, mw) or None, optional, default=None
        Endogenous regressors not of interest.
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md) or None, optional, default=None
        Exogenous regressors of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\mathrm{LR}(\\beta)`.
    p_value: float
        The p-value of the test. Equal to
        :math:`1 - F_{\\chi^2(m_X)}(\\mathrm{LR}(\\beta)`, where
        :math:`F_{\\chi^2(m_X)}` is the cumulative distribution function of the
        :math:`\\chi^2(m_X)` distribution.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.
    """
    Z, X, y, W, C, D, beta = _check_inputs(Z, X, y, W=W, C=C, D=D, beta=beta)

    n, k = Z.shape
    mx, mw, mc, md = X.shape[1], W.shape[1], C.shape[1], D.shape[1]

    if k < mx + mw:
        raise ValueError(
            "The number of instruments must be at least the number of endogenous "
            "regressors."
        )

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W, D = oproj(C, X, y, Z, W, D)

    if md > 0:
        Z = np.concatenate([Z, D], axis=1)

    X_proj, y_proj, W_proj = proj(Z, X, y, W)

    XWy = np.concatenate([X, W, y.reshape(-1, 1)], axis=1)

    XWy_proj = np.hstack([X_proj, W_proj, y_proj.reshape(-1, 1)])

    if k == mx + mw:
        ar_min = 0
    else:
        ar_min = _characteristic_roots(
            a=oproj(D, XWy).T @ XWy_proj,
            b=XWy.T @ (XWy - XWy_proj),
            subset_by_index=[0, 0],
        )[0]

    if md > 0:
        X = np.hstack([X, D])
        X_proj = np.hstack([X_proj, D])

    residuals = y - X @ beta
    residuals_proj = y_proj - X_proj @ beta

    if mw == 0:
        statistic = np.linalg.norm(residuals_proj) ** 2
        statistic /= np.linalg.norm(residuals - residuals_proj) ** 2
        statistic -= ar_min

        statistic *= n - k - mc - md - fit_intercept
    else:
        Wy = np.hstack([W, residuals.reshape(-1, 1)])
        Wy_proj = np.hstack([W_proj, residuals_proj.reshape(-1, 1)])

        statistic = (
            np.real(
                _characteristic_roots(
                    a=Wy_proj.T @ Wy_proj,
                    b=Wy.T @ (Wy - Wy_proj),
                    subset_by_index=[0, 0],
                )[0]
            )
            - ar_min
        )
        statistic *= n - k - mc - md - fit_intercept

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx + md)

    return statistic, p_value


def inverse_likelihood_ratio_test(
    Z, X, y, alpha=0.05, W=None, C=None, D=None, fit_intercept=True
):
    """
    Return the quadric for the inverse likelihood ratio test's acceptance region.

    The quadric is defined as

    .. math::

       \\mathrm{LR}(\\beta) \\leq F_{\\chi^2(m_X)}(1 - \\alpha).

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
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md) or None, optional, default=None
        Exogenous regressors of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, C, D, _ = _check_inputs(Z, X, y, W=W, C=C, D=D)

    n, k = Z.shape
    mx, mw, mc, md = X.shape[1], W.shape[1], C.shape[1], D.shape[1]

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W, D = oproj(C, X, y, Z, W, D)

    if md > 0:
        Z = np.concatenate([Z, D], axis=1)
        X = np.hstack([X, D])

    XWy = np.concatenate([X, W, y.reshape(-1, 1)], axis=1)

    XWy_proj = proj(Z, XWy)
    XWy_orth = XWy - XWy_proj

    a = XWy_proj.T @ XWy_proj
    b = XWy_orth.T @ XWy_orth

    if k == mx + mw:
        kappa_liml = 0
    else:
        kappa_liml = np.real(
            _characteristic_roots(
                a=a,
                b=b,
                subset_by_index=[0, 0],
            )[0]
        )

    dfd = n - k - mc - md - fit_intercept
    quantile = scipy.stats.chi2.ppf(1 - alpha, df=mx + md) + dfd * kappa_liml

    R = a - 1 / dfd * quantile * b

    A = R[: (mx + md + mw), : (mx + md + mw)]
    b = -2 * R[: (mx + md + mw), (mx + md + mw)]
    c = R[(mx + md + mw), (mx + md + mw)]

    return Quadric(A, b, c).project(np.arange(mx + md))
