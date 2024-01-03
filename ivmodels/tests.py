import numpy as np
import scipy

from ivmodels.kclass import KClass
from ivmodels.quadric import Quadric
from ivmodels.utils import proj


def _check_test_inputs(Z, X, y, W=None, beta=None):
    """
    Test dimensions of inputs to tests.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    W: np.ndarray of dimension (n, r), optional, default=None
        Regressors to control for.
    beta: np.ndarray of dimension (p,), optional, default=None
        Coefficients.

    Returns
    -------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    W: np.ndarray of dimension (n, r) or None
        Regressors to control for.
    beta: np.ndarray of dimension (p,) or None
        Coefficients.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be a matrix. Got shape {Z.shape}.")
    if X.ndim != 2:
        raise ValueError(f"X must be a matrix. Got shape {X.shape}.")
    if y.ndim != 1:
        if y.shape[1] != 1:
            raise ValueError(f"y must be a vector. Got shape {y.shape}.")
        else:
            y = y.flatten()

    if not Z.shape[0] == X.shape[0] == y.shape[0]:
        raise ValueError(
            f"Z, X, and y must have the same number of rows. Got shapes {Z.shape}, {X.shape}, and {y.shape}."
        )

    if beta is not None and beta.ndim != 1:
        if beta.shape[1] != 1:
            raise ValueError(f"beta must be a vector. Got shape {beta.shape}.")
        else:
            beta = beta.flatten()

        if beta.shape[0] != X.shape[1]:
            raise ValueError(
                f"beta must have the same length or number of rows as X has columns. Got shapes {beta.shape} and {X.shape}."
            )

    if W is not None:
        if W.ndim != 2:
            raise ValueError(f"W must be a matrix. Got shape {W.shape}.")
        if not W.shape[0] == X.shape[0]:
            raise ValueError(
                f"W and X must have the same number of rows. Got shapes {W.shape} and {X.shape}."
            )

    return Z, X, y, W, beta


def pulse_test(Z, X, y, beta):
    """
    Test proposed by :cite:t:`jakobsen2022distributional` with null hypothesis: :math:`Z` and :math:`y - X \\beta` are uncorrelated.

    The test statistic is defined as

    .. math:: T := n \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| (y - X \\beta) \\|_2^2}.

    Under the null, :math:`T` is asymptotically distributed as :math:`\\chi^2(q)`.
    See Section 3.2 of :cite:p:`jakobsen2022distributional` for details.

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

    Returns
    -------
    statistic: float
        The test statistic :math:`T`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(q)}(T)`, where
        :math:`F_\\chi^2(q)` is the cumulative distribution function of the
        :math:`\\chi^2(q)` distribution.

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

    n, q = Z.shape

    residuals = y - X @ beta
    proj_residuals = proj(Z, residuals)
    statistic = np.square(proj_residuals).sum() / np.square(residuals).sum()
    statistic *= n - q

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=q)
    return statistic, p_value


def wald_test(Z, X, y, beta, W=None, estimator="tsls"):
    """
    Test based on asymptotic normality of the TSLS (or LIML) estimator.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       Wald := (\\beta - \\hat{\\beta})^T X^T P_Z X (\\beta - \\hat{\\beta}) / \\hat{\\sigma}^2,

    where :math:`\\hat \\beta` is the TSLS (or LIML) estimator,
    :math:`\\hat \\sigma^2 = \\frac{1}{n - p} \\| y - X \\hat \\beta \\|^2_2` is an
    estimate of the variance of the errors, and :math:`P_Z` is the projection matrix
    onto the column space of :math:`Z`. If
    :math:`(X^T P_Z X)^{1/2}(\\hat \\beta - \\beta_0) \\overset{d}{\\to} \\mathcal{N}(0, \\sigma^2 \\mathrm{Id})`,
    for :math:`n \\to \\infty`, the test statistic is asymptotically distributed as
    :math:`\\chi^2(p)` under the null. This requires strong instruments.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::

        Wald := (\\beta - \\hat{\\beta})^T (D ( (X W)^T P_Z (X W) )^{-1} D)^{-1} (\\beta - \\hat{\\beta}) / \\hat{\\sigma}^2,

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
        :math:`F_\\chi^2(p)` is the cumulative distribution function of the
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

    if W.shape[1] == 0:
        statistic = (
            (beta_gamma_hat - beta).T @ XW_proj.T @ XW_proj @ (beta_gamma_hat - beta)
        )
    else:
        beta_hat = beta_gamma_hat[:p]
        statistic = (
            (beta_hat - beta).T
            @ np.linalg.inv(np.linalg.inv(XW_proj.T @ XW_proj)[:p, :p])
            @ (beta_hat - beta)
        )

    statistic /= sigma_hat_sq

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=X.shape[1])

    return statistic, p_value


def anderson_rubin_test(Z, X, y, beta, W=None):
    """
    Perform the Anderson Rubin test :cite:p:`anderson1949estimation`.

    Test the null hypothesis that the residuals are uncorrelated with the instruments.
    If ``W`` is ``None``, the test statistic is defined as

    .. math:: AR := \\frac{n - q}{q} \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2},

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z` and
    :math:`M_Z = \\mathrm{Id} - P_Z`.

    Under the null and normally distributed errors, this test statistic is distributed as
    :math:`F_{q, n - q}``, where :math:`q` is the number of instruments and :math:`n` is
    the number of observations. The statistic is asymptotically distributed as
    :math:`\\chi^2(q) / q` under the null and non-normally distributed errors, even for
    weak instruments.

    If ``W`` is not ``None``, the test statistic is

    .. math:: AR &:= \\min_\\gamma \\frac{n - q}{q - r} \\frac{\\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\gamma) \\|_2^2} \\\\
    &= \frac{n - q}{q - r} \\frac{\\| P_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2},

    where :math:`\\hat\\gamma_\\mathrm{LIML}` is the LIML estimate using instruments
    :math:`Z`, covariates :math:`W` and outcomes :math:`y - X \\beta`.
    Under the null, this test statistic is asymptotically distributed as
    :math:`\\frac{1}{q - r} \\chi^2(q - r)`, where :math:`r = \\mathrm{dim}(W)`. See
    :cite:p:`guggenberger2012asymptotic`.

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
        The test statistic :math:`AR`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{F_{q - r, n - q}}(AR)`, where
        :math:`F_{F_{q - r, n - q}}` is the cumulative distribution function of the
        :math:`F_{q - r, n - q}` distribution and ``r = 0`` if ``W`` is ``None``.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    References
    ----------
    .. bibliography::
       :filter: False

       anderson1949estimation
       guggenberger2012asymptotic
    """
    Z, X, y, W, beta = _check_test_inputs(Z, X, y, W=W, beta=beta)
    n, q = Z.shape

    if W is None:
        residuals = y - X @ beta
        proj_residuals = proj(Z, residuals)
        ar = (
            np.square(proj_residuals).sum()
            / np.square(residuals - proj_residuals).sum()
        )
        dfn = q
    else:
        ar = KClass.ar_min(X=W, y=y - X @ beta, Z=Z)
        dfn = q - W.shape[1]

    statistic = ar * (n - q)
    p_value = 1 - scipy.stats.f.cdf(statistic / dfn, dfn=dfn, dfd=n - q)

    return statistic, p_value


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
    :math:`\\mathrm{AR}(\\beta)` (see :py:func:`ivmodels.tests.anderson_rubin_test`) at
    :math:`\\mathrm{AR}(\\hat\\beta_\\mathrm{LIML}) = \\frac{n - q}{q} (\\hat\\kappa_\\mathrm{LIML} - 1)`.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::

       \\mathrm{LR(\\beta)} := (n - q) \\frac{ \\|P_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|^2_2 }{\\| M_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2 } - (n - q) \\frac{\\| P_Z (y - (X \\ W) \\hat\\delta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - (X \\ W) \\hat\\delta_\\mathrm{LIML}) \\|_2^2}

    where :math:`\\gamma_\\mathrm{LIML}` is the LIML estimator (see
    :py:class:`ivmodels.kclass.KClass`) using instruments :math:`Z`, endogenous
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


def _LM(X, X_proj, Y, Y_proj, W, W_proj, beta):
    """
    Compute the Lagrange multiplier test statistic and its derivative at ``beta``.

    Parameters
    ----------
    X: np.ndarray of dimension (n, p)
        Regressors.
    X_proj: np.ndarray of dimension (n, p)
        Projection of ``X`` onto the column space of ``Z``.
    Y: np.ndarray of dimension (n,)
        Outcomes.
    Y_proj: np.ndarray of dimension (n,)
        Projection of ``Y`` onto the column space of ``Z``.
    W: np.ndarray of dimension (n, r)
        Regressors.
    W_proj: np.ndarray of dimension (n, r)
        Projection of ``W`` onto the column space of ``Z``.
    beta: np.ndarray of dimension (p,)
        Coefficients to test.

    Returns
    -------
    lm: float
        The Lagrange multiplier test statistic.
    d_lm: float
        The derivative of the Lagrange multiplier test statistic at ``beta``.
    """
    n = X.shape[0]

    residuals = Y - X @ beta
    residuals_proj = Y_proj - X_proj @ beta
    residuals_orth = residuals - residuals_proj

    sigma_hat = residuals_orth.T @ residuals_orth

    XW_proj = np.hstack((X_proj, W_proj))
    XW = np.hstack((X, W))
    Sigma = residuals_orth.T @ XW / sigma_hat
    St = XW - np.outer(residuals, Sigma)
    St_proj = XW_proj - np.outer(residuals_proj, Sigma)

    solved = np.linalg.solve(St_proj.T @ St_proj, St_proj.T @ residuals_proj)
    residuals_proj_St = St_proj @ solved

    lm = residuals_proj_St.T @ residuals_proj_St / sigma_hat

    first_term = -2 * residuals_proj.T @ St_proj[:, : X.shape[1]] * sigma_hat
    second_term = (
        2
        * residuals_proj.T
        @ (residuals_proj - residuals_proj_St)
        * (X - X_proj).T
        @ St
        @ solved
    )
    d_lm = (first_term + second_term) / (sigma_hat**2)
    return (n * lm, n * d_lm)


def lagrange_multiplier_test(Z, X, y, beta, W=None):
    """
    Perform the Lagrange multiplier test for ``beta`` by :cite:t:`kleibergen2002pivotal`.

    Test the null hypothesis that the residuals are uncorrelated with the instruments.
    Let

    .. math:: \\tilde X(\\beta) := X - (y - X \\beta) \\frac{(y - X \\beta) M_Z X}{(y - X \\beta) M_Z (y - X \\beta)}.

    The test statistic is

    .. math:: LM := (n - q) \\frac{\\| P_{P_Z \\tilde X(\\beta)} (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2},

    This test statistic is asymptotically distributed as :math:`\\chi^2(p)` under the
    null, even if the instruments are weak.

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

    Returns
    -------
    statistic: float
        The test statistic :math:`LM`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(p)}(LM)`, where
        :math:`F_{\\chi^2(p)}` is the cumulative distribution function of the
        :math:`\\chi^2(p)` distribution.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.
    """
    Z, X, y, W, beta = _check_test_inputs(Z, X, y, beta=beta, W=W)

    n, q = Z.shape
    p = X.shape[1]

    if W is not None and W.shape[1] > 0:
        gamma_hat = KClass(kappa="liml").fit(X=W, y=y - X @ beta, Z=Z).coef_
        res = scipy.optimize.minimize(
            lambda gamma: _LM(
                X=W,
                X_proj=proj(Z, W),
                Y=y - X @ beta,
                Y_proj=proj(Z, y - X @ beta),
                W=X,
                W_proj=proj(Z, X),
                beta=gamma,
            ),
            jac=True,
            x0=gamma_hat,
        )

        res2 = scipy.optimize.minimize(
            lambda gamma: _LM(
                X=W,
                X_proj=proj(Z, W),
                Y=y - X @ beta,
                Y_proj=proj(Z, y - X @ beta),
                W=X,
                W_proj=proj(Z, X),
                beta=gamma,
            ),
            jac=True,
            x0=np.zeros_like(gamma_hat),
        )

        statistic = min(res.fun, res2.fun) / n

        statistic *= n - q

        p_value = 1 - scipy.stats.chi2.cdf(statistic, df=p)

    else:
        residuals = (y - X @ beta).reshape(-1, 1)
        proj_residuals = proj(Z, residuals)
        orth_residuals = residuals - proj_residuals

        # X - (y - X beta) * (y - X beta)^T M_Z X / (y - X beta)^T M_Z (y - X beta)
        X_tilde = X - residuals @ (orth_residuals.T @ X) / (
            residuals.T @ orth_residuals
        )
        proj_X_tilde = proj(Z, X_tilde)
        X_tilde_proj_residuals = proj(proj_X_tilde, residuals)
        # (y - X beta) P_{P_Z X_tilde} (y - X beta) / (y - X_beta) M_Z (y - X beta)
        statistic = np.square(X_tilde_proj_residuals).sum() / (
            residuals.T @ orth_residuals
        )

        statistic *= n - q

        p_value = 1 - scipy.stats.chi2.cdf(statistic, df=p)

    return statistic, p_value


def conditional_likelihood_ratio_test(Z, X, y, beta, W=None):
    """
    Perform the conditional likelihood ratio test for ``beta``.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       \\mathrm{CLR(\\beta)} &:= (n - q) \\frac{ \\| P_Z (y - X \\beta) \\|_2^2}{ \\| M_Z (y - X \\beta) \\|_2^2} - (n - q) \\frac{ \\| P_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 } \\\\
       &= q \\ \\mathrm{AR}(\\beta) - q \\ \\min_\\beta \\mathrm{AR}(\\beta),

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z`,
    :math:`M_Z = \\mathrm{Id} - P_Z`, and :math:`\\hat\\beta_\\mathrm{LIML}` is the LIML
    estimator of :math:`\\beta` (see :py:class:`ivmodels.kclass.KClass`), minimizing the
    Anderson-Rubin test statistic :math:`\\mathrm{AR}(\\beta)`
    (see :py:func:`ivmodels.tests.anderson_rubin_test`) at

    .. math:: \\mathrm{AR}(\\hat\\beta_\\mathrm{LIML}) = \\frac{n - q}{q} \\lambda_\\mathrm{min}( (X \\ y)^T M_Z (X \\ y))^{-1} (X \\ y)^T P_Z (X \\ y) ).

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::
       \\mathrm{CLR(\\beta)} &:= (n - q) \\min_\\gamma \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2} - (n - q) \\min_{\\beta, \\gamma} \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2 }{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2 } \\\\
       &= (n - q) \\frac{ \\| P_Z (y - X \\beta - W \\hat\\gamma_\\textrm{liml}) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\hat\\gamma_\\textrm{liml}) \\|_2^2} - (n - q) \\frac{ \\| P_Z (y - (X \\ W) \\hat\\delta_\\mathrm{liml}) \\|_2^2 }{ \\| M_Z (y - (X \\ W) \\hat\\delta_\\mathrm{liml}) \\|_2^2 },

    where :math:`\\hat\\gamma_\\mathrm{LIML}` is the LIML estimator of :math:`\\gamma`
    (see :py:class:`ivmodels.kclass.KClass`) using instruments :math:`Z`, endogenous
    covariates :math:`W`, and outcomes :math:`y - X \\beta` and
    :math:`\\hat\\delta_\\mathrm{LIML}` is the LIML estimator of
    :math:`(\\beta, \\gamma)` using instruments :math:`Z`, endogenous covariates
    :math:`(X \\ W)`, and outcomes :math:`y`.

    Let

    .. math:: \\tilde X(\\beta) := X - (y - X \\beta) \\cdot \\frac{(y - X \\beta)^T M_Z X}{(y - X \\beta)^T M_Z (y - X \\beta)}

    and

    .. math:: s_\\mathrm{min}(\\beta) := (n - q) \\cdot \\lambda_\\mathrm{min}((\\tilde X(\\beta)^T M_Z \\tilde X(\\beta))^{-1} \\tilde X(\\beta)^T P_Z \\tilde X(\\beta)).

    Then, conditionally on :math:`s_\\mathrm{min}(\\beta_0)`, the statistic
    :math:`\\mathrm{CLR(\\beta_0)}` is asymptotically bounded from above by a random
    variable that is distributed as

    .. math:: \\frac{1}{2} \\left( Q_p + Q_{q - p - r} - Q_r - s_\\mathrm{min} + \\sqrt{ (Q_p + Q_r + Q_{q - p - r})^2 - 4 Q_{q - p - r} s_\\textrm{min} } \\right),

    where :math:`Q_p \\sim \\chi^2(p)`, :math:`Q_r \\sim \\chi^2(r)`, and
    :math:`Q_{q - p - r} \\sim \\chi^2(q - p - r)` are independent chi-squared random
    variables. This is robust to weak instruments. If identification is strong, that is
    :math:`s_\\mathrm{min}(\\beta_0) \\to \\infty`, the conditional likelihood ratio
    test is equivalent to the likelihood ratio test
    (see :py:func:`ivmodels.tests.likelihood_ratio_test`).
    See :cite:p:`moreira2003conditional` for details.

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
        The test statistic :math:`CLR`.
    p_value: float
        The p-value of the test, computed using a Monte Carlo simulation.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.
    """
    Z, X, y, W, beta = _check_test_inputs(Z, X, y, beta=beta, W=W)

    n, q = Z.shape
    p = X.shape[1]
    r = W.shape[1] if W is not None else 0

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)

    if r == 0:
        residuals = y - X @ beta
        residuals_proj = y_proj - X_proj[:, :p] @ beta
        residuals_orth = residuals - residuals_proj

        Sigma = (residuals_orth.T @ X) / (residuals_orth.T @ residuals_orth)
        Xt = X - np.outer(residuals, Sigma)
        Xt_proj = X_proj - np.outer(residuals_proj, Sigma)
        Xt_orth = Xt - Xt_proj
        mat_Xt = np.linalg.solve(Xt_orth.T @ Xt_orth, Xt_proj.T @ Xt_proj)
        s_min = min(np.real(np.linalg.eigvals(mat_Xt))) * (n - q)

        # TODO: This can be done with efficient rank-1 updates.
        ar_min = KClass.ar_min(X=X, y=y, Z=Z)
        ar = residuals_proj.T @ residuals_proj / (residuals_orth.T @ residuals_orth)

        statistic = (n - q) * (ar - ar_min)

    elif r > 0:
        W_proj = proj(Z, W)
        XWy = np.concatenate([X, W, y.reshape(-1, 1)], axis=1)
        XWy_proj = np.concatenate([X_proj, W_proj, y_proj.reshape(-1, 1)], axis=1)

        XWy_eigenvals = np.sort(
            np.real(
                scipy.linalg.eigvals(
                    np.linalg.solve((XWy - XWy_proj).T @ XWy, XWy_proj.T @ XWy)
                )
            )
        )
        ar = KClass().ar_min(
            X=W, y=y - X @ beta, X_proj=W_proj, y_proj=y_proj - X_proj @ beta
        )

        statistic = (n - q) * (ar - XWy_eigenvals[0])

        s_min = XWy_eigenvals[0] + XWy_eigenvals[1] - ar

    chi2p = scipy.stats.chi2.rvs(size=1000, df=p, random_state=0)
    if q - p > 0:
        chi2q = scipy.stats.chi2.rvs(size=1000, df=q - p, random_state=2)
    else:
        chi2q = 0

    chiqppms = chi2p + chi2q - s_min
    D = np.sqrt(chiqppms**2 + 4 * chi2p * s_min)
    Q = 1 / 2 * (chiqppms + D)
    p_value = np.mean(Q > statistic)

    return statistic, p_value


def inverse_pulse_test(Z, X, y, alpha=0.05):
    """Return the quadric for the inverse pulse test's acceptance region."""
    Z, X, y, _, _ = _check_test_inputs(Z, X, y)

    n, q = Z.shape

    quantile = scipy.stats.chi2.ppf(1 - alpha, df=q)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)

    A = X.T @ (X_proj - 1 / (n - q) * quantile * X)
    b = -2 * (X_proj - 1 / (n - q) * quantile * X).T @ y
    c = y.T @ (y_proj - 1 / (n - q) * quantile * y)

    if isinstance(c, np.ndarray):
        c = c.item()

    return Quadric(A, b, c)


def inverse_anderson_rubin_test(Z, X, y, alpha=0.05, W=None):
    """
    Return the quadric for to the inverse Anderson-Rubin test's acceptance region.

    The returned quadric satisfies ``quadric(x) <= 0`` if and only if
    ``anderson_rubin_test(Z, X, y, W=W)[1] > alpha``. It is thus a confidence region
    for the causal parameter corresponding to the endogenous regressors of interest
    ``X``.

    If ``W`` is ``None``, let :math:`q := \\frac{q}{n-q}F_{F(q, n-q)}(1 - \\alpha)`, where
    :math:`F_{F(q, n-q)}` is the cumulative distribution function of the
    :math:`F(q, n-q)` distribution. The quadric is defined as

    .. math::

       AR(\\beta) = \\frac{n - q}{q} \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2} \\leq F_{F(q, n-q)}(1 - \\alpha) \\\\
       \\Leftrightarrow \\beta^T X^T (P_Z - q M_Z) X \\beta - 2 y^T (P_Z - q M_Z) X \\beta + y^T (P_Z - q M_Z) y \\leq 0.

    If ``W`` is not ``None``, let :math:`q := \\frac{q - r}{n-q}F_{F(q - r, n-q)}(1 - \\alpha)`.
    The quadric is defined as

    .. math::
        AR(\\beta) = \\min_\\gamma \\frac{n - q}{q - r} \\frac{\\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\gamma) \\|_2^2} \\leq F_{q - r, n-q}(1 - \\alpha).


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

    Returns
    -------
    Quadric
        The quadric for the acceptance region.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, _ = _check_test_inputs(Z, X, y, W=W)

    n, q = Z.shape

    if W is not None:
        X = np.concatenate([X, W], axis=1)
        dfn = q - W.shape[1]
    else:
        dfn = q

    quantile = scipy.stats.f.ppf(1 - alpha, dfn=dfn, dfd=n - q) * dfn / (n - q)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    X_proj = proj(Z, X)
    X_orth = X - X_proj
    y_proj = proj(Z, y)
    y_orth = y - y_proj

    A = X.T @ (X_proj - quantile * X_orth)
    b = -2 * (X_proj - quantile * X_orth).T @ y
    c = y.T @ (y_proj - quantile * y_orth)

    if isinstance(c, np.ndarray):
        c = c.item()

    if W is not None:
        return Quadric(A, b, c).project(range(X.shape[1] - W.shape[1]))

    else:
        return Quadric(A, b, c)


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

    beta = KClass(kappa=estimator).fit(X, y, Z).coef_
    # Avoid settings where (X @ beta).shape = (n, 1) and y.shape = (n,), resulting in
    # predictions.shape = (n, n) and residuals.shape = (n, n).
    predictions = X @ beta
    residuals = y.reshape(predictions.shape) - predictions
    hat_sigma_sq = np.mean(np.square(residuals))

    A = X.T @ X_proj
    b = -2 * A @ beta
    c = beta.T @ A @ beta - hat_sigma_sq * z_alpha

    if isinstance(c, np.ndarray):
        c = c.item()

    if W is not None:
        return Quadric(A, b, c).project(range(X.shape[1] - W.shape[1]))
    else:
        return Quadric(A, b, c)


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


def bounded_inverse_anderson_rubin(Z, X):
    """
    Return the largest p-value such that the inverse-AR confidence set is unbounded.

    In practice, the confidence set might be unbounded for ``1.001 * p`` only.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors.
    """
    n, q = Z.shape

    X = X - X.mean(axis=0)

    X_proj = proj(Z, X)

    W = np.linalg.solve(X.T @ (X - X_proj), X.T @ X_proj)
    kappa = min(np.real(np.linalg.eigvals(W)))

    cdf = scipy.stats.f.cdf((n - q) / q * kappa, dfn=q, dfd=n - q)
    return 1 - cdf
