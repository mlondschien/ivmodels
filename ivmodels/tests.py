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
        ar = KClass(kappa="liml").fit(X=W, y=y - X @ beta, Z=Z).kappa_ - 1
        dfn = q - W.shape[1]

    p_value = 1 - scipy.stats.f.cdf(ar * (n - q) / dfn, dfn=dfn, dfd=n - q)

    return ar, p_value


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


def lagrange_multiplier_test(Z, X, y, beta):
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
    Z, X, y, _, beta = _check_test_inputs(Z, X, y, beta=beta)
    n, q = Z.shape
    p = X.shape[1]

    residuals = (y - X @ beta).reshape(-1, 1)
    proj_residuals = proj(Z, residuals)
    orth_residuals = residuals - proj_residuals

    # X - (y - X beta) * (y - X beta)^T M_Z X / (y - X beta)^T M_Z (y - X beta)
    X_tilde = X - residuals @ (orth_residuals.T @ X) / (residuals.T @ orth_residuals)
    proj_X_tilde = proj(Z, X_tilde)
    X_tilde_proj_residuals = proj(proj_X_tilde, residuals)
    # (y - X beta) P_{P_Z X_tilde} (y - X beta) / (y - X_beta) M_Z (y - X beta)
    statistic = np.square(X_tilde_proj_residuals).sum() / (residuals.T @ orth_residuals)
    statistic *= n - q

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=p)

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
        AR(\\beta) = \\min_\\gamma \\frac{n - q}{q - r} \\frac{\\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\gamma) \\|_2^2} \\leq F_(q - r, n-q)}(1 - \\alpha).


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
