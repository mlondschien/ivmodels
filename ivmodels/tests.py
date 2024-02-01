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
        :math:`F_{\\chi^2(q)}` is the cumulative distribution function of the
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

       Wald := (\\beta - \\hat{\\beta})^T \\hat\\Cov(\\hat\\beta)^{-1} (\\beta - \\hat{\\beta}) / \\hat{\\sigma}^2,

    where :math:`\\hat \\beta = \\hat \\beta(\\kappa)` is the k-class estimator given by
    ``estimator`` with parameter :math:`\\kappa`,
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


def more_powerful_subvector_anderson_rubin_critical_value_function(
    z, kappa_1_hat, k, mW
):
    """
    Implement the critical value function proposed by :cite:t`guggenberger2019more`.

    This is done by numerically integrating the approximate conditional density from
    Equation (A.21) of :cite:p:`guggenberger2019more`.

    Parameters
    ----------
    z: float
        The test statistic.
    kappa_1_hat: float
        The maximum eigenvalue of the matrix
        :math:`M:=((X \\ y- X \\beta)^T M_Z (X \\ y - X \\beta))^{-1} (X \\ y-X \\beta)^T P_Z (X \\ y-X \\beta)`. This is the conditioning statistic.
    k: int
        Number of instruments.
    mW: int
        Number of endogenous regressors not of interest, i.e., :math:`\\mathrm{dim}(W)`.

    Returns
    -------
    p_value: float
        The (approximate) p-value of observing :math:`\\lambda_\\mathrm{min}(M) > z` given :math:`\\lambda_\\mathrm{max}(M) = \\hat\\kappa_1` under the null
        :math:`\\beta = \\beta_0`.

    References
    ----------
    .. bibliography::
       :filter: False

       guggenberger2019more

    """
    # Code from them:
    # decl g_kap1hat;						// value of conditioning statistic, largest eigenvalue of Wishart matrix
    # decl g_k;							// # of instruments
    # decl g_b = 1e6;						// constant to approximate the 1F1 function from 2F1 (must be very large)

    # d1F1 = hyper_2F1((g_k-1)/2, g_b, (g_k+2)/2, -g_kap1hat/2/g_b);
    # if(g_kap1hat > 1000 && d1F1 == .Inf)		   		// when the argument is large, numerical evaluation of 1F1 can fail (and it does!)
    # d1F1 = exp(g_kap1hat)/sqrt(2*M_PI*g_kap1hat);   // in those cases, use the asymptotic expression lim_x->\infty [1F1(a,b;x) = exp(x)/sqrt(2*M_PI*x) (Olver, 1974, p. 435)
    # return gammafact((g_k+2)/2) / gammafact((g_k-1)/2) * 2 .* exp(-x/2)
    # *g_kap1hat^(-g_k/2) .* x.^((g_k-3)/2) .* sqrt(g_kap1hat-x)/sqrt(M_PI)
    # /d1F1;

    if z > kappa_1_hat:
        raise ValueError("z must be smaller than kappa_1_hat")

    # Page 494, footnote 3: "For general mW, discussed in the next subsection, the role
    # of k − 1 is played by k − mW"
    # Thus, k - 1 <- k - mW or k <- k - mW + 1
    k_prime = k - mW + 1

    # Equation A.22
    # g = scipy.special.gamma(k_prime / 2 + 1) * np.square(k_prime / 2 + 0.5) / np.power(kappa_1_hat, k_prime / 2) / np.sqrt(np.pi) / scipy.special.hyp1f1(k_prime/2 - 0.5, k_prime/2 + 1, - kappa_1_hat / 2)

    # This code would copy their code in Python
    # hyp1f1 = scipy.special.hyp2f1(k_prime/2 - 0.5, 1e6, k_prime/2 + 1, - kappa_1_hat / 2 / 1e6)
    # if not np.isfinite(hyp1f1):
    #     hyp1f1 = np.exp(kappa_1_hat) / np.sqrt(2 * np.pi * kappa_1_hat)

    hyp1f1 = scipy.special.hyp1f1(k_prime / 2 - 0.5, k_prime / 2 + 1, -kappa_1_hat / 2)

    const = (
        scipy.special.gamma(k_prime / 2 + 1)
        * 2
        / scipy.special.gamma(k_prime / 2 - 0.5)
        / np.power(kappa_1_hat, k_prime / 2)
        / np.sqrt(np.pi)
        / hyp1f1
    )

    def f(x):
        return (
            np.exp(-x / 2) * np.power(x, k_prime / 2 - 1.5) * np.sqrt(kappa_1_hat - x)
        )

    return 1 - scipy.integrate.quad(f, 0, z, limit=50)[0] * const


def anderson_rubin_test(Z, X, y, beta, W=None, critical_values="chi2"):
    """
    Perform the Anderson Rubin test :cite:p:`anderson1949estimation`.

    Test the null hypothesis that the residuals are uncorrelated with the instruments.
    If ``W`` is ``None``, the test statistic is defined as

    .. math:: AR := \\frac{n - q}{q} \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2},

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z` and
    :math:`M_Z = \\mathrm{Id} - P_Z`.

    Under the null and normally distributed errors, this test statistic is distributed as
    :math:`F_{q, n - q}`, where :math:`q` is the number of instruments and :math:`n` is
    the number of observations. The statistic is asymptotically distributed as
    :math:`\\chi^2(q) / q` under the null and non-normally distributed errors, even for
    weak instruments.

    If ``W`` is not ``None``, the test statistic is

    .. math::

       AR &:= \\min_\\gamma \\frac{n - q}{q - r} \\frac{\\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\gamma) \\|_2^2} \\\\
       &= \\frac{n - q}{q - r} \\frac{\\| P_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2},

    where :math:`\\hat\\gamma_\\mathrm{LIML}` is the LIML estimate using instruments
    :math:`Z`, covariates :math:`W` and outcomes :math:`y - X \\beta`.
    Under the null, this test statistic is asymptotically bounded from above by a random
    variable that is distributed as
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
    critical_values: str
        If ``"chi2"``, use the :math:`\\chi^2(q)` distribution to compute the p-value.
        If ``"guggenberger2019"``, use the critical value function proposed by
        :cite:t:`guggenberger2019more` to compute the p-value. Only relevant if ``W`` is
        not ``None``.

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
       guggenberger2019more
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
        spectrum = KClass._spectrum(X=W, y=y - X @ beta, Z=Z)
        ar = np.min(spectrum)
        dfn = q - W.shape[1]

    statistic = ar * (n - q)

    if W is None or critical_values == "chi2":
        p_value = 1 - scipy.stats.f.cdf(statistic / dfn, dfn=dfn, dfd=n - q)
    else:
        kappa_max = (n - q) * np.max(spectrum)
        p_value = more_powerful_subvector_anderson_rubin_critical_value_function(
            statistic, kappa_max, q, W.shape[1]
        )

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
    If ``W`` is ``None``, let

    .. math:: \\tilde X(\\beta) := X - (y - X \\beta) \\frac{(y - X \\beta) M_Z X}{(y - X \\beta) M_Z (y - X \\beta)}.

    The test statistic is

    .. math:: LM := (n - q) \\frac{\\| P_{P_Z \\tilde X(\\beta)} (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2}.

    If ``W`` is not ``None``, let

    .. math:: \\tilde S(\\beta, \\gamma) := (X \\ W) - (y - X \\beta - W \\gamma) \\frac{(y - X \\beta - W \\gamma) M_Z (X \\ W)}{(y - X \\beta - W \\gamma) M_Z (y - X \\beta - W \\gamma)}.

    The test statistic is

    .. math:: LM := (n - q) \\min_{\\gamma} \\frac{\\| P_{P_Z \\tilde S(\\beta, \\gamma)} (y - X \\beta - W \\gamma) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\gamma) \\|_2^2}.

    This test statistic is asymptotically distributed as :math:`\\chi^2(p)` under the
    null, even if the instruments are weak.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    X: np.ndarray of dimension (n, p)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    beta: np.ndarray of dimension (p,)
        Coefficients to test.
    W: np.ndarray of dimension (n, r) or None
        Endogenous regressors not of interest.

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


def _conditional_likelihood_ratio_critical_value_function(p, q, s_min, z, tol=1e-6):
    """
    Approximate the critical value function of the conditional likelihood ratio test.

    Let

    .. math: Z = 1/2 \\left( Q_{q-p} + Q_p - s_\\mathrm{min} + \\sqrt{ (Q_{q-p} + Q_p - s_\\mathrm{min})^2 + 4 Q_{p} s_\\mathrm{min} } \\right),

    where :math:`Q_p \\sim \\chi^2(p)` and :math:`Q_{q-p} \\sim \\chi^2(q - p)` are
    independent chi-squared random variables. This function approximates

    .. math: \\mathbb{P}[ Z > z ]

    up to tolerance ``tol``.

    Uses a formualtion by :cite:p:`hillier2009conditional` to approximate the critical
    value function of the conditional likelihood ratio test. In particular, computes the
    first terms of Equation (28) of :cite:p:`hillier2009conditional` which is equal to
    Equation (41) of :cite:p:`hillier2009exact`.

    .. math: Q_{k, p} = (1 - a)^{p / 2} \\sum_{j = 0}^\\infty a^j \\frac{(p / 2)_j}{j!} \\F_{k + 2 j}(z + s_{\\min}),

    where :math:`(x)_j` is the Pochhammer symbol, defined as
    :math:`(x)_j = x (x + 1) ... (x + j - 1)`, :math:`\\F_k` is the cumulative
    distribution function of the :math:`\\chi^2(k)` distribution, and
    :math:`a = s_{\\min} / (z + s_{\\min})`.

    References
    ----------
    .. bibliography::
       :filter: False

       hillier2009conditional
    """
    if z <= 0:
        return 1

    if s_min <= 0:
        return 1 - scipy.stats.chi2(q).cdf(z)

    a = s_min / (z + s_min)

    p_value = 0

    # Equal to (1 - a)^{p / 2} * a^j * (m/2)_j / j!, where (x)_j is the Pochhammer
    # symbol, defined as (x)_j = x (x + 1) ... (x + j - 1). See end of Section 1.0 of
    # Hillier's "Exact properties of the conditional likelihood ratio test in an IV
    # regression model"
    factor = 1
    j = 0

    # In the Appendix of Hillier's paper, they show that the error when truncating the
    # infinite sum at j = J is bounded by a^J * (m/2)_J / J!, which is equal to
    # `factor` here. However, their claim
    # "F_1(r+1, 1 - m/2, r+2, a) is less than 1 for 0 <= a <= 1"
    # is incorrect for m = 1, where F_1(r+1, 1 - m/2, r+2, a) <= 1 / (1 - a) via the
    # geometric series. Thus, the error is bounded by a^J * (m/2)_J / J! / (1 - a).
    # As G_k(z + l), the c.d.f of a chi^2(k), is decreasing in k, one can
    # keep the term G_{k + 2J}(z + l) from the first sum. Thus, we can stop when
    # `delta / (1 - a) = factor * G_{k + 2J}(z + l) / (1 - a)` is smaller than the
    # desired tolerance.
    delta = scipy.stats.chi2(q).cdf(z + s_min)
    p_value += delta

    sqrt_minus_log_a = np.sqrt(-np.log(a))
    tol = tol / (1 + (1 - scipy.special.erf(sqrt_minus_log_a)) / sqrt_minus_log_a)

    while delta >= tol:
        factor *= (p / 2 + j) / (j + 1) * a
        delta = scipy.stats.chi2(q + 2 * j + 2).cdf(z + s_min) * factor

        p_value += delta

        j += 1
        if j > 1000:
            raise RuntimeError("Failed to converge.")

    p_value *= (1 - a) ** (p / 2)

    return 1 - p_value


def conditional_likelihood_ratio_test(Z, X, y, beta, W=None, tol=1e-6):
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

    Let

    .. math:: \\tilde X(\\beta) := X - (y - X \\beta) \\cdot \\frac{(y - X \\beta)^T M_Z X}{(y - X \\beta)^T M_Z (y - X \\beta)}

    and

    .. math:: s_\\mathrm{min}(\\beta) := (n - q) \\cdot \\lambda_\\mathrm{min}((\\tilde X(\\beta)^T M_Z \\tilde X(\\beta))^{-1} \\tilde X(\\beta)^T P_Z \\tilde X(\\beta)).

    Then, conditionally on :math:`s_\\mathrm{min}(\\beta_0)`, the statistic
    :math:`\\mathrm{CLR(\\beta_0)}` is asymptotically bounded from above by a random
    variable that is distributed as

    .. math:: \\frac{1}{2} \\left( Q_p + Q_{q - p} - s_\\mathrm{min} + \\sqrt{ (Q_p + Q_{q - p}  - s_\\mathrm{min})^2 + 4 Q_{p} s_\\textrm{min} } \\right),

    where :math:`Q_p \\sim \\chi^2(p)` and
    :math:`Q_{q - p} \\sim \\chi^2(q - p)` are independent chi-squared random
    variables. This is robust to weak instruments. If identification is strong, that is
    :math:`s_\\mathrm{min}(\\beta_0) \\to \\infty`, the conditional likelihood ratio
    test is equivalent to the likelihood ratio test
    (see :py:func:`ivmodels.tests.likelihood_ratio_test`), with :math:`\\chi^2(p)`
    limiting distribution.
    If identification is weak, that is :math:`s_\\mathrm{min}(\\beta_0) \\to 0`, the
    conditional likelihood ratio test is equivalent to the Anderson-Rubin test
    (see :py:func:`ivmodels.tests.anderson_rubin_test`) with :math:`\\chi^2(q)` limiting
    distribution.
    See :cite:p:`moreira2003conditional` for details.

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

    .. math:: \\Sigma_{X, W, y} := ((X \\ \\ W \\ \\ y)^T M_Z (X \\ \\ W \\ \\ y))^{-1} (X \\ \\ W \\ \\ y)^T P_Z (X \\ \\ W \\ \\ y)

    and

    .. math:: \\Sigma_{W, y - X \\beta} := ((W \\ \\ y - X \\beta)^T M_Z (W \\ \\ y - X \\beta))^{-1} (W \\ \\ y - X \\beta)^T P_Z (W \\ \\ y - X \\beta)

    and

    .. math:: s_\\mathrm{min}(\\beta) := \\lambda_1(\\Sigma_{X, W, y}) + \\lambda_2(\\Sigma_{X, W, y}) - \\lambda_1(\\Sigma_{W, y - X \\beta}),

    where :math:`\\lambda_1` and :math:`\\lambda_2` are the smallest and second smallest
    eigenvalues, respectively.
    Note that
    :math:`\\lambda_1(\\Sigma_{X, W, y}) = \\min_{\\beta, \\gamma} \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2 }{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2 }`
    and
    :math:`\\lambda_1(\\Sigma_{W, y - X \\beta}) = \\min_\\gamma \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2}`.

    :cite:t:`kleibergen2021efficient` conjectures and motivates that, conditionally on :math:`s_\\mathrm{min}(\\beta_0)`, the statistic
    :math:`\\mathrm{CLR(\\beta_0)}` is asymptotically bounded from above by a random
    variable that is distributed as

    .. math:: \\frac{1}{2} \\left( Q_p + Q_{q - p - r} - s_\\mathrm{min}(\\beta_0) + \\sqrt{ (Q_p + Q_{q - p - r}  - s_\\mathrm{min}(\\beta_0))^2 + 4 Q_{p} s_\\textrm{min} } \\right),

    where :math:`Q_p \\sim \\chi^2(p)` and
    :math:`Q_{q - p - r} \\sim \\chi^2(q - p - r)` are independent chi-squared random
    variables. This is robust to weak instruments. If identification is strong, that is
    :math:`s_\\mathrm{min}(\\beta_0) \\to \\infty`, the conditional likelihood ratio
    test is equivalent to the likelihood ratio test
    (see :py:func:`ivmodels.tests.likelihood_ratio_test`), with :math:`\\chi^2(p)`
    limiting distribution.
    If identification is weak, that is :math:`s_\\mathrm{min}(\\beta_0) \\to 0`, the
    conditional likelihood ratio test is equivalent to the Anderson-Rubin test
    (see :py:func:`ivmodels.tests.anderson_rubin_test`) with :math:`\\chi^2(q - r)`
    limiting distribution.
    See :cite:p:`kleibergen2021efficient` for details.

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
    tol: float, optional, default: 1e-6
        Tolerance for the approximation of the critical value function.

    Returns
    -------
    statistic: float
        The test statistic :math:`CLR(\\beta)`.
    p_value: float
        The p-value of the test, correct up to tolerance ``tol``.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.

    References
    ----------
    .. bibliography::
       :filter: False

       moreira2003conditional
       kleibergen2021efficient
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
        kclass = KClass(kappa="liml").fit(X=W, y=y - X @ beta, Z=Z)
        ar = kclass.ar_min(
            X=W, y=y - X @ beta, X_proj=W_proj, y_proj=y_proj - X_proj @ beta
        )

        statistic = (n - q) * (ar - XWy_eigenvals[0])

        # if type_ == "k":
        s_min = XWy_eigenvals[0] + XWy_eigenvals[1] - ar
        # else:
        #     XW = np.concatenate([X, W], axis=1)
        #     XW_proj = np.concatenate([X_proj, W_proj], axis=1)

        #     residuals = y - X @ beta - kclass.predict(X=W)
        #     residuals_proj = proj(Z, residuals)
        #     residuals_orth = residuals - residuals_proj

        #     Sigma = (residuals_orth.T @ XW) / (residuals_orth.T @ residuals_orth)
        #     XWt = XW - np.outer(residuals, Sigma)
        #     XWt_proj = XW_proj - np.outer(residuals_proj, Sigma)
        #     s_min = min(np.real(scipy.linalg.eigvals(np.linalg.solve((XWt - XWt_proj).T @ XWt, XWt_proj.T @ XWt))))
    p_value = _conditional_likelihood_ratio_critical_value_function(
        p, q, s_min, statistic, tol=tol
    )

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
    ``anderson_rubin_test(Z, X, y, beta=x, W=W)[1] > alpha``. It is thus a confidence
    region for the causal parameter corresponding to the endogenous regressors of
    interest ``X``.

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

       (\\beta - \\hat{\\beta})^T (X^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X) (\\beta - \\hat{\\beta}) \\leq \\hat{\\sigma}^2 F_{\\chi^2(p)}(1 - \\alpha),

    where :math:`\\hat \\beta` is an estimate of the causal parameter :math:`\\beta_0`
    (controlled by the parameter ``estimator``),
    :math:`\\hat \\sigma^2 = \\frac{1}{n} \\| y - X \\hat \\beta \\|^2_2`,
    :math:`P_Z` is the projection matrix onto the column space of :math:`Z`,
    and :math:`F_{\\chi^2(p)}` is the cumulative distribution function of the
    :math:`\\chi^2(p)` distribution.

    If ``W`` is not ``None``, the quadric is defined as

    .. math::

       (\\beta - B \\hat{\\beta})^T (B ((X W)^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X) (X W))^{-1} B^T)^{-1} (\\beta - B \\hat{\\beta}) \\leq \\hat{\\sigma}^2 F_{\\chi^2(p)}(1 - \\alpha).

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
