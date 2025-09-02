import numpy as np
import scipy
from numba import njit, prange

from ivmodels.confidence_set import ConfidenceSet
from ivmodels.models.kclass import KClass
from ivmodels.quadric import Quadric
from ivmodels.tests.anderson_rubin import inverse_anderson_rubin_test
from ivmodels.utils import (
    _characteristic_roots,
    _check_inputs,
    _find_roots,
    oproj,
    proj,
)


def conditional_likelihood_ratio_critical_value_function(
    k,
    mx,
    md,
    lambdas,
    z,
    critical_values="londschien2025exact",
    tol=1e-6,
    num_samples=100_000,
):
    """
    Approximate the critical value function of the conditional likelihood ratio test.

    If ``critical_values`` is ``"londschien2025exact"``, computes the exact distribution
    from Theorem 1 of :cite:p:`londschien2025exact` using Monte Carlo simulation:

    .. math::

        \\mathrm{CLR}(\\beta_0) \\overset{d}{\\to} \\sum_{i=0}^m q_i - \\mu_\\mathrm{min},

    where :math:`q_0 \\sim \\chi^2(k-m)`, :math:`q_1, \\ldots, q_m \\sim \\chi^2(1)` are
    independent, and :math:`\\mu_\\mathrm{min}` is the smallest root of the polynomial

    .. math::

        p(\\mu) := \\left(\\mu - \\sum_{i=0}^m q_i\\right) \\prod_{i=1}^m (\\mu - \\lambda_i) - \\sum_{i=1}^m \\lambda_i q_i \\prod_{j \\geq 1, j \\neq i} (\\mu - \\lambda_j).

    This is computed by Monte Carlo simulation, generating samples of the :math:`q_i` and
    solving for :math:`\\mu_\\mathrm{min}` using Newton's method.

    If ``critical_values`` is ``"kleibergen2007generalizing"`` or ``"moreira2003conditional"``,
    uses the upper bound from Corollary 2 of :cite:p:`londschien2025exact`:

    .. math::

        \\mathrm{CLR}(\\beta_0) \\leq \\Gamma(k-m, m, \\lambda_1),

    where

    .. math::

        \\Gamma(k-m, m, \\lambda_1) := \\frac{1}{2} \\left( Q_{k-m} + Q_m - \\lambda_1 + \\sqrt{ (Q_{k-m} + Q_m + \\lambda_1)^2 - 4 Q_{k-m} \\lambda_1 } \\right),

    with :math:`Q_{k-m} \\sim \\chi^2(k-m)`, :math:`Q_m \\sim \\chi^2(m)` independent, and
    :math:`\\lambda_1` the smallest eigenvalue. This is computed by numerical integration
    using the formulation

    .. math::

        \\mathbb{P}[\\Gamma(k-m, m, \\lambda_1) > z] = \\mathbb{E}_{B \\sim \\mathrm{Beta}((k-m)/2, m/2)}[F_{\\chi^2(k)}(z/(1-aB))],

    where :math:`F_{\\chi^2(k)}` is the CDF of :math:`\\chi^2(k)` and :math:`a = \\lambda_1/(z + \\lambda_1)`.

    Parameters
    ----------
    k: int
        Number of instruments.
    mx: int
        Number of endogenous variables.
    md: int
        Number of included exogenous variables.
    lambdas: array_like
        Eigenvalues of the concentration matrix.
    z: float
        Test statistic.
    critical_values: {"londschien2025exact", "kleibergen2007generalizing", "moreira2003conditional"}, default="londschien2025exact"
        Which critical values to use. If ``"londschien2025exact"``, uses the exact
        distribution conditional on all eigenvalues via Monte Carlo simulation.
        If ``"kleibergen2007generalizing"`` or ``"moreira2003conditional"``, uses
        the upper bound conditional on the smallest eigenvalue via numerical integration.
    tol: float, default=1e-6
        Tolerance for the approximation of the CDF and thus the p-value.
    num_samples: int, default=100_000
        Number of Monte Carlo samples when using ``"londschien2025exact"``.

    Returns
    -------
    float
        The p-value :math:`\\mathbb{P}[\\mathrm{CLR} > z]`.

    References
    ----------
    .. bibliography::
       :filter: False

       hillier2009conditional
       hillier2009exact
       londschien2025exact
    """
    if z <= 0:
        return 1

    if k < mx:
        raise ValueError("k must be greater than or equal to mx.")
    if k == mx:
        return 1 - scipy.stats.chi2(k + md).cdf(z)

    lambdas = np.sort(lambdas)

    if critical_values == "londschien2025exact" and lambdas[-1] <= 0:
        return 1 - scipy.stats.chi2(k + md).cdf(z)
    elif critical_values != "londschien2025exact" and lambdas[0] <= 0:
        return 1 - scipy.stats.chi2(k + md).cdf(z)

    if critical_values == "londschien2025exact" and mx + md > 1:
        return _clr_critical_value_function_monte_carlo(
            mx=mx, md=md, k=k, lambdas=lambdas, z=z, tol=tol, num_samples=num_samples
        )

    else:
        p = mx + md
        q = k + md
        alpha = (q - p) / 2.0
        beta = p / 2.0
        a = lambdas[0] / (z + lambdas[0])

        # We wish to integrate
        # beta = beta(alpha, beta); chi2 = chi2(q)
        # int_0^1 chi2.cdf(z / (1 - a * x)) * beta.pdf(x, (q - p) / 2, p / 2) dx
        #
        # use
        #  beta(alpha, beta).pdf = 1/B(alpha, beta) * x^{alpha - 1} * (1 - x)^{beta - 1}
        # and
        #  chi2(q).cdf(x) = 1/Gamma(k/2) * gamma(k/2, x/2)
        #                 = scipy.special.gammainc(k/2, x/2).
        # substitute y <- 1 - a * x and use the QUADPACK routine qawse
        # (see scipy.special.gammainc)
        k = q / 2.0
        z_over_2 = z / 2.0

        const = np.power(a, -alpha - beta + 1) / scipy.special.beta(alpha, beta)

        def integrand(y):
            return const * scipy.special.gammainc(k, z_over_2 / y)

        res = scipy.integrate.quad(
            integrand,
            1 - a,
            1,
            weight="alg",
            wvar=(beta - 1, alpha - 1),
            epsabs=tol,
        )
        return 1 - res[0]


@njit
def _newton_minimal_root(q_sum, q, lambdas, tol, num_iter):
    """
    Find the minimal root of the polynomial using Newton's method.

    Instead of computing the polynomials minimal root, we compute that of the secular
    equation

    f(mu) = (mu - q_sum) - sum(d_i * u_i / (d_i - μ))

    with derivative

    f'(mu) = 1 + sum(d_i * u_i / (d_i - μ)^2).

    This has exactly one root in the interval (0, lambdas[0]), which we approximate
    using Newton's method.

    Parameters
    ----------
    q_sum: float
        The sum of the chi^2(1) and chi^2(k - m) distributions.
    q: array of floats
        The values taken by the individual chi^2(1) distributions.
    lambdas: array of floats
        Conditioning statistic.
    atol: float
        Absolute tolerance for convergence.
    num_iter: int
        Maximum number of iterations.
    """
    m = len(lambdas)

    # Initial guess. Exact if lambdas are all equal.
    mu = lambdas[0] - q_sum
    mu = mu - np.sqrt(mu**2 + 4 * lambdas[0] * np.sum(q))
    mu /= 2

    for _ in range(num_iter):
        f = mu - q_sum
        df = 1
        for i in range(m):
            diff = mu - lambdas[i]
            term = q[i] * lambdas[i] / diff
            f -= term
            df += term / diff

        mu_new = mu - f / df

        if mu_new < 0:
            mu_new = mu / 2

        if mu_new > lambdas[0]:
            mu_new = (mu + lambdas[0]) / 2

        if np.abs(mu - mu_new) < tol:
            break

        mu = mu_new

    return mu


@njit
def _clr_critical_value_function_monte_carlo(
    mx: int,
    md: int,
    k: int,
    lambdas: np.ndarray,
    z: float,
    tol=1e-6,
    num_iter=100,
    num_samples=10_000,
):
    """
    Compute the CLR's exact critical value function using Monte Carlo simulation.

    If ``mx == 1`` or all entries of ``d`` are equal, this is the same as the numerical
    integration method in
    ``conditional_likelihood_ratio_critical_value_function``.

    Parameters
    ----------
    mx: int
        The number of endogenous variables.
    md: int
        The number of included exogenous variables.
    k: int
        The number of instruments.
    lambdas: np.ndarray
        The conditioning statistic.
    z: float
        The test statistic.
    atol: float
        The absolute tolerance for convergence.
    num_iter: int
        The maximum number of iterations.
    num_samples: int
        The number of Monte Carlo samples.

    """
    count = 0

    np.random.seed(0)

    qx_samples = np.random.standard_normal((num_samples, mx)) ** 2
    qd_samples = np.sum(np.random.standard_normal((num_samples, md)) ** 2, axis=1)
    q0_samples = np.sum(np.random.standard_normal((num_samples, k - mx)) ** 2, axis=1)

    for i in prange(num_samples):
        qx = qx_samples[i, :]
        qd = qd_samples[i]
        q0 = q0_samples[i]
        q_sum = np.sum(qx) + q0

        mu_min = _newton_minimal_root(q_sum, qx, lambdas, tol=tol, num_iter=num_iter)
        if qd + q_sum - mu_min > z:
            count += 1

    return count / num_samples


def conditional_likelihood_ratio_test(
    Z,
    X,
    y,
    beta,
    W=None,
    C=None,
    D=None,
    fit_intercept=True,
    critical_values="londschien2025exact",
    tol=1e-6,
    num_samples=10_000,
):
    """
    Perform the conditional likelihood ratio test for ``beta``.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       \\mathrm{CLR}(\\beta) &:= (n - k) \\frac{ \\| P_Z (y - X \\beta) \\|_2^2}{ \\| M_Z (y - X \\beta) \\|_2^2} - (n - k) \\frac{ \\| P_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 } \\\\
       &= k \\cdot \\mathrm{AR}(\\beta) - k \\cdot \\min_\\beta \\mathrm{AR}(\\beta),

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z`,
    :math:`M_Z = \\mathrm{Id} - P_Z`, and :math:`\\hat\\beta_\\mathrm{LIML}` is the LIML
    estimator of :math:`\\beta` (see :py:class:`~ivmodels.KClass`), minimizing the
    Anderson-Rubin test statistic :math:`\\mathrm{AR}(\\beta)`
    (see :py:func:`~ivmodels.tests.anderson_rubin_test`) at

    .. math::

        \\mathrm{AR}(\\hat\\beta_\\mathrm{LIML}) = \\frac{n - k}{k} \\lambda_\\mathrm{min}\\left( \\left(\\begin{pmatrix} X & y \\end{pmatrix}^T M_Z \\begin{pmatrix} X & y \\end{pmatrix}\\right)^{-1} \\begin{pmatrix} X & y \\end{pmatrix}^T P_Z \\begin{pmatrix} X & y \\end{pmatrix} \\right).

    Let

    .. math::

        \\tilde X(\\beta) := X - (y - X \\beta) \\cdot \\frac{(y - X \\beta)^T M_Z X}{(y - X \\beta)^T M_Z (y - X \\beta)}

    and let :math:`\\lambda_1, \\ldots, \\lambda_m` be the eigenvalues of

    .. math::

        (n - k) \\cdot \\left[\\tilde X(\\beta)^T M_Z \\tilde X(\\beta)\\right]^{-1} \\tilde X(\\beta)^T P_Z \\tilde X(\\beta).

    If ``critical_values="londschien2025exact"``, the exact asymptotic distribution from
    Theorem 1 of :cite:p:`londschien2025exact` is used:

    .. math::

        \\mathrm{CLR}(\\beta_0) \\overset{d}{\\to} \\sum_{i=0}^m q_i - \\mu_\\mathrm{min},

    where :math:`q_0 \\sim \\chi^2(k-m)`, :math:`q_1, \\ldots, q_m \\sim \\chi^2(1)`, and
    :math:`\\mu_\\mathrm{min}` is the smallest root of the polynomial

    .. math::

        p(\\mu) := \\left(\\mu - \\sum_{i=0}^m q_i\\right) \\prod_{i=1}^m (\\mu - \\lambda_i) - \\sum_{i=1}^m \\lambda_i q_i \\prod_{j \\geq 1, j \\neq i} (\\mu - \\lambda_j).

    This distribution is conditional on all eigenvalues :math:`\\lambda_1, \\ldots, \\lambda_m`
    and provides substantially more power when eigenvalues differ.

    If ``critical_values`` is ``"kleibergen2007generalizing"`` or ``"moreira2003conditional"``,
    uses the upper bound conditional on only the smallest eigenvalue :math:`\\lambda_1`:

    .. math::

        \\mathrm{CLR}(\\beta_0) \\leq \\frac{1}{2} \\left( Q_{m_X} + Q_{k - m_X} - \\lambda_1 + \\sqrt{ (Q_{m_X} + Q_{k - m_X}  + \\lambda_1)^2 - 4 Q_{k - m_X} \\lambda_1 } \\right),

    where :math:`Q_{m_X} \\sim \\chi^2(m_X)` and :math:`Q_{k - m_X} \\sim \\chi^2(k - m_X)`
    are independent. This bound is sharp when all eigenvalues are equal.

    This test is robust to weak instruments. If identification is strong
    (:math:`\\lambda_i \\to \\infty`), the test is equivalent to the likelihood ratio test
    with :math:`\\chi^2(m_X)` distribution. If identification is weak
    (:math:`\\lambda_i \\to 0`), the test is equivalent to the Anderson-Rubin test
    with :math:`\\chi^2(k)` distribution.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::

       \\mathrm{CLR}(\\beta) &:= (n - k) \\min_\\gamma \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2} - (n - k) \\min_{\\beta, \\gamma} \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2 }{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2 } \\\\
       &= (n - k) \\frac{ \\| P_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2} - (n - k) \\frac{ \\| P_Z (y - \\begin{pmatrix} X & W \\end{pmatrix} \\hat\\delta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - \\begin{pmatrix} X & W \\end{pmatrix} \\hat\\delta_\\mathrm{LIML}) \\|_2^2 },

    where :math:`\\hat\\gamma_\\mathrm{LIML}` and :math:`\\hat\\delta_\\mathrm{LIML}` are
    LIML estimators. In this case, only the upper bound method is available
    (``critical_values`` must be ``"kleibergen2007generalizing"`` or ``"moreira2003conditional"``).

    Parameters
    ----------
    Z: array_like of shape (n, k)
        Instruments.
    X: array_like of shape (n, mx)
        Endogenous regressors of interest.
    y: array_like of shape (n,)
        Outcomes.
    beta: array_like of shape (mx + md,)
        Coefficients to test.
    W: array_like of shape (n, mw) or None, default=None
        Endogenous regressors not of interest.
    C: array_like of shape (n, mc) or None, default=None
        Exogenous regressors not of interest.
    D: array_like of shape (n, md) or None, default=None
        Exogenous regressors of interest. Will be included into both ``X`` and ``Z`` if
        supplied.
    fit_intercept: bool, default=True
        Whether to include an intercept. This is equivalent to centering the inputs.
    critical_values: {"londschien2025exact", "kleibergen2007generalizing", "moreira2003conditional"}, default="londschien2025exact"
        Which critical values to use. If ``"londschien2025exact"``, uses the exact
        distribution conditional on all eigenvalues via Monte Carlo simulation of the
        polynomial root :math:`\\mu_\\mathrm{min}`. Only available when ``W`` is ``None``.
        If ``"kleibergen2007generalizing"`` or ``"moreira2003conditional"``, uses
        the upper bound conditional on the smallest eigenvalue via numerical integration.
    tol: float, default=1e-6
        Tolerance for the approximation of the CDF and thus the p-value.
    num_samples: int, default=10000
        Number of Monte Carlo samples when using ``"londschien2025exact"``.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\mathrm{CLR}(\\beta)`.
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
       kleibergen2007generalizing
       londschien2025exact
    """
    Z, X, y, W, C, D, beta = _check_inputs(Z, X, y, W=W, C=C, D=D, beta=beta)

    n, k = Z.shape
    mx, mw, mc, md = X.shape[1], W.shape[1], C.shape[1], D.shape[1]

    if md > 0:
        X = np.hstack([X, D])
        Z = np.hstack([Z, D])

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W = oproj(C, X, y, Z, W)

    X_proj, y_proj, W_proj = proj(Z, X, y, W)

    residuals = y - X @ beta
    residuals_proj = y_proj - X_proj @ beta

    if mw == 0:
        residuals_orth = residuals - residuals_proj

        Sigma = (residuals_orth.T @ X) / (residuals_orth.T @ residuals_orth)
        Xt = X - np.outer(residuals, Sigma)
        Xt_proj = X_proj - np.outer(residuals_proj, Sigma)
        Xt_orth = Xt - Xt_proj

        Xy = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        Xy_proj = np.hstack([X_proj, y_proj.reshape(-1, 1)])

        lambdas = np.sort(
            np.real(_characteristic_roots(a=Xt_proj.T @ Xt_proj, b=Xt_orth.T @ Xt_orth))
        ) * (n - k - mc - md - fit_intercept)

        Xy_orth = Xy - Xy_proj

        ar_min = (
            _characteristic_roots(
                a=Xy_proj.T @ Xy_proj, b=Xy_orth.T @ Xy_orth, subset_by_index=[0, 0]
            )
        )[0]
        ar = residuals_proj.T @ residuals_proj / (residuals_orth.T @ residuals_orth)
        statistic = (n - k - mc - md - fit_intercept) * (ar - ar_min)

        p_value = conditional_likelihood_ratio_critical_value_function(
            k=k,
            mx=mx,
            md=md,
            lambdas=lambdas,
            z=statistic,
            critical_values=critical_values,
            tol=tol,
            num_samples=num_samples,
        )
        return statistic, p_value
    elif mw > 0:
        XWy = np.concatenate([X, W, y.reshape(-1, 1)], axis=1)
        XWy_proj = np.concatenate([X_proj, W_proj, y_proj.reshape(-1, 1)], axis=1)

        XWy_eigenvals = np.sort(
            np.real(
                _characteristic_roots(
                    a=XWy_proj.T @ XWy_proj,
                    b=(XWy - XWy_proj).T @ (XWy - XWy_proj),
                    subset_by_index=[0, 1],
                )
            )
        )

        kclass = KClass(kappa="liml")
        ar = kclass.ar_min(X=W, y=residuals, X_proj=W_proj, y_proj=residuals_proj)

        dof = n - k - mc - fit_intercept - md
        statistic = dof * (ar - XWy_eigenvals[0])
        s_min = dof * (XWy_eigenvals[0] + XWy_eigenvals[1] - ar)

    p_value = conditional_likelihood_ratio_critical_value_function(
        k=k - mw,
        mx=mx,
        md=md,
        lambdas=np.ones(mx + md) * s_min,
        z=statistic,
        critical_values="moreira2003conditional",
        tol=tol,
    )

    return statistic, p_value


def inverse_conditional_likelihood_ratio_test(
    Z,
    X,
    y,
    alpha=0.05,
    W=None,
    C=None,
    D=None,
    fit_intercept=True,
    tol=1e-6,
    max_value=1e6,
    max_eval=1000,
):
    """
    Return an approximation of the confidence set by inversion of the CLR test.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    alpha: float, optional, default = 0.05
        Significance level of the test.
    W: np.ndarray of dimension (n, mw) or None, optional, default = None
        Endogenous regressors not of interest.
    C: np.ndarray of dimension (n, mc) or None, optional, default = None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, 0) or None, optional, default = None
        Exogenous regressors of interest. Not supported for this test.
    fit_intercept: bool, optional, default: True
        Whether to include an intercept. This is equivalent to centering the inputs.
    tol: float, optional, default: 1e-4
        The boundaries of the confidence set are computed up to this tolerance.
    max_value: float, optional, default: 1e6
        The maximum value to consider when searching for the boundaries of the
        confidence set. That is, if the true confidence set is of the form
        [0, max_value + 1], the confidence returned set will be [0, np.inf].
    max_eval: int, optional, default: 1000
        The maximum number of evaluations of the critical value function to find the
        boundaries of the confidence set.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, C, D, _ = _check_inputs(Z, X, y, W=W, C=C, D=D)

    n, k = Z.shape
    mx, mw, mc, md = X.shape[1], W.shape[1], C.shape[1], D.shape[1]

    if k == mx + mw:
        return inverse_anderson_rubin_test(
            Z=Z, X=X, W=W, y=y, C=C, D=D, fit_intercept=fit_intercept, alpha=alpha
        )

    if md > 0:
        X = np.hstack([X, D])
        Z = np.hstack([Z, D])

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W = oproj(C, X, y, Z, W)

    XWy = np.concatenate([X, W, y.reshape(-1, 1)], axis=1)
    XWy_proj = proj(Z, XWy)
    XWy_orth = XWy - XWy_proj

    XWy_p_XWy = XWy_proj.T @ XWy_proj
    XWy_o_XWy = XWy_orth.T @ XWy_orth

    XWy_eigvals = np.sort(
        np.real(_characteristic_roots(a=XWy_p_XWy, b=XWy_o_XWy, subset_by_index=[0, 1]))
    )

    dof = n - k - mc - md - fit_intercept

    # "lower bound" on the confidence set to be computed. That is, the confidence set
    # to be computed will contain the lower bound.
    quantile_lower = scipy.stats.chi2.ppf(1 - alpha, df=mx + md) + dof * XWy_eigvals[0]

    R = XWy_p_XWy - quantile_lower / dof * XWy_o_XWy
    A = R[: (mx + md + mw), : (mx + md + mw)]
    b = -2 * R[: (mx + md + mw), (mx + md + mw)]
    c = R[(mx + md + mw), (mx + md + mw)]
    cs_lower = ConfidenceSet.from_quadric(Quadric(A, b, c).project(np.arange(mx + md)))

    # "upper bound" on the confidence set to be computed. That is, the confidence set
    # to be computed will be contained in the upper bound.
    quantile_upper = (
        scipy.stats.chi2.ppf(1 - alpha, df=k + md - mw) + dof * XWy_eigvals[0]
    )

    R = XWy_p_XWy - quantile_upper / dof * XWy_o_XWy
    A = R[: (mx + md + mw), : (mx + md + mw)]
    b = -2 * R[: (mx + md + mw), (mx + md + mw)]
    c = R[(mx + md + mw), (mx + md + mw)]
    cs_upper = ConfidenceSet.from_quadric(Quadric(A, b, c).project(np.arange(mx + md)))

    def f(x):
        a = XWy_p_XWy.copy()
        a[:, -1:] -= a[:, : (mx + md)] * x
        a[-1:, :] -= a[: (mx + md), :] * x

        b = XWy_o_XWy.copy()
        b[:, -1:] -= b[:, : (mx + md)] * x
        b[-1:, :] -= b[: (mx + md), :] * x

        ar_min = _characteristic_roots(
            a=a[(mx + md) :, (mx + md) :],
            b=b[(mx + md) :, (mx + md) :],
            subset_by_index=[0, 0],
        )[0]

        s_min = dof * (XWy_eigvals[0] + XWy_eigvals[1] - ar_min)
        statistic = dof * (ar_min - XWy_eigvals[0])

        return alpha - (
            conditional_likelihood_ratio_critical_value_function(
                k=k - mw,
                mx=mx,
                md=md,
                lambdas=np.ones(mx + md) * s_min,
                z=statistic,
                critical_values="moreira2003conditional",
                tol=tol,
            )
        )

    roots = []

    for left_upper, right_upper in cs_upper.boundaries:
        left_lower_, right_lower_ = None, None
        for left_lower, right_lower in cs_lower.boundaries:
            if left_upper <= left_lower and right_lower <= right_upper:
                left_lower_, right_lower_ = left_lower, right_lower
                break

        if left_lower_ is None:
            bounds = [max(left_upper, -max_value), min(right_upper, max_value)]
            res = scipy.optimize.minimize_scalar(f, bounds=bounds)
            if f(res.x) < 0:
                left_lower_ = res.x
                right_lower_ = res.x
            else:
                continue

        roots += _find_roots(
            f,
            left_lower_,
            left_upper,
            tol=tol,
            max_value=max_value,
            max_eval=max_eval,
            max_depth=5,
        )
        roots += _find_roots(
            f,
            right_lower_,
            right_upper,
            tol=tol,
            max_value=max_value,
            max_eval=max_eval,
            max_depth=5,
        )

    roots = sorted(roots)

    assert len(roots) % 2 == 0
    boundaries = [(left, right) for left, right in zip(roots[::2], roots[1::2])]
    return ConfidenceSet(boundaries=boundaries)
