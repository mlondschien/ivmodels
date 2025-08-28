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
    p, q, s_min, z, method="numerical_integration", tol=1e-6
):
    """
    Approximate the critical value function of the conditional likelihood ratio test.

    Let

    .. math: \\Gamma(q-p, p, s_\\mathrm{min}) := 1/2 \\left( Q_{q-p} + Q_p - s_\\mathrm{min} + \\sqrt{ (Q_{q-p} + Q_p - s_\\mathrm{min})^2 + 4 Q_{p} s_\\mathrm{min} } \\right),

    where :math:`Q_p \\sim \\chi^2(p)` and :math:`Q_{q-p} \\sim \\chi^2(q - p)` are
    independent chi-squared random variables. This function approximates

    .. math: Q_{q, p}(z) := \\mathbb{P}[ \\Gamma(q-p, p, s_\\mathrm{min}) > z ]

    up to tolerance ``tol``.

    If ``method`` is ``"numerical_integration"``, numerically integrates the formulation

    .. math: Q_{q, p}(z) = \\mathbb{E}_{B \\sim \\mathrm{Beta}((k - m)/2, m/2)}[ F_{\\chi^2(k)}(z / (1 - a B)) ],

    where :math:`F_{\\chi^2(k)}` is the cumulative distribution function of a
    :math:`\\chi^2(k)` distribution, and :math:`a = s_{\\min} / (z + s_{\\min})`. This
    is Equation (27) of :cite:p:`hillier2009conditional` or Equation (40) of
    :cite:p:`hillier2009exact`.

    If ``method`` is ``"power_series"``, truncates the formulation

    .. math: Q_{k, p} = (1 - a)^{p / 2} \\sum_{j = 0}^\\infty a^j \\frac{(p / 2)_j}{j!} \\F_{\\chi^2(k + 2 j)}(z + s_{\\min}),

    where :math:`(x)_j` is the Pochhammer symbol, defined as
    :math:`(x)_j = x (x + 1) ... (x + j - 1)`, :math:`\\F_k` is the cumulative
    distribution function of the :math:`\\chi^2(k)` distribution, and
    :math:`a = s_{\\min} / (z + s_{\\min})`. This is Equation (28) of
    :cite:p:`hillier2009conditional` or Equation (41) of :cite:p:`hillier2009exact`.
    The truncation is done such that the error is bounded by a tolerance ``tol``.

    Uses numerical integration by default.

    Parameters
    ----------
    p: int
        Degrees of freedom of the first chi-squared random variable.
    q: int
        Total degrees of freedom.
    s_min: float
        Identification measure.
    z: float
        Test statistic.
    method: str, optional, default: "numerical_integration"
        Method to approximate the critical value function. Must be
        ``"numerical_integration"`` or ``"power_series"``.
    tol: float, optional, default: 1e-6
        Tolerance for the approximation of the cdf of the critical value function and
        thus the p-value.

    References
    ----------
    .. bibliography::
       :filter: False

       hillier2009conditional
       hillier2009exact
    """
    if z <= 0:
        return 1

    if s_min <= 0:
        return 1 - scipy.stats.chi2(q).cdf(z)

    if q < p:
        raise ValueError("q must be greater than or equal to p.")
    if p == q:
        return 1 - scipy.stats.chi2(q).cdf(z)

    if method in ["numerical_integration"]:
        alpha = (q - p) / 2.0
        beta = p / 2.0
        a = s_min / (z + s_min)

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

    elif method == "power_series":
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
            if j > 10000:
                raise RuntimeError("Failed to converge.")

        p_value *= (1 - a) ** (p / 2)

        return 1 - p_value

    else:
        raise ValueError(
            "method argument should be 'numerical_integration' or 'power_series'. "
            f"Got {method}."
        )


@njit
def _newton_minimal_root(q_sum, q, lambdas, atol, num_iter):
    """
    Find the minimal root of the polynomial using Newton's method.

    Instead of computing the polynomials minimal root, we compute that of the secular
    equation

    f(mu) = (mu - q_sum) - sum(d_i * u_i / (d_i - μ))

    with derivative

    f'(mu) = 1 + sum(d_i * u_i / (d_i - μ)^2).

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

    mu = lambdas[0] / 2  # initial guess

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

        if np.abs(mu - mu_new) < atol:
            break

        mu = mu_new

    return mu


@njit
def conditional_likelihood_ratio_critical_value_function_monte_carlo(
    mx: int,
    md: int,
    k: int,
    lambdas: np.ndarray,
    z: float,
    atol=1e-8,
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

    for i in prange(num_samples):
        np.random.seed(i)
        qx = np.random.standard_normal(mx) ** 2
        qd = np.sum(np.random.standard_normal(md) ** 2)
        q0 = np.sum(np.random.standard_normal(k - mx) ** 2)
        q_sum = np.sum(qx) + q0

        mu_min = _newton_minimal_root(q_sum, qx, lambdas, atol=atol, num_iter=num_iter)
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
    method="numerical_integration",
    tol=1e-6,
):
    """
    Perform the conditional likelihood ratio test for ``beta``.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       \\mathrm{CLR}(\\beta) &:= (n - k) \\frac{ \\| P_Z (y - X \\beta) \\|_2^2}{ \\| M_Z (y - X \\beta) \\|_2^2} - (n - k) \\frac{ \\| P_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 } \\\\
       &= k \\ \\mathrm{AR}(\\beta) - k \\ \\min_\\beta \\mathrm{AR}(\\beta),

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z`,
    :math:`M_Z = \\mathrm{Id} - P_Z`, and :math:`\\hat\\beta_\\mathrm{LIML}` is the LIML
    estimator of :math:`\\beta` (see :py:class:`~ivmodels.KClass`), minimizing the
    Anderson-Rubin test statistic :math:`\\mathrm{AR}(\\beta)`
    (see :py:func:`~ivmodels.tests.anderson_rubin_test`) at

    .. math:: \\mathrm{AR}(\\hat\\beta_\\mathrm{LIML}) = \\frac{n - k}{k} \\lambda_\\mathrm{min}( (X \\ y)^T M_Z (X \\ y))^{-1} (X \\ y)^T P_Z (X \\ y) ).

    Let

    .. math:: \\tilde X(\\beta) := X - (y - X \\beta) \\cdot \\frac{(y - X \\beta)^T M_Z X}{(y - X \\beta)^T M_Z (y - X \\beta)}

    and

    .. math:: s_\\mathrm{min}(\\beta) := (n - k) \\cdot \\lambda_\\mathrm{min}((\\tilde X(\\beta)^T M_Z \\tilde X(\\beta))^{-1} \\tilde X(\\beta)^T P_Z \\tilde X(\\beta)).

    Then, conditionally on :math:`s_\\mathrm{min}(\\beta_0)`, the statistic
    :math:`\\mathrm{CLR(\\beta_0)}` is asymptotically bounded from above by a random
    variable that is distributed as

    .. math:: \\frac{1}{2} \\left( Q_{m_X} + Q_{k - m_X} - s_\\mathrm{min} + \\sqrt{ (Q_{m_X} + Q_{k - m_X}  - s_\\mathrm{min})^2 + 4 Q_{m_X} s_\\textrm{min} } \\right),

    where :math:`Q_{m_X} \\sim \\chi^2(m_X)` and
    :math:`Q_{k - m_X} \\sim \\chi^2(k - m_X)` are independent chi-squared random
    variables. This is robust to weak instruments. If identification is strong, that is
    :math:`s_\\mathrm{min}(\\beta_0) \\to \\infty`, the conditional likelihood ratio
    test is equivalent to the likelihood ratio test
    (see :py:func:`~ivmodels.tests.likelihood_ratio_test`), with :math:`\\chi^2(m_X)`
    limiting distribution.
    If identification is weak, that is :math:`s_\\mathrm{min}(\\beta_0) \\to 0`, the
    conditional likelihood ratio test is equivalent to the Anderson-Rubin test
    (see :py:func:`~ivmodels.tests.anderson_rubin_test`) with :math:`\\chi^2(k)`
    limiting distribution.
    See :cite:p:`moreira2003conditional` for details.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::
       \\mathrm{CLR(\\beta)} &:= (n - k) \\min_\\gamma \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2} - (n - k) \\min_{\\beta, \\gamma} \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2 }{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2 } \\\\
       &= (n - k) \\frac{ \\| P_Z (y - X \\beta - W \\hat\\gamma_\\textrm{liml}) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\hat\\gamma_\\textrm{liml}) \\|_2^2} - (n - k) \\frac{ \\| P_Z (y - (X \\ W) \\hat\\delta_\\mathrm{liml}) \\|_2^2 }{ \\| M_Z (y - (X \\ W) \\hat\\delta_\\mathrm{liml}) \\|_2^2 },

    where :math:`\\hat\\gamma_\\mathrm{LIML}` is the LIML estimator of :math:`\\gamma`
    (see :py:class:`~ivmodels.KClass`) using instruments :math:`Z`, endogenous
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

    :cite:t:`kleibergen2021efficient` conjectures and motivates that, conditionally on
    :math:`s_\\mathrm{min}(\\beta_0)`, the statistic :math:`\\mathrm{CLR(\\beta_0)}` is
    asymptotically bounded from above by a random variable that is distributed as

    .. math:: \\frac{1}{2} \\left( Q_{m_X} + Q_{k - m_X - m_W} - s_\\mathrm{min}(\\beta_0) + \\sqrt{ (Q_{m_X} + Q_{k - m_X - m_W}  - s_\\mathrm{min}(\\beta_0))^2 + 4 Q_{m_X} s_\\textrm{min} } \\right),

    where :math:`Q_{m_X} \\sim \\chi^2(m_X)` and
    :math:`Q_{k - m_X - m_W} \\sim \\chi^2(k - m_X - m_W)` are independent chi-squared
    random variables. This is robust to weak instruments. If identification is strong,
    that is :math:`s_\\mathrm{min}(\\beta_0) \\to \\infty`, the conditional likelihood
    ratio test is equivalent to the likelihood ratio test
    (see :py:func:`~ivmodels.tests.likelihood_ratio_test`), with :math:`\\chi^2(m_X)`
    limiting distribution.
    If identification is weak, that is :math:`s_\\mathrm{min}(\\beta_0) \\to 0`, the
    conditional likelihood ratio test is equivalent to the Anderson-Rubin test
    (see :py:func:`~ivmodels.tests.anderson_rubin_test`) with :math:`\\chi^2(k - m_W)`
    limiting distribution.
    See :cite:p:`kleibergen2021efficient` for details.

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
    W: np.ndarray of dimension (n, mw) or None, optional, default = None
        Endogenous regressors not of interest.
    C: np.ndarray of dimension (n, mc) or None, optional, default = None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md) or None, optional, default = None
        Exogenous regressors of interest. Will be included into both ``X`` and ``Z`` if
        supplied.
    fit_intercept: bool, optional, default: True
        Whether to include an intercept. This is equivalent to centering the inputs.
    method: str, optional, default: "numerical_integration"
        Method to approximate the critical value function. Must be
        ``"numerical_integration"`` or ``"power_series"``. See
        :py:func:`~conditional_likelihood_ratio_critical_value_function`.
    tol: float, optional, default: 1e-6
        Tolerance for the approximation of the cdf of the critical value function and
        thus the p-value.

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

        if mx + md == 1:
            return statistic, conditional_likelihood_ratio_critical_value_function(
                1, k + 1, lambdas[0], z=statistic
            )
        else:
            return (
                statistic,
                conditional_likelihood_ratio_critical_value_function_monte_carlo(
                    mx, md, k, lambdas, z=statistic
                ),
            )

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
        mx + md, k + md - mw, s_min, statistic, method=method, tol=tol
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
                mx + md,
                k + md - mw,
                s_min,
                statistic,
                method="numerical_integration",
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
