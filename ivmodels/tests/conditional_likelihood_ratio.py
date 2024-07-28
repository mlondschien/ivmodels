import numpy as np
import scipy

from ivmodels.confidence_set import ConfidenceSet
from ivmodels.models.kclass import KClass
from ivmodels.quadric import Quadric
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
    tol=1e-4,
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
    (see :py:func:`~ivmodels.tests.anderson_rubin_test`) with :math:`\\chi^2(k)` limiting
    distribution.
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

    :cite:t:`kleibergen2021efficient` conjectures and motivates that, conditionally on :math:`s_\\mathrm{min}(\\beta_0)`, the statistic
    :math:`\\mathrm{CLR(\\beta_0)}` is asymptotically bounded from above by a random
    variable that is distributed as

    .. math:: \\frac{1}{2} \\left( Q_{m_X} + Q_{k - m_X - m_W} - s_\\mathrm{min}(\\beta_0) + \\sqrt{ (Q_{m_X} + Q_{k - m_X - m_W}  - s_\\mathrm{min}(\\beta_0))^2 + 4 Q_{m_X} s_\\textrm{min} } \\right),

    where :math:`Q_{m_X} \\sim \\chi^2(m_X)` and
    :math:`Q_{k - m_X - m_W} \\sim \\chi^2(k - m_X - m_W)` are independent chi-squared random
    variables. This is robust to weak instruments. If identification is strong, that is
    :math:`s_\\mathrm{min}(\\beta_0) \\to \\infty`, the conditional likelihood ratio
    test is equivalent to the likelihood ratio test
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
    D: np.ndarray of dimension (n, 0) or None, optional, default = None
        Exogenous regressors of interest. Not supported for this test.
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
        return (np.nan, 1)

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W, D = oproj(C, X, y, Z, W, D)

    X_proj, y_proj, W_proj = proj(Z, X, y, W)

    residuals = y - X @ beta
    residuals_proj = y_proj - X_proj @ beta

    if mw == 0:
        residuals_orth = residuals - residuals_proj

        Sigma = (residuals_orth.T @ X) / (residuals_orth.T @ residuals_orth)
        Xt = X - np.outer(residuals, Sigma)
        Xt_proj = X_proj - np.outer(residuals_proj, Sigma)
        Xt_orth = Xt - Xt_proj

        s_min = np.real(
            _characteristic_roots(
                a=Xt_proj.T @ Xt_proj, b=Xt_orth.T @ Xt_orth, subset_by_index=[0, 0]
            )[0]
        ) * (n - k - mc - md - fit_intercept)

        Xy = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        Xy_proj = np.hstack([X_proj, y_proj.reshape(-1, 1)])

        ar_min = (
            _characteristic_roots(
                a=Xy.T @ Xy,
                b=(Xy - Xy_proj).T @ Xy,
                subset_by_index=[0, 0],
            )[0]
            - 1
        )

        ar = residuals_proj.T @ residuals_proj / (residuals_orth.T @ residuals_orth)

        statistic = (n - k - mc - md - fit_intercept) * (ar - ar_min)

    elif mw > 0:
        XWy = np.concatenate([X, W, y.reshape(-1, 1)], axis=1)
        XWy_proj = np.concatenate([X_proj, W_proj, y_proj.reshape(-1, 1)], axis=1)

        XWy_eigenvals = (
            np.sort(
                np.real(
                    _characteristic_roots(
                        a=XWy.T @ XWy,
                        b=(XWy - XWy_proj).T @ XWy,
                        subset_by_index=[0, 1],
                    )
                )
            )
            - 1
        )

        kclass = KClass(kappa="liml").fit(X=W, y=residuals, Z=Z)
        ar = kclass.ar_min(X=W, y=residuals, X_proj=W_proj, y_proj=residuals_proj)

        dof = n - k - mc - fit_intercept - md
        statistic = dof * (ar - XWy_eigenvals[0])
        s_min = dof * (XWy_eigenvals[0] + XWy_eigenvals[1] - ar)

    p_value = conditional_likelihood_ratio_critical_value_function(
        mx + md, k + md - mw, s_min, statistic, method=method, tol=tol
    )

    return statistic, p_value


def inverse_conditional_likelihood_ratio_test(
    Z, X, y, alpha=0.05, W=None, C=None, D=None, fit_intercept=True, tol=1e-4
):
    """Return an approximation of the confidence set by inversion of the CLR test."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, C, D, _ = _check_inputs(Z, X, y, W=W, C=C, D=D)

    n, k = Z.shape
    mx, mw, mc, md = X.shape[1], W.shape[1], C.shape[1], D.shape[1]

    if md > 0:
        return ConfidenceSet(boundaries=[(-np.inf, np.inf)])

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W = oproj(C, X, y, Z, W)

    S = np.concatenate([X, W], axis=1)

    S_proj, y_proj = proj(Z, S, y)
    S_orth = S - S_proj
    y_orth = y - y_proj

    Sy_proj = np.concatenate([S_proj, y_proj.reshape(-1, 1)], axis=1)
    Sy_orth = np.concatenate([S_orth, y_orth.reshape(-1, 1)], axis=1)

    Sy_eigvals = np.real(
        _characteristic_roots(
            a=Sy_proj.T @ Sy_proj, b=Sy_orth.T @ Sy_orth, subset_by_index=[0, 1]
        )
    )

    dof = n - k - mc - md - fit_intercept

    # "lower bound" on the confidence set to be computed. That is, the confidence set
    # to be computed will contain the lower bound.
    quantile_lower = scipy.stats.chi2.ppf(1 - alpha, df=mx + md) + dof * Sy_eigvals[0]

    A = S.T @ (dof * S_proj - quantile_lower * S_orth)
    b = -2 * (dof * S_proj - quantile_lower * S_orth).T @ y
    c = y.T @ (dof * y_proj - quantile_lower * y_orth)
    coordinates = np.concatenate([np.arange(mx), np.arange(mx + mw, mx + mw + md)])
    cs_lower = ConfidenceSet.from_quadric(Quadric(A, b, c).project(coordinates))

    quantile_upper = (
        scipy.stats.chi2.ppf(1 - alpha, df=k + md - mw) + dof * Sy_eigvals[0]
    )

    A = S.T @ (dof * S_proj - quantile_upper * S_orth)
    b = -2 * (dof * S_proj - quantile_upper * S_orth).T @ y
    c = y.T @ (dof * y_proj - quantile_upper * y_orth)
    cs_upper = ConfidenceSet.from_quadric(Quadric(A, b, c).project(coordinates))

    Wy_proj = Sy_proj[:, mx:]
    Wy_orth = Sy_orth[:, mx:]

    def f(x):
        Wy_proj_ = np.copy(Wy_proj)
        Wy_proj_[:, -1:] -= S_proj[:, :mx] * x
        Wy_orth_ = np.copy(Wy_orth)
        Wy_orth_[:, -1:] -= S_orth[:, :mx] * x

        eigval = np.real(
            _characteristic_roots(
                a=Wy_proj_.T @ Wy_proj_, b=Wy_orth_.T @ Wy_orth_, subset_by_index=[0, 0]
            )
        )

        s_min = dof * (Sy_eigvals[0] + Sy_eigvals[1] - eigval[0])
        statistic = dof * eigval[0] - dof * Sy_eigvals[0]
        return (
            conditional_likelihood_ratio_critical_value_function(
                mx,
                k - mx - mw,
                s_min,
                statistic,
                method="numerical_integration",
                tol=tol,
            )
            - 1
            + alpha
        )

    boundaries = []
    for left_upper, right_upper in cs_upper.boundaries:
        left_lower_, right_lower_ = right_upper, left_upper
        for left_lower, right_lower in cs_lower.boundaries:
            if left_upper <= left_lower and right_lower <= right_upper:
                left_lower_, right_lower_ = left_lower, right_lower
                break

        boundaries.append(
            (
                _find_roots(
                    f,
                    left_lower_,
                    left_upper,
                    tol=tol,
                    max_value=1e6,
                    max_eval=1000,
                ),
                _find_roots(
                    f,
                    right_lower_,
                    right_upper,
                    tol=tol,
                    max_value=1e6,
                    max_eval=1000,
                ),
            )
        )
    return ConfidenceSet(boundaries=boundaries)
