import numpy as np
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


def _conditional_likelihood_ratio_critical_value_function(
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
        beta = scipy.stats.beta((q - p) / 2, p / 2)
        chi2 = scipy.stats.chi2(q)
        a = s_min / (z + s_min)

        def integrand(b):
            return beta.pdf(b) * chi2.cdf(z / (1 - a * b))

        res = scipy.integrate.quad(
            integrand,
            0,
            1,
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
            if j > 1000:
                raise RuntimeError("Failed to converge.")

        p_value *= (1 - a) ** (p / 2)

        return 1 - p_value

    else:
        raise ValueError(
            "method argument should be 'numerical_integration' or 'power_series'. "
            f"Got {method}."
        )


def conditional_likelihood_ratio_test(
    Z, X, y, beta, W=None, method="numerical_integration", tol=1e-6
):
    """
    Perform the conditional likelihood ratio test for ``beta``.

    If ``W`` is ``None``, the test statistic is defined as

    .. math::

       \\mathrm{CLR(\\beta)} &:= (n - q) \\frac{ \\| P_Z (y - X \\beta) \\|_2^2}{ \\| M_Z (y - X \\beta) \\|_2^2} - (n - q) \\frac{ \\| P_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 }{ \\| M_Z (y - X \\hat\\beta_\\mathrm{LIML}) \\|_2^2 } \\\\
       &= q \\ \\mathrm{AR}(\\beta) - q \\ \\min_\\beta \\mathrm{AR}(\\beta),

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z`,
    :math:`M_Z = \\mathrm{Id} - P_Z`, and :math:`\\hat\\beta_\\mathrm{LIML}` is the LIML
    estimator of :math:`\\beta` (see :py:class:`~ivmodels.KClass`), minimizing the
    Anderson-Rubin test statistic :math:`\\mathrm{AR}(\\beta)`
    (see :py:func:`~ivmodels.tests.anderson_rubin_test`) at

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
    (see :py:func:`~ivmodels.tests.likelihood_ratio_test`), with :math:`\\chi^2(p)`
    limiting distribution.
    If identification is weak, that is :math:`s_\\mathrm{min}(\\beta_0) \\to 0`, the
    conditional likelihood ratio test is equivalent to the Anderson-Rubin test
    (see :py:func:`~ivmodels.tests.anderson_rubin_test`) with :math:`\\chi^2(q)` limiting
    distribution.
    See :cite:p:`moreira2003conditional` for details.

    If ``W`` is not ``None``, the test statistic is defined as

    .. math::
       \\mathrm{CLR(\\beta)} &:= (n - q) \\min_\\gamma \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2} - (n - q) \\min_{\\beta, \\gamma} \\frac{ \\| P_Z (y - X \\beta - W \\gamma) \\|_2^2 }{ \\| M_Z (y - X \\beta - W \\gamma) \\|_2^2 } \\\\
       &= (n - q) \\frac{ \\| P_Z (y - X \\beta - W \\hat\\gamma_\\textrm{liml}) \\|_2^2}{ \\| M_Z (y - X \\beta - W \\hat\\gamma_\\textrm{liml}) \\|_2^2} - (n - q) \\frac{ \\| P_Z (y - (X \\ W) \\hat\\delta_\\mathrm{liml}) \\|_2^2 }{ \\| M_Z (y - (X \\ W) \\hat\\delta_\\mathrm{liml}) \\|_2^2 },

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

    .. math:: \\frac{1}{2} \\left( Q_p + Q_{q - p - r} - s_\\mathrm{min}(\\beta_0) + \\sqrt{ (Q_p + Q_{q - p - r}  - s_\\mathrm{min}(\\beta_0))^2 + 4 Q_{p} s_\\textrm{min} } \\right),

    where :math:`Q_p \\sim \\chi^2(p)` and
    :math:`Q_{q - p - r} \\sim \\chi^2(q - p - r)` are independent chi-squared random
    variables. This is robust to weak instruments. If identification is strong, that is
    :math:`s_\\mathrm{min}(\\beta_0) \\to \\infty`, the conditional likelihood ratio
    test is equivalent to the likelihood ratio test
    (see :py:func:`~ivmodels.tests.likelihood_ratio_test`), with :math:`\\chi^2(p)`
    limiting distribution.
    If identification is weak, that is :math:`s_\\mathrm{min}(\\beta_0) \\to 0`, the
    conditional likelihood ratio test is equivalent to the Anderson-Rubin test
    (see :py:func:`~ivmodels.tests.anderson_rubin_test`) with :math:`\\chi^2(q - r)`
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
    method: str, optional, default: "numerical_integration"
        Method to approximate the critical value function. Must be
        ``"numerical_integration"`` or ``"power_series"``. See
        :py:func:`_conditional_likelihood_ratio_critical_value_function`.
    tol: float, optional, default: 1e-6
        Tolerance for the approximation of the cdf of the critical value function and
        thus the p-value.

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
