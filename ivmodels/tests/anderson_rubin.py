import numpy as np
import scipy

from ivmodels.confidence_set import ConfidenceSet
from ivmodels.models.kclass import KClass
from ivmodels.quadric import Quadric
from ivmodels.utils import _check_inputs, _find_roots, oproj, proj


def more_powerful_subvector_anderson_rubin_critical_value_function(
    z, kappa_1_hat, k, mw
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
        :math:`M:=((W \\ y - X \\beta)^T M_Z (W \\ y - X \\beta))^{-1} (W \\ y-X \\beta)^T P_Z (W \\ y-X \\beta)`.
        This is the conditioning statistic.
    k: int
        Number of instruments.
    mw: int
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
    k_prime = k - mw + 1

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


def anderson_rubin_test(
    Z, X, y, beta, W=None, C=None, D=None, critical_values="chi2", fit_intercept=True
):
    """
    Perform the Anderson Rubin test :cite:p:`anderson1949estimation`.

    Test the null hypothesis that the residuals are uncorrelated with the instruments.
    If ``W`` is ``None``, the test statistic is defined as

    .. math:: \\mathrm{AR}(\\beta) := \\frac{n - k}{k} \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| M_Z (y - X \\beta) \\|_2^2},

    where :math:`P_Z` is the projection matrix onto the column space of :math:`Z` and
    :math:`M_Z = \\mathrm{Id} - P_Z`.

    Under the null and normally distributed errors, this test statistic is distributed as
    :math:`F_{k, n - k}`, where :math:`k` is the number of instruments and :math:`n` is
    the number of observations. The statistic is asymptotically distributed as
    :math:`\\chi^2(k) / k` under the null and non-normally distributed errors, even for
    weak instruments.

    If ``W`` is not ``None``, the test statistic is

    .. math::

       \\mathrm{AR}(\\beta) &:= \\min_\\gamma \\frac{n - k}{k - m_W} \\frac{\\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\gamma) \\|_2^2} \\\\
       &= \\frac{n - k}{k - m_W} \\frac{\\| P_Z (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\hat\\gamma_\\mathrm{LIML}) \\|_2^2},

    where :math:`\\hat\\gamma_\\mathrm{LIML}` is the LIML estimate using instruments
    :math:`Z`, covariates :math:`W` and outcomes :math:`y - X \\beta`.
    Under the null, this test statistic is asymptotically bounded from above by a random
    variable that is distributed as
    :math:`\\frac{1}{k - m_W} \\chi^2(k - m_W)`, where :math:`r = \\mathrm{dim}(W)`. See
    :cite:p:`guggenberger2012asymptotic`.

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
        Exogenous regressors of interest.
    critical_values: str, optional, default = "chi2"
        If ``"chi2"``, use the :math:`\\chi^2(k - m_W)` distribution to compute the p-value.
        If ``"f"``, use the :math:`F_{k - m_W, n - k}` distribution to compute the p-value.
        If ``"guggenberger"`` or ``"GKM"``, use the critical value function proposed by
        :cite:t:`guggenberger2019more` to compute the p-value.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\mathrm{AR}(\\beta)`.
    p_value: float
        The p-value of the test.

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
    Z, X, y, W, C, D, beta = _check_inputs(Z, X, y, W=W, C=C, D=D, beta=beta)

    n, k = Z.shape
    mw, mc, md = W.shape[1], C.shape[1], D.shape[1]

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W, D = oproj(C, X, y, Z, W, D)

    if md > 0:
        X, Z = np.hstack([X, D]), np.hstack([Z, D])

    if mw == 0:
        residuals = y - X @ beta
        proj_residuals = proj(Z, residuals)
        ar = (
            np.square(proj_residuals).sum()
            / np.square(residuals - proj_residuals).sum()
        )
        dfn = k + md
    else:
        ar = KClass.ar_min(X=W, y=y - X @ beta, Z=Z)
        dfn = k - mw + md

    dfd = n - k - mc - md - fit_intercept
    statistic = ar * dfd / dfn

    if critical_values == "chi2":
        p_value = 1 - scipy.stats.chi2.cdf(statistic * dfn, df=dfn)
    elif critical_values == "f":
        p_value = 1 - scipy.stats.f.cdf(statistic, dfn=dfn, dfd=dfd)
    elif critical_values.startswith("guggenberger") or "gkm" in critical_values.lower():
        if mw == 0:
            # For mw == 0, the GKM critical values are just the chi2 critical values.
            p_value = 1 - scipy.stats.chi2.cdf(statistic * dfn, df=dfn)
        else:
            kappa_max = (n - k - mc - md) * KClass._spectrum(X=W, y=y - X @ beta, Z=Z)[
                -1
            ]

            p_value = more_powerful_subvector_anderson_rubin_critical_value_function(
                statistic * dfn, kappa_max, k=k + md, mw=mw
            )
    else:
        raise ValueError(
            "critical_values must be one of 'chi2', 'f', or 'guggenberger'. Got "
            f"{critical_values}."
        )
    return statistic, p_value


def inverse_anderson_rubin_test(
    Z,
    X,
    y,
    alpha=0.05,
    W=None,
    C=None,
    D=None,
    critical_values="chi2",
    fit_intercept=True,
    tol=1e-6,
    max_eval=1000,
):
    """
    Return the quadric for to the inverse Anderson-Rubin test's acceptance region.

    The returned quadric satisfies ``quadric(x) <= 0`` if and only if
    ``anderson_rubin_test(Z, X, y, beta=x, W=W)[1] > alpha``. It is thus a confidence
    region for the causal parameter corresponding to the endogenous regressors of
    interest ``X``.

    If ``W`` is ``None``, let :math:`q := \\frac{k}{n-k}F_{F(k, n-k}(1 - \\alpha)`, where
    :math:`F_{F(k, n-k)}` is the cumulative distribution function of the
    :math:`F(k, n-k)` distribution. The quadric is defined as

    .. math::

       \\mathrm{AR}(\\beta) = \\frac{n - k}{k} \\frac{\\| P_Z (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2} \\leq F_{F(k, n-k)}(1 - \\alpha) \\\\
       \\Leftrightarrow \\beta^T X^T (P_Z - q M_Z) X \\beta - 2 y^T (P_Z - q M_Z) X \\beta + y^T (P_Z - q M_Z) y \\leq 0.

    If ``W`` is not ``None``, let :math:`q := \\frac{k - m_W}{n-k}F_{F(k - m_W, n-k)}(1 - \\alpha)`.
    The quadric is defined as

    .. math::
        \\mathrm{AR}(\\beta) = \\min_\\gamma \\frac{n - k}{k - m_W} \\frac{\\| P_Z (y - X \\beta - W \\gamma) \\|_2^2}{\\| M_Z (y - X \\beta - W \\gamma) \\|_2^2} \\leq F_{F(k - m_W, n-k)}(1 - \\alpha).


    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    alpha: float
        Significance level.
    W: np.ndarray of dimension (n, mw) or None, optional, default = None
        Endogenous regressors not of interest.
    C: np.ndarray of dimension (n, mc) or None, optional, default = None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md) or None, optional, default = None
        Exogenous regressors of interest.
    critical_values: str, optional, default = "chi2"
        If ``"chi2"``, use the :math:`\\chi^2(k - m_W)` distribution to compute the
        p-value.
        If ``"f"``, use the :math:`F_{k - m_W, n - k}` distribution to compute the
        p-value.
        If ``"gkm"``, use the critical value function proposed by
        :cite:t:`guggenberger2019more`.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.
    tol: float, optional, default = 1e-6
        Tolerance for the root finding algorithm when critical_values is ``"gkm"``.
    max_eval: int, optional, default = 1000
        Maximum number of evaluations for the root finding algorithm when
        critical_values is ``"gkm"``.

    Returns
    -------
    Quadric
        The quadric for the acceptance region.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, C, D, _ = _check_inputs(Z, X, y, W=W, C=C, D=D)

    n, k = Z.shape
    mx, mw, mc, md = X.shape[1], W.shape[1], C.shape[1], D.shape[1]

    if critical_values.startswith("guggenberger") or "gkm" in critical_values.lower():
        # For mw == 0, the GKM critical values are just the chi2 critical values.
        if mw == 0:
            critical_values = "chi2"
        elif mx + md != 1:
            raise ValueError(
                "Test inversion for the Anderson-Rubin test with GKM critical values "
                "is only implemented for mx = X.shape[1] == 1 and md = D.shape[1] == 0."
                f"Got mx={mx}, md={md}."
            )
        else:
            critical_values = "gkm"

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W, D = oproj(C, X, y, Z, W, D)

    dfn = k + md - mw
    dfd = n - k - mc - md - fit_intercept

    if md > 0:
        Z = np.hstack([Z, D])
        X = np.hstack([X, D])

    S = np.concatenate([X, W], axis=1)

    S_proj, y_proj = proj(Z, S, y)

    S_orth = S - S_proj
    y_orth = y - y_proj

    if critical_values in ["chi2", "gkm"]:
        quantile = scipy.stats.chi2.ppf(1 - alpha, df=dfn) / dfd
    elif critical_values == "f":
        quantile = scipy.stats.f.ppf(1 - alpha, dfn=dfn, dfd=dfd) * dfn / dfd
    else:
        raise ValueError(
            "critical_values must be one of 'chi2', 'f', 'gkm', or 'guggenberger'. "
            f"Got {critical_values}."
        )

    # equal to S @ (S_proj - quantile * S_orth), but latter might result in A.T != A
    # due to numerical errors
    A = S_proj.T @ S_proj - quantile * S_orth.T @ S_orth
    b = -2 * (S_proj.T @ y_proj - quantile * S_orth.T @ y_orth)
    c = y_proj.T @ y_proj - quantile * y_orth.T @ y_orth

    if critical_values in ["chi2", "f"]:
        return Quadric(A, b, c).project(np.arange(mx + md))

    # For GKM's critical values, the "standard" inverse AR test CS with chi2 critical
    # values acts as an upper bound on the confidence set to be computed. That is, the
    # "standard" inverse AR test CS will contain the confidence set to be computed. This
    # holds as the GKM critical values are (strictly) more powerful than the chi2
    # critical values.
    upper_bound = Quadric(A, b, c).project(np.arange(mx + md))

    def f(x):
        x = np.array([x]).flatten()
        spectrum = KClass._spectrum(
            X=S[:, (mx + md) :],
            X_proj=S_proj[:, (mx + md) :],
            y=y - S[:, : (mx + md)] @ x,
            y_proj=y_proj - S_proj[:, : (mx + md)] @ x,
            subset_by_index=[0, mw],
        )

        return (
            more_powerful_subvector_anderson_rubin_critical_value_function(
                spectrum[0] * dfd, spectrum[-1] * dfd, k=k + md, mw=mw
            )
            - alpha
        )

    # The inverse AR CS is empty -> return empty.
    if upper_bound.is_empty():
        return upper_bound
    # The inverse AR CS is a bounded set.
    elif upper_bound.is_bounded():
        # The liml minimizes the AR test statistic. As the inverse AR CS is nonempty
        # it contains the liml and thus f(liml) > 0.
        liml = KClass(kappa="liml", fit_intercept=False).fit(X=S, y=y, Z=Z, C=C)

        boundary = upper_bound._boundary()
        left = _find_roots(
            f,
            np.min(boundary),
            liml.coef_[0],
            max_value=None,
            tol=tol,
            max_eval=max_eval,
        )[0]
        right = _find_roots(
            f,
            np.max(boundary),
            liml.coef_[0],
            max_value=None,
            tol=tol,
            max_eval=max_eval,
        )[0]
        return ConfidenceSet(boundaries=[(left, right)])
    # The inverse AR CS is unbounded.
    else:
        boundary = upper_bound._boundary()
        if len(boundary) == 0:
            # If the upper bound is the entire space, return the entire space.
            return upper_bound

        max_value = 1e3 * np.max(np.abs(np.diff(boundary.flatten())))
        left = _find_roots(
            f,
            min(upper_bound._boundary()),
            -np.inf,
            max_value=max_value,
            tol=tol,
            max_eval=max_eval,
        )[0]
        right = _find_roots(
            f,
            max(upper_bound._boundary()),
            np.inf,
            max_value=max_value,
            tol=tol,
            max_eval=max_eval,
        )[0]
        return ConfidenceSet(boundaries=[(-np.inf, left), (right, np.inf)])
