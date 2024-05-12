import numpy as np
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.quadric import Quadric
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


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
        :math:`M:=((X \\ y - X \\beta)^T M_Z (X \\ y - X \\beta))^{-1} (X \\ y-X \\beta)^T P_Z (X \\ y-X \\beta)`.
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
    Z, X, y, beta, W=None, critical_values="chi2", fit_intercept=True
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
    beta: np.ndarray of dimension (mx,)
        Coefficients to test.
    W: np.ndarray of dimension (n, mw) or None, optional, default = None
        Endogenous regressors not of interest.
    critical_values: str, optional, default = "chi2"
        If ``"chi2"``, use the :math:`\\chi^2(k - m_W)` distribution to compute the p-value.
        If ``"f"``, use the :math:`F_{k - m_W, n - k}` distribution to compute the p-value.
        If ``"guggenberger"``, use the critical value function proposed by
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
    Z, X, y, W, beta = _check_test_inputs(Z, X, y, W=W, beta=beta)
    n, k = Z.shape

    if fit_intercept:
        Z = Z - Z.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y - y.mean()
        W = W - W.mean(axis=0)

    if W.shape[1] == 0:
        residuals = y - X @ beta
        proj_residuals = proj(Z, residuals)
        ar = (
            np.square(proj_residuals).sum()
            / np.square(residuals - proj_residuals).sum()
        )
        dfn = k
    else:
        spectrum = KClass._spectrum(X=W, y=y - X @ beta, Z=Z)
        ar = np.min(spectrum)
        dfn = k - W.shape[1]

    statistic = ar * (n - k - fit_intercept) / dfn

    if critical_values == "chi2":
        p_value = 1 - scipy.stats.chi2.cdf(statistic * dfn, df=dfn)
    elif critical_values == "f":
        p_value = 1 - scipy.stats.f.cdf(statistic, dfn=dfn, dfd=n - k - fit_intercept)
    elif critical_values.startswith("guggenberger"):
        if W.shape[1] == 0:
            raise ValueError(
                "The critical value function proposed by Guggenberger et al. (2019) is "
                "only available for the subvector variant where W is not None."
            )

        kappa_max = (n - k - fit_intercept) * np.max(spectrum)
        p_value = more_powerful_subvector_anderson_rubin_critical_value_function(
            statistic * dfn, kappa_max, k, mw=W.shape[1]
        )
    else:
        raise ValueError(
            "critical_values must be one of 'chi2', 'f', or 'guggenberger'. Got "
            f"{critical_values}."
        )

    return statistic, p_value


def inverse_anderson_rubin_test(
    Z, X, y, alpha=0.05, W=None, critical_values="chi2", fit_intercept=True
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
    critical_values: str, optional, default = "chi2"
        If ``"chi2"``, use the :math:`\\chi^2(k - m_W)` distribution to compute the
        p-value.
        If ``"f"``, use the :math:`F_{k - m_W, n - k}` distribution to compute the
        p-value.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.

    Returns
    -------
    Quadric
        The quadric for the acceptance region.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, _ = _check_test_inputs(Z, X, y, W=W)

    n, k = Z.shape

    X = np.concatenate([X, W], axis=1)
    dfn = k - W.shape[1]

    if fit_intercept:
        Z = Z - Z.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y - y.mean()

    if critical_values == "chi2":
        quantile = scipy.stats.chi2.ppf(1 - alpha, df=dfn) / (n - k - fit_intercept)
    elif critical_values == "f":
        quantile = (
            scipy.stats.f.ppf(1 - alpha, dfn=dfn, dfd=n - k - fit_intercept)
            * dfn
            / (n - k - fit_intercept)
        )
    else:
        raise ValueError(
            "critical_values must be one of 'chi2', 'f'. Got " f"{critical_values}."
        )

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
