import numpy as np
import scipy

from ivmodels.quadric import Quadric
from ivmodels.utils import proj


def pulse_test(Z, residuals):
    """
    Test proposed in [1]_ with H0: Z and residuals are uncorrelated.

    See [1]_ Section 3.2 for details.

    References
    ----------
    .. [1] https://arxiv.org/abs/2005.03353
    """
    proj_residuals = proj(Z, residuals)
    statistic = np.square(proj_residuals).sum() / np.square(residuals).sum()
    statistic *= Z.shape[0]
    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=Z.shape[1])
    return statistic, p_value


def anderson_rubin_test(Z, residuals):
    """
    Perform the Anderson Rubin test.

    Test the null hypothesis that the residuals are uncorrelated with the instruments.
    Under the null, the test statistic is distributed as `F_{q, n - q}`, where `q` is
    the number of instruments and `n` is the number of observations. Requires normally
    distributed errors for exactness.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d).
        Instruments.
    residuals: np.ndarray of dimension (n,).
        The residuals to test.

    Returns
    -------
    statistic: float
        The test statistic.
    p_value: float
        The p-value of the test.

    References
    ----------
    .. [1] Anderson, T.W. and Rubin, H. (1949), Estimation of the parameters of a single
           equation in a complete system of stochastic equations, Annals of Mathematical
           Statistics, 20, 46-63.
    """
    if residuals.ndim != 1 and residuals.shape[1] != 1:
        raise ValueError(f"residuals must be a vector. Got shape {residuals.shape}.")

    n, q = Z.shape
    proj_residuals = proj(Z, residuals)
    statistic = (
        np.square(proj_residuals).sum() / np.square(residuals - proj_residuals).sum()
    )
    statistic *= (n - q) / q

    p_value = 1 - scipy.stats.f.cdf(statistic, dfn=n - q, dfd=q)

    return statistic, p_value


def inverse_anderson_rubin(Z, X, y, alpha=0.05):
    """Return the quadric for to the inverse Anderson-Rubin test's acceptance region."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    assert Z.shape[0] == X.shape[0] == y.shape[0]

    n, q = Z.shape

    quantile = scipy.stats.f.ppf(1 - alpha, dfn=n - q, dfd=q)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    X_proj = proj(Z, X)
    X_orth = X - X_proj
    y_proj = proj(Z, y)
    y_orth = y - y_proj

    A = X.T @ (X_proj - q / (n - q) * quantile * X_orth)
    b = -2 * (X_proj - q / (n - q) * quantile * X_orth).T @ y
    c = y.T @ (y_proj - q / (n - q) * quantile * y_orth)

    if isinstance(c, np.ndarray):
        c = c.item()

    return Quadric(A, b, c)


def asymptotic_confidence_interval(Z, X, y, beta, alpha=0.95):
    """Return the quadric for the acceptance region based on asymptotic normality."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    z_alpha = scipy.stats.chi2.ppf(alpha, df=X.shape[1])

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    X_proj = proj(Z, X)

    hat_sigma_sq = np.mean(np.square(y - X @ beta))
    A = X.T @ X_proj
    b = -2 * A @ beta
    c = beta.T @ A @ beta - hat_sigma_sq * z_alpha
    return Quadric(A, b, c)


def bounded_inverse_anderson_rubin(Z, X):
    """
    Return the largest p-value `p` such that the inverse-AR confidence set is unbounded.

    In practice, the confidence set might be unbounded for `1.001 * p` only.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q).
        Instruments.
    X: np.ndarray of dimension (n, p).
        Regressors.
    """
    n, q = Z.shape

    X = X - X.mean(axis=0)

    X_proj = proj(Z, X)

    W = np.linalg.solve(X.T @ X, X.T @ X_proj)
    eta_min = min(np.real(np.linalg.eigvals(W)))

    cdf = scipy.stats.f.cdf((n - q) / q * eta_min / (1 - eta_min), dfn=n - q, dfd=q)
    return 1 - cdf
