import numpy as np
import scipy

from anchor_regression.quadric import Quadric


def proj(Z, f):
    """Project f onto the subspace spanned by Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z).
        The Z matrix.
    f: np.ndarray of dimension (n, d_f) or (n,).
        The vector to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of f onto the subspace spanned by Z. Same dimension as f.
    """
    Z = Z - Z.mean(axis=0)
    # f = f - f.mean(axis=0)

    return np.dot(Z, np.linalg.lstsq(Z, f, rcond=None)[0])


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

    A = X.T @ ((n - q) / q * X_proj - quantile * X_orth)
    b = -2 * y.T @ ((n - q) / q * X_proj - quantile * X_orth)
    c = y @ ((n - q) / q * y_proj - quantile * y_orth)

    return Quadric(A, b, c)


def asymptotic_confidence_interval(Z, X, y, beta, alpha=0.05):
    """Return the quadric for the acceptance region based on asymptotic normality."""
    z_alpha = scipy.stats.norm.ppf(1 - alpha)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    X_proj = proj(Z, X)

    hat_sigma_sq = np.mean(np.power((y - X @ beta), 2))
    A = hat_sigma_sq * np.linalg.inv(X_proj.T @ X_proj)
    b = -2 * A @ beta
    c = beta.T @ A @ beta - z_alpha
    return Quadric(A, b, c)
