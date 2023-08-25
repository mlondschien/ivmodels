import numpy as np
import scipy

from ivmodels.quadric import Quadric
from ivmodels.utils import proj


def pulse_test(Z, residuals):
    """
    Test proposed by :cite:t:`jakobsen2022distributional` with null hypothesis: ``Z`` and ``residuals`` are uncorrelated.

    The test statistic is defined as

    .. math:: T := n \\frac{\\| P_Z r \\|_2^2}{\\| r \\|_2^2}.

    Under the null, :math:`T` is asymptotically distributed as :math:`\\chi^2(q)`.
    See Section 3.2 of :cite:p:`jakobsen2022distributional` for details.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    residuals: np.ndarray of dimension (n,)
        The residuals to test.

    Returns
    -------
    statistic: float
        The test statistic :math:`T`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{\\chi^2(q)}(T)`, where
        :math:`F_\\chi^2(q)` is the cumulative distribution function of the
        :math:`\\chi^2(q)` distribution.

    References
    ----------
    .. bibliography::
       :filter: False

       jakobsen2022distributional
    """
    proj_residuals = proj(Z, residuals)
    statistic = np.square(proj_residuals).sum() / np.square(residuals).sum()
    statistic *= Z.shape[0]

    p_value = 1 - scipy.stats.chi2.cdf(statistic, df=Z.shape[1])
    return statistic, p_value


def anderson_rubin_test(Z, residuals):
    """
    Perform the Anderson Rubin test :cite:p:`anderson1949estimation`.

    Test the null hypothesis that the residuals are uncorrelated with the instruments.
    The test statistic is defined as

    .. math:: AR := \\frac{n - q}{q} \\frac{\\| P_Z r \\|_2^2}{\\| r - P_Z r \\|_2^2}.

    Under the null and normally distributed errors, the test statistic is distributed as
    :math:`F_{q, n - q}``, where :math:`q` is the number of instruments and :math:`n` is
    the number of observations. The statistic is asymptotically distributed as
    :math:`\\chi^2(q) / q` under the null and non-normally distributed errors.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, q)
        Instruments.
    residuals: np.ndarray of dimension (n,)
        The residuals to test.

    Returns
    -------
    statistic: float
        The test statistic :math:`AR`.
    p_value: float
        The p-value of the test. Equal to :math:`1 - F_{F_{q, n - q}}(AR)`, where
        :math:`F_{F_{q, n - q}}` is the cumulative distribution function of the
        :math:`F_{q, n - q}` distribution.

    References
    ----------
    .. bibliography::
       :filter: False

       anderson1949estimation
    """
    if residuals.ndim != 1 and residuals.shape[1] != 1:
        raise ValueError(f"residuals must be a vector. Got shape {residuals.shape}.")

    n, q = Z.shape
    proj_residuals = proj(Z, residuals)
    statistic = (
        np.square(proj_residuals).sum() / np.square(residuals - proj_residuals).sum()
    )
    statistic *= (n - q) / q

    p_value = 1 - scipy.stats.f.cdf(statistic, dfn=q, dfd=n - q)
    return statistic, p_value


def inverse_anderson_rubin(Z, X, y, alpha=0.05):
    """Return the quadric for to the inverse Anderson-Rubin test's acceptance region."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    assert Z.shape[0] == X.shape[0] == y.shape[0]

    n, q = Z.shape

    quantile = scipy.stats.f.ppf(1 - alpha, dfn=q, dfd=n - q)

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


def asymptotic_confidence_interval(Z, X, y, beta, alpha=0.05):
    """Return the quadric for the acceptance region based on asymptotic normality."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    z_alpha = scipy.stats.chi2.ppf(1 - alpha, df=X.shape[1])

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    X_proj = proj(Z, X)

    # Avoid setting where (X @ beta).shape = (n, 1) and y.shape = (n,), resulting in
    # predictions.shape = (n, n) and residuals.shape = (n, n).
    predictions = X @ beta
    residuals = y.reshape(predictions.shape) - predictions
    hat_sigma_sq = np.mean(np.square(residuals))

    A = X.T @ X_proj
    b = -2 * A @ beta
    c = beta.T @ A @ beta - hat_sigma_sq * z_alpha
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

    W = np.linalg.solve(X.T @ X, X.T @ X_proj)
    eta_min = min(np.real(np.linalg.eigvals(W)))

    cdf = scipy.stats.f.cdf((n - q) / q * eta_min / (1 - eta_min), dfn=q, dfd=n - q)
    return 1 - cdf
