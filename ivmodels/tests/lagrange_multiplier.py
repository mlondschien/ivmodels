import numpy as np
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


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

    S = np.hstack((X, W))
    S_proj = np.hstack((X_proj, W_proj))
    Sigma = residuals_orth.T @ S / sigma_hat
    St = S - np.outer(residuals, Sigma)
    St_proj = S_proj - np.outer(residuals_proj, Sigma)

    solved = np.linalg.solve(St_proj.T @ St_proj, St_proj.T @ residuals_proj)
    residuals_proj_St = St_proj @ solved

    lm = residuals_proj_St.T @ residuals_proj_St / sigma_hat
    ar = residuals_proj.T @ residuals_proj / sigma_hat
    kappa = ar - lm

    first_term = -St_proj[:, : X.shape[1]].T @ residuals_proj
    second_term = kappa * (St - St_proj)[:, : X.shape[1]].T @ St @ solved

    d_lm = 2 * (first_term + second_term) / sigma_hat

    return (n * lm.item(), n * d_lm.flatten())


def lagrange_multiplier_test(Z, X, y, beta, W=None, fit_intercept=True):
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
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    beta: np.ndarray of dimension (mx,)
        Coefficients to test.
    W: np.ndarray of dimension (n, mw) or None, optional, default=None
        Endogenous regressors not of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

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

    if fit_intercept:
        Z = Z - Z.mean(axis=0)
        W = W - W.mean(axis=0)
        X = X - X.mean(axis=0)
        y = y - y.mean()

    n, k = Z.shape
    mx = X.shape[1]

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

        statistic *= n - k - fit_intercept

        p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx)

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

        statistic *= n - k - fit_intercept

        p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx)

    return statistic, p_value
