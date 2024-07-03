import numpy as np
import scipy

from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import oproj, proj


class _LM:
    """
    Helper class to compute the Lagrange multiplier test statistic and its derivative.

    Parameters
    ----------
    X: np.ndarray of dimension (n, mx)
        Regressors.
    Y: np.ndarray of dimension (n,) or (n, 1)
        Outcomes.
    W: np.ndarray of dimension (n, mw)
        Regressors to minimize over.
    dof: int
        Degrees of freedom for variance computation. Equal to n - k - mc.
    Z: np.ndarray of dimension (n, k), optional, default=None
        Instruments. Either ``Z`` or ``X_proj``, ``Y_proj``, and ``W_proj`` must be
        provided.
    X_proj: np.ndarray of dimension (n, mx), optional, default=None
        Projection of ``X`` onto the column space of ``Z``.
    Y_proj: np.ndarray of dimension (n,) or (n, 1), optional, default=None
        Projection of ``Y`` onto the column space of ``Z``.
    W_proj: np.ndarray of dimension (n, mw), optional, default=None
        Projection of ``W`` onto the column space of ``Z``.
    """

    def __init__(self, X, y, W, dof, Z=None, X_proj=None, y_proj=None, W_proj=None):

        self.X = X
        self.y = y.reshape(-1, 1)
        self.W = W
        if X_proj is None or y_proj is None or W_proj is None:
            if Z is None:
                raise ValueError("Z must be provided to compute the projection.")
            else:
                X_proj, y_proj, W_proj = proj(Z, self.X, self.y, self.W)

        self.X_proj = X_proj
        self.X_orth = X - X_proj
        self.y_proj = y_proj.reshape(-1, 1)
        self.y_orth = self.y - self.y_proj
        self.W_proj = W_proj
        self.W_orth = W - W_proj
        self.yS = np.hstack((self.y, X, W))
        self.yS_proj = np.hstack((self.y_proj, X_proj, W_proj))
        self.yS_orth = np.hstack((self.y_orth, self.X_orth, self.W_orth))

        self.n, self.mx = X.shape
        self.mw = W.shape[1]
        self.dof = dof

        # precomputation for Sigma, sigma_hat, and Omega_hat for liml
        self.yS_orth_at_yS = self.yS_orth.T @ self.yS_orth
        # for liml
        self.yS_proj_at_yS = self.yS_proj.T @ self.yS_proj

    def derivative(self, beta, gamma=None):
        """Return LM and derivative of LM at beta, gamma w.r.t. (beta, gamma)."""
        if gamma is not None:
            one_beta_gamma = np.hstack(([1], -beta.flatten(), -gamma.flatten()))
        else:
            one_beta_gamma = np.hstack(([1], -beta.flatten()))

        residuals = self.yS @ one_beta_gamma
        residuals_proj = self.yS_proj @ one_beta_gamma
        # residuals_orth = residuals - residuals_proj

        Sigma = one_beta_gamma.T @ self.yS_orth_at_yS
        sigma_hat = Sigma @ one_beta_gamma
        Sigma = Sigma[1:] / sigma_hat

        St = self.yS[:, 1:] - np.outer(residuals, Sigma)
        St_proj = self.yS_proj[:, 1:] - np.outer(residuals_proj, Sigma)

        solved = np.linalg.solve(St_proj.T @ St_proj, St_proj.T @ residuals_proj)
        residuals_proj_St = St_proj @ solved

        lm = residuals_proj_St.T @ residuals_proj_St / sigma_hat
        ar = residuals_proj.T @ residuals_proj / sigma_hat
        kappa = ar - lm

        first_term = -St_proj.T @ residuals_proj
        second_term = kappa * (St - St_proj).T @ St @ solved

        d_lm = 2 * (first_term + second_term) / sigma_hat

        return (self.dof * lm.item(), self.dof * d_lm.flatten())

    def lm(self, beta):
        beta = beta.reshape(-1, 1)

        one_beta_id = np.zeros((1 + self.mx + self.mw, 1 + self.mw))
        one_beta_id[0, 0] = 1
        one_beta_id[1 : (1 + self.mx), 0] = -beta[:, 0]
        one_beta_id[(1 + self.mx) :, 1:] = np.diag(np.ones(self.mw))

        gamma_liml = scipy.linalg.eigh(
            b=one_beta_id.T @ self.yS_orth_at_yS @ one_beta_id,
            a=one_beta_id.T @ self.yS_proj_at_yS @ one_beta_id,
            subset_by_index=[0, 0],
        )[1][:, 0]
        gamma_liml = -gamma_liml[1:] / gamma_liml[0]

        def _derivative(gamma):
            result = self.derivative(beta, gamma)
            return (result[0], result[1][self.mx :])

        res1 = scipy.optimize.minimize(
            _derivative,
            jac=True,
            x0=gamma_liml,
        )

        res2 = scipy.optimize.minimize(
            _derivative,
            jac=True,
            x0=np.zeros_like(gamma_liml),
        )

        return min(res1.fun, res2.fun)

    def lm_and_derivative(self, betas, gamma_0):
        assert len(betas.shape) == 2
        assert betas.shape[1] == self.mx

        lms = np.zeros(betas.shape[0])
        derivatives = np.zeros((betas.shape[0], self.mx))

        class _derivatives:
            """Helper class to recover last derivative called in the optimization."""

            def __init__(self, beta):
                self.beta = beta
                self.jac = None

            def __call__(self, gamma):
                self.derivative = self.derivative(self.beta, gamma)
                return (self.derivative[0], self.derivative[1][self.mx :])

        for idx in betas.shape[0]:
            _derivative = _derivatives(betas[idx, :])
            res = scipy.optimize.minimize(
                _derivative,
                jac=True,
                x0=gamma_0,
            )

            lms[idx] = res.fun
            derivatives[idx, :] = _derivative.derivative[: self.mx]
            gamma_0 = res.x

        return lms, derivatives


# def _LM(X, X_proj, Y, Y_proj, W, W_proj, beta):
#     """
#     Compute the Lagrange multiplier test statistic and its derivative at ``beta``.

#     Parameters
#     ----------
#     X: np.ndarray of dimension (n, mx)
#         Regressors.
#     X_proj: np.ndarray of dimension (n, mx)
#         Projection of ``X`` onto the column space of ``Z``.
#     Y: np.ndarray of dimension (n,)
#         Outcomes.
#     Y_proj: np.ndarray of dimension (n,)
#         Projection of ``Y`` onto the column space of ``Z``.
#     W: np.ndarray of dimension (n, mw)
#         Regressors.
#     W_proj: np.ndarray of dimension (n, mw)
#         Projection of ``W`` onto the column space of ``Z``.
#     beta: np.ndarray of dimension (mx,)
#         Coefficients to test.

#     Returns
#     -------
#     lm: float
#         The Lagrange multiplier test statistic.
#     d_lm: float
#         The derivative of the Lagrange multiplier test statistic at ``beta``.
#     """
#     n = X.shape[0]

#     residuals = Y - X @ beta
#     residuals_proj = Y_proj - X_proj @ beta
#     residuals_orth = residuals - residuals_proj

#     sigma_hat = residuals_orth.T @ residuals_orth

#     S = np.hstack((X, W))
#     S_proj = np.hstack((X_proj, W_proj))
#     Sigma = residuals_orth.T @ S / sigma_hat
#     St = S - np.outer(residuals, Sigma)
#     St_proj = S_proj - np.outer(residuals_proj, Sigma)

#     solved = np.linalg.solve(St_proj.T @ St_proj, St_proj.T @ residuals_proj)
#     residuals_proj_St = St_proj @ solved

#     lm = residuals_proj_St.T @ residuals_proj_St / sigma_hat
#     ar = residuals_proj.T @ residuals_proj / sigma_hat
#     kappa = ar - lm

#     first_term = -St_proj[:, : X.shape[1]].T @ residuals_proj
#     second_term = kappa * (St - St_proj)[:, : X.shape[1]].T @ St @ solved

#     d_lm = 2 * (first_term + second_term) / sigma_hat

#     return (n * lm.item(), n * d_lm.flatten())


def lagrange_multiplier_test(Z, X, y, beta, W=None, C=None, fit_intercept=True):
    """
    Perform the Lagrange multiplier test for ``beta`` by :cite:t:`kleibergen2002pivotal`.

    Test the null hypothesis that the residuals are uncorrelated with the instruments.
    If ``W`` is ``None``, let

    .. math:: \\tilde X(\\beta) := X - (y - X \\beta) \\frac{(y - X \\beta) M_Z X}{(y - X \\beta) M_Z (y - X \\beta)}.

    The test statistic is

    .. math:: \\mathrm{LM}(\\beta) := (n - k) \\frac{\\| P_{P_Z \\tilde X(\\beta)} (y - X \\beta) \\|_2^2}{\\| M_Z  (y - X \\beta) \\|_2^2}.

    If ``W`` is not ``None``, let

    .. math:: \\tilde S(\\beta, \\gamma) := (X \\ W) - (y - X \\beta - W \\gamma) \\frac{(y - X \\beta - W \\gamma) M_Z (X \\ W)}{(y - X \\beta - W \\gamma) M_Z (y - X \\beta - W \\gamma)}.

    The test statistic is

    .. math:: \\mathrm{LM}(\\beta) := (n - k) \\min_{\\gamma} \\frac{\\| P_{P_Z \\tilde S(\\beta, \\gamma)} (y - X \\beta - W \\gamma) \\|_2^2}{\\| M_Z  (y - X \\beta - W \\gamma) \\|_2^2}.

    This test statistic is asymptotically distributed as :math:`\\chi^2(m_X)` under the
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
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\mathrm{LM}(\\beta)`.
    p_value: float
        The p-value of the test. Equal to
        :math:`1 - F_{\\chi^2(m_X)}(\\mathrm{LM}(\\beta)`, where
        :math:`F_{\\chi^2(m_X)}` is the cumulative distribution function of the
        :math:`\\chi^2(m_X)` distribution.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.
    """
    Z, X, y, W, C, beta = _check_test_inputs(Z, X, y, W=W, C=C, beta=beta)

    n, k = Z.shape
    mx = X.shape[1]

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W = oproj(C, X, y, Z, W)

    if W.shape[1] > 0:
        statistic = _LM(
            X=X,
            y=y,
            W=W,
            Z=Z,
            dof=n - k - C.shape[1],
        ).lm(beta)

        p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx)

    else:
        residuals = y - X @ beta
        residuals_proj, X_proj = proj(Z, residuals, X)

        orth_residuals = residuals - residuals_proj

        sigma_hat = residuals.T @ orth_residuals
        Sigma = orth_residuals.T @ X / sigma_hat

        # X - (y - X beta) * (y - X beta)^T M_Z X / (y - X beta)^T M_Z (y - X beta)
        X_tilde_proj = X_proj - np.outer(residuals_proj, Sigma)

        X_tilde_proj_residuals = proj(X_tilde_proj, residuals)
        # (y - X beta) P_{P_Z X_tilde} (y - X beta) / (y - X_beta) M_Z (y - X beta)
        statistic = np.square(X_tilde_proj_residuals).sum() / sigma_hat

        statistic *= n - k - C.shape[1]

        p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx)

    return statistic, p_value
