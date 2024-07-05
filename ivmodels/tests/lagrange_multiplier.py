import numpy as np
import scipy
import scipy.interpolate
import scipy.optimize
from scipy.optimize._optimize import MemoizeJac

from ivmodels.confidence_set import ConfidenceSet
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.tests.wald import inverse_wald_test
from ivmodels.utils import oproj, proj


# https://stackoverflow.com/a/68608349/10586763
class MemoizeJacHess(MemoizeJac):
    """Cache the return values of a function returning (fun, grad, hess)."""

    def __init__(self, fun):
        super().__init__(fun)
        self.hess = None

    def _compute_if_needed(self, x, *args):
        if (
            not np.all(x == self.x)
            or self._value is None
            or self.jac is None
            or self.hess is None
        ):
            self.x = np.asarray(x).copy()
            self._value, self.jac, self.hess = self.fun(x, *args)

    def hessian(self, x, *args):  # noqa D
        self._compute_if_needed(x, *args)
        return self.hess


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

    def liml(self, beta=None):
        """
        Efficiently compute the LIML.

        This is the LIML using outcomes y (or y - X beta if beta is not none),
        covariates S (or W), and instruments Z.
        """
        if beta is not None:
            beta = beta.reshape(-1, 1)

            one_beta_id = np.zeros((1 + self.mx + self.mw, 1 + self.mw))
            one_beta_id[0, 0] = 1
            one_beta_id[1 : (1 + self.mx), 0] = -beta[:, 0]
            one_beta_id[(1 + self.mx) :, 1:] = np.diag(np.ones(self.mw))

            X_proj_X = one_beta_id.T @ self.yS_proj_at_yS @ one_beta_id
            X_orth_X = one_beta_id.T @ self.yS_orth_at_yS @ one_beta_id
        else:
            X_proj_X = self.yS_proj_at_yS
            X_orth_X = self.yS_orth_at_yS

        eigval = scipy.linalg.eigh(
            a=X_proj_X,
            b=X_orth_X,
            subset_by_index=[0, 0],
        )[1]

        return -eigval[1:, 0] / eigval[0, 0]

    def derivative(self, beta, gamma=None, jac_and_hess=True):
        """Return LM and derivative of LM at beta, gamma w.r.t. (beta, gamma)."""
        if gamma is not None:
            one_beta_gamma = np.hstack(([1], -beta.flatten(), -gamma.flatten()))
        else:
            one_beta_gamma = np.hstack(([1], -beta.flatten()))

        residuals_proj = self.yS_proj @ one_beta_gamma

        Sigma = one_beta_gamma.T @ self.yS_orth_at_yS
        sigma_hat = Sigma @ one_beta_gamma
        Sigma = Sigma[1:] / sigma_hat

        St_proj = self.yS_proj[:, 1:] - np.outer(residuals_proj, Sigma)

        if not jac_and_hess:
            residuals_proj_St = proj(St_proj, residuals_proj)
            return self.dof * residuals_proj_St.T @ residuals_proj_St / sigma_hat

        residuals = self.yS @ one_beta_gamma
        St = self.yS[:, 1:] - np.outer(residuals, Sigma)
        St_orth = St - St_proj

        mat = St_proj.T @ St_proj
        cond = np.linalg.cond(mat)
        if cond > 1e12:
            mat += 1e-6 * np.eye(mat.shape[0])

        solved = np.linalg.solve(
            mat,
            np.hstack(
                [
                    St_proj.T @ residuals_proj.reshape(-1, 1),
                    St_orth.T @ St[:, self.mx :],
                ]
            ),
        )
        residuals_proj_St = St_proj @ solved[:, 0]

        ar = residuals_proj.T @ residuals_proj / sigma_hat
        lm = residuals_proj_St.T @ residuals_proj_St / sigma_hat
        kappa = ar - lm

        first_term = -St_proj[:, self.mx :].T @ residuals_proj
        second_term = St_orth[:, self.mx :].T @ St @ solved[:, 0]
        S = self.yS[:, 1:]
        S_proj = self.yS_proj[:, 1:]
        S_orth = S - S_proj

        d_lm = 2 * (first_term + kappa * second_term) / sigma_hat

        dd_lm = (
            2
            * (
                -3 * kappa * np.outer(second_term, second_term) / sigma_hat
                + kappa**2 * St_orth[:, self.mx :].T @ St_orth @ solved[:, 1:]
                - kappa * St_orth[:, self.mx :].T @ St_orth[:, self.mx :]
                - kappa
                * St_orth[:, self.mx :].T
                @ St_orth
                @ np.outer(solved[:, 0], Sigma[self.mx :])
                + St[:, self.mx :].T
                @ (S_proj[:, self.mx :] - ar * S_orth[:, self.mx :])
                - np.outer(
                    Sigma[self.mx :],
                    (St_proj - kappa * St_orth)[:, self.mx :].T @ St @ solved[:, 0],
                )
                + 2
                * kappa
                * np.outer(S_orth[:, self.mx :].T @ St @ solved[:, 0], Sigma[self.mx :])
                - 2 * np.outer(St_proj[:, self.mx :].T @ residuals, Sigma[self.mx :])
            )
            / sigma_hat
        )

        return (self.dof * lm.item(), self.dof * d_lm.flatten(), self.dof * dd_lm)

    def lm(self, beta):
        """
        Compute the Lagrange multiplier test statistic at ``beta``.

        Computed by minimization over ``gamma``.
        """
        if isinstance(beta, float):
            beta = np.array([[beta]])

        if self.mw == 0:
            return self.derivative(beta, jac_and_hess=False)

        gamma_0 = self.liml(beta=beta)

        def _derivative(gamma):
            result = self.derivative(beta, gamma, jac_and_hess=True)
            return (result[0], result[1], result[2])

        objective = MemoizeJacHess(_derivative)
        jac = objective.derivative
        hess = objective.hessian

        res1 = scipy.optimize.minimize(
            objective, jac=jac, hess=hess, x0=gamma_0, method="newton-cg"
        )

        res2 = scipy.optimize.minimize(
            objective,
            jac=jac,
            hess=hess,
            method="newton-cg",
            x0=np.zeros_like(gamma_0),
        )

        return np.min([res1.fun, res2.fun])


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
        statistic = _LM(X=X, y=y, W=W, Z=Z, dof=n - k - C.shape[1]).lm(beta)

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


def inverse_lagrange_multiplier_test(
    Z, X, y, alpha=0.05, W=None, C=None, fit_intercept=True
):
    """
    Return an approximation of the confidence set by inversion of the LM test.

    This is only implemented if ``X.shape[1] == 1``. The confidence set is essentially
    computed by a grid search plus a root finding algorithm to improve the precision.
    Due to the numerical nature, this is in no way guaranteed to return the true
    confidence set, or contain it.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, C, _ = _check_test_inputs(Z, X, y, W=W, C=C)

    n, mx = X.shape
    if not mx == 1:
        raise ValueError("mx must be 1.")

    k = Z.shape[1]

    if fit_intercept:
        C_ = np.hstack([np.ones((n, 1)), C])
    else:
        C_ = C

    if C_.shape[1] > 0:
        X_, y_, Z_, W_ = oproj(C_, X, y, Z, W)
    else:
        X_, y_, Z_, W_ = X, y, Z, W

    dof = n - k - C_.shape[1]
    lm = _LM(X=X_, W=W_, y=y_, Z=Z_, dof=dof)
    critical_value = scipy.stats.chi2(df=mx).ppf(1 - alpha)

    # Use a test with closed-form solution to get an idea of the "scale".
    approx = inverse_wald_test(
        Z=Z, X=X, y=y, alpha=alpha, W=W, C=C, fit_intercept=fit_intercept
    )

    left, right = approx.left, approx.right
    step = right - left

    while lm.lm(right) < critical_value:
        right += step
        step *= 2

        if right > 1e6:
            return ConfidenceSet(
                left=-np.inf, right=np.inf, convex=True, message="no bounds found"
            )

    right += 4 * step

    while lm.lm(left) < critical_value:
        left -= step
        step *= 2

        if left < -1e6:
            return ConfidenceSet(
                left=-np.inf, right=np.inf, convex=True, message="no bounds found"
            )

    left -= 4 * step

    n_points = 200
    x_ = np.linspace(left, right, n_points)
    y__ = np.zeros(n_points)

    for idx in range(n_points):
        y__[idx] = lm.lm(x_[idx])

    where = np.where(y__ < critical_value)[0]
    left_bracket = x_[where[0] - 1], x_[where[where > where[0] + 1][0]]
    right_bracket = x_[where[-1] + 1], x_[where[where < where[-1] - 1][-1]]

    def f(x):
        arr = np.ones(1)
        arr[0] = x
        return lm.lm(arr) - critical_value

    new_left = scipy.optimize.root_scalar(
        f, bracket=left_bracket, maxiter=1000, xtol=1e-5
    ).root

    new_right = scipy.optimize.root_scalar(
        f,
        bracket=right_bracket,
        maxiter=1000,
        xtol=1e-5,
    ).root

    if not max(np.abs([f(new_left), f(new_right)])) < 1e-3:
        return ConfidenceSet(
            left=new_left,
            right=new_right,
            convex=True,
            empty=False,
            message="No roots found.",
        )
    else:
        return ConfidenceSet(left=new_left, right=new_right, convex=True, empty=False)
