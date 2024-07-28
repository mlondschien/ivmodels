import numpy as np
import scipy
import scipy.interpolate
import scipy.optimize
from scipy.optimize._optimize import MemoizeJac

from ivmodels.confidence_set import ConfidenceSet
from ivmodels.models.kclass import KClass
from ivmodels.utils import (
    _characteristic_roots,
    _check_inputs,
    _find_roots,
    oproj,
    proj,
)


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
    optimizer: str, optional, default="bfgs"
        Optimization method to use. Passed to ``scipy.optimize.minimize``.
    gamma_0: list of str or np.ndarray of dimension (mw), optional, default=None
        Initial value for the minimization. If ``str``, must be one of "liml" or "zero".
        If ``None``, ``"liml"`` is used.
    """

    def __init__(
        self,
        X,
        y,
        W,
        dof,
        Z=None,
        X_proj=None,
        y_proj=None,
        W_proj=None,
        optimizer="bfgs",
        gamma_0=None,
    ):

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

        self.optimizer = optimizer
        self.gamma_0 = ["liml"] if gamma_0 is None else gamma_0
        if isinstance(self.gamma_0, str):
            self.gamma_0 = [self.gamma_0]

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

            resX_proj_resX = one_beta_id.T @ self.yS_proj_at_yS @ one_beta_id
            resX_orth_resX = one_beta_id.T @ self.yS_orth_at_yS @ one_beta_id
        else:
            resX_proj_resX = self.yS_proj_at_yS
            resX_orth_resX = self.yS_orth_at_yS

        cond = np.linalg.cond(resX_orth_resX)
        if cond < 0.5 / np.finfo(resX_orth_resX.dtype).eps:
            eigval = scipy.linalg.eigh(
                a=resX_proj_resX,
                b=resX_orth_resX,
                subset_by_index=[0, 0],
            )[1]

            liml = -eigval[1:, 0] / eigval[0, 0]
        else:
            kappa_liml = _characteristic_roots(
                resX_proj_resX, resX_orth_resX, subset_by_index=[0, 0]
            )[0]
            liml = np.linalg.solve(
                resX_proj_resX[1:, 1:] - kappa_liml * resX_orth_resX[1:, 1:],
                resX_proj_resX[1:, 0] - kappa_liml * resX_orth_resX[1:, 0],
            )

        return liml

    def derivative(self, beta, gamma=None, jac=True, hess=True):
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

        if not jac:  # not jac -> not hess
            residuals_proj_St = proj(St_proj, residuals_proj)

            return (
                self.dof * residuals_proj_St.T @ residuals_proj_St / sigma_hat,
                None,
                None,
            )

        residuals = self.yS @ one_beta_gamma
        St = self.yS[:, 1:] - np.outer(residuals, Sigma)
        St_orth = St - St_proj

        mat = St_proj.T @ St_proj
        cond = np.linalg.cond(mat)

        if hess:
            f = np.hstack(
                [
                    St_proj.T @ residuals_proj.reshape(-1, 1),
                    St_orth.T @ St[:, self.mx :],
                ]
            )
            if cond > 1e8:
                solved = np.linalg.pinv(mat) @ f
            else:
                solved = np.linalg.solve(mat, f)

        else:
            # If mat is well conditioned, both should be equivalent, but the pinv
            # solution is defined even if mat is singular. In theory, solve should be
            # faster. In practice, not so clear. The lstsq solution tends to be slower.
            if cond > 1e8:
                solved = np.linalg.pinv(mat) @ St_proj.T @ residuals_proj.reshape(-1, 1)
            else:
                # solved = scipy.linalg.lstsq(St_proj, residuals_proj.reshape(-1, 1), cond=None, lapack_driver="gelsy")[0]
                # solved = np.linalg.pinv(mat) @ St_proj.T @ residuals_proj.reshape(-1, 1)
                solved = np.linalg.solve(mat, St_proj.T @ residuals_proj.reshape(-1, 1))

        residuals_proj_St = St_proj @ solved[:, 0]

        ar = residuals_proj.T @ residuals_proj / sigma_hat
        lm = residuals_proj_St.T @ residuals_proj_St / sigma_hat
        kappa = ar - lm

        first_term = -St_proj[:, self.mx :].T @ residuals_proj
        second_term = St_orth[:, self.mx :].T @ St @ solved[:, 0]

        d_lm = 2 * (first_term + kappa * second_term) / sigma_hat

        if not hess:
            return (self.dof * lm, self.dof * d_lm, None)

        S = self.yS[:, 1:]
        S_proj = self.yS_proj[:, 1:]
        S_orth = S - S_proj
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

    def lm(self, beta, return_minimizer=False):
        """
        Compute the Lagrange multiplier test statistic at ``beta``.

        Computed by minimization over ``gamma``.
        """
        if isinstance(beta, float):
            beta = np.array([[beta]])

        if self.mw == 0:
            return self.derivative(beta, jac=False, hess=False)[0]

        methods_with_hessian = [
            "newton-cg",
            "dogleg",
            "trust-ncg",
            "trust-krylov",
            "trust-exact",
        ]
        hess = self.optimizer.lower() in methods_with_hessian

        def _derivative(gamma):
            result = self.derivative(beta, gamma, jac=True, hess=hess)
            return result

        objective = MemoizeJacHess(_derivative)
        jac = objective.derivative
        hess = (
            objective.hessian
            if self.optimizer.lower() in methods_with_hessian
            else None
        )

        results = []
        for g in self.gamma_0:
            if g == "liml":
                gamma_0 = self.liml(beta=beta)
            elif g == "zero":
                gamma_0 = np.zeros(self.mw)
            else:
                raise ValueError(f"unknown gamma_0: {g}")

            results.append(
                scipy.optimize.minimize(
                    objective, jac=jac, hess=hess, x0=gamma_0, method=self.optimizer
                )
            )

        minimizer = min(results, key=lambda r: r.fun).x
        statistic = self.derivative(beta, minimizer, jac=False, hess=False)[0]

        if return_minimizer:
            return statistic, minimizer
        return statistic


def lagrange_multiplier_test(
    Z, X, y, beta, W=None, C=None, D=None, fit_intercept=True, **kwargs
):
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
    beta: np.ndarray of dimension (mx + md,)
        Coefficients to test.
    W: np.ndarray of dimension (n, mw) or None, optional, default=None
        Endogenous regressors not of interest.
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md) or None, optional, default=None
        Exogenous regressors of interest.
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
    Z, X, y, W, C, D, beta = _check_inputs(Z, X, y, W=W, C=C, D=D, beta=beta)

    n, k = Z.shape
    mx, mw, mc, md = X.shape[1], W.shape[1], C.shape[1], D.shape[1]

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W, D = oproj(C, X, y, Z, W, D)

    if md > 0:
        X = np.hstack([X, D])
        Z = np.hstack([Z, D])

    dof = n - k - mc - md - fit_intercept
    if mw > 0:
        lm = _LM(X=X, y=y, W=W, Z=Z, dof=dof, **kwargs)
        statistic = lm.lm(beta)

        p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx + md)

    else:
        residuals = y - X @ beta
        residuals_proj, X_proj = proj(Z, residuals, X)

        orth_residuals = residuals - residuals_proj

        sigma_hat = orth_residuals.T @ orth_residuals
        Sigma = orth_residuals.T @ X / sigma_hat

        # X - (y - X beta) * (y - X beta)^T M_Z X / (y - X beta)^T M_Z (y - X beta)
        X_tilde_proj = X_proj - np.outer(residuals_proj, Sigma)

        X_tilde_proj_residuals = proj(X_tilde_proj, residuals)
        # (y - X beta) P_{P_Z X_tilde} (y - X beta) / (y - X_beta) M_Z (y - X beta)
        statistic = dof * np.square(X_tilde_proj_residuals).sum() / sigma_hat

        p_value = 1 - scipy.stats.chi2.cdf(statistic, df=mx + md)

    return statistic, p_value


def inverse_lagrange_multiplier_test(
    Z,
    X,
    y,
    alpha=0.05,
    W=None,
    C=None,
    D=None,
    fit_intercept=True,
    tol=1e-4,
    max_value=1e8,
    max_eval=1000,
):
    """
    Return an approximation of the confidence set by inversion of the LM test.

    This is only implemented if `mx + md = 1`. The confidence set is
    computed by a root finding algorithm, see the docs of
    :func:`~ivmodels.tests.utils._find_roots` for more details.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors of interest.
    y: np.ndarray of dimension (n,)
        Outcomes.
    alpha: float, optional, default=0.05
        Significance level. The confidence level is ``1 - alpha``.
    W: np.ndarray of dimension (n, mw) or None, optional, default=None
        Endogenous regressors not of interest.
    C: np.ndarray of dimension (n, mc) or None, optional, default=None
        Exogenous regressors not of interest.
    D: np.ndarray of dimension (n, md) or None, optional, default=None
        Exogenous regressors of interest.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. This is equivalent to centering the inputs.
    tol: float, optional, default=1e-4
        Tolerance for the root finding algorithm.
    max_value: float, optional, default=1e8
        Maximum value for the root finding algorithm. Returns a confidence set with
        infinite bounds if the algorithm reaches this value.
    max_eval: int, optional, default=1000
        Maximum number of evaluations of the statistic for the root finding algorithm.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    Z, X, y, W, C, D, _ = _check_inputs(Z, X, y, W=W, C=C, D=D)

    n, k = Z.shape
    mx, mc, md = X.shape[1], C.shape[1], D.shape[1]

    if not mx + md == 1:
        raise ValueError("mx + md must be 1.")

    if fit_intercept:
        C = np.hstack([np.ones((n, 1)), C])

    if C.shape[1] > 0:
        X, y, Z, W, D = oproj(C, X, y, Z, W, D)

    dof = n - k - mc - md - fit_intercept

    lm = _LM(X=np.hstack([X, D]), W=W, y=y, Z=np.hstack([Z, D]), dof=dof)
    critical_value = scipy.stats.chi2(df=mx + md).ppf(1 - alpha)

    if md == 0:
        liml = lm.liml()[0]
    else:
        liml = KClass(kappa="liml", fit_intercept=False).fit(W, y, Z=Z, C=D).coef_[-1]

    left = _find_roots(
        lambda x: lm.lm(x) - critical_value,
        a=liml,
        b=-np.inf,
        tol=tol,
        max_value=max_value,
        max_eval=max_eval,
    )
    right = _find_roots(
        lambda x: lm.lm(x) - critical_value,
        a=liml,
        b=np.inf,
        tol=tol,
        max_value=max_value,
        max_eval=max_eval,
    )
    return ConfidenceSet(boundaries=[(left, right)])
