import numpy as np
import pylops
import pyproximal
import scipy

from ivmodels.utils import proj


def anderson_rubin_basis_pursuit(Z, X, y, alpha=0.05):
    """
    Run basis pursuit on the Anderson-Rubin alpha-confidence set.

    Returns `argmin ||beta||_1` subject to `AR(beta) <= alpha-quantile`.
    """
    n, q = Z.shape
    quantile = scipy.stats.f.ppf(1 - alpha, dfn=n - q, dfd=q)

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    X_proj = proj(Z, X)
    X_orth = X - X_proj
    y_proj = proj(Z, y)
    y_orth = y - y_proj

    X_tilde = scipy.linalg.cholesky(X.T @ (X_proj - q / (n - q) * quantile * X_orth))
    y_tilde = (
        scipy.linalg.inv(X_tilde.T) @ X.T @ (y_proj - q / (n - q) * quantile * y_orth)
    )
    s_tilde = (y_proj - q / (n - q) * quantile * y_orth).T @ X @ scipy.linalg.inv(
        X_tilde.T @ X_tilde
    ) @ X.T @ (y_proj - q / (n - q) * quantile * y_orth) - (
        y_proj - q / (n - q) * quantile * y_orth
    ).T @ y

    return basis_pursuit(X_tilde, y_tilde, s_tilde)


def basis_pursuit(A, y, s, n_iter=1000):
    """
    Solve the basis pursuit problem.

    Returns `argmin ||x||_1` subject to `||Ax - y||_2 <= s`.
    """
    _, m = A.shape

    Aop = pylops.MatrixMult(A)
    Aop.explicit = False

    f = pyproximal.L1()
    g = pyproximal.proximal.EuclideanBall(y, np.sqrt(s))

    L = np.real(pylops.MatrixMult(A.T @ A).eigs(1))[0]
    tau = 0.99
    mu = tau / L

    xinv = pyproximal.optimization.primaldual.PrimalDual(
        f, g, Aop, np.zeros(m), tau, mu, niter=n_iter
    )

    return xinv
