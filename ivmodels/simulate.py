import numpy as np
import scipy

from ivmodels.utils import oproj


def simulate_guggenberger12(
    n, *, k, seed=0, h11=100, h12=1, rho=0.95, cov=None, return_beta=False, md=0
):
    """
    Generate data by process as proposed by :cite:t:`guggenberger2012asymptotic`.

    Will generate data

    .. math::

            X = Z \\Pi_X + V_X
            W = Z \\Pi_W + V_W
            y = X \\beta + W \\gamma + \\epsilon

        where :math:`\\epsilon, V_X, V_W` are jointly Gaussian with covariance matrix `cov` and `Z` is a matrix of independent
        centered Gaussian instruments.

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of instruments.
    seed : int, optional, default 0
        Random seed.
    h11 : float, optional, default 100
        Equal to :math:`\\sqrt{n} || \\Pi_X ||`.
    h12 : float, optional, default 1
        Equal to :math:`\\sqrt{n} || \\Pi_W ||`.
    rho : float, optional, default 0.95
        Equal to :math:`< \\Pi_X, \\Pi_W > / (|| \\Pi_X || || \\Pi_W ||)`.
    cov : np.ndarray, optional, default None
        Covariance matrix of the noise. If None, defaults to `[[1, 0, 0.95], [0, 1, 0.3], [0.95, 0.3, 1]]`.
    return_beta : bool, optional, default False
        Whether to return the true beta.

    Returns
    -------
    Z : np.ndarray of dimension (n, k)
        Instruments.
    X : np.ndarray of dimension (n, 1)
        Endogenous variables.
    y : np.ndarray of dimension (n,)
        Outcomes.
    C : None
        Empty
    W : np.ndarray of dimension (n, 1)
        Endogenous variables not of interest.
    beta : np.ndarray of dimension (1,)
        True beta. Only returned if ``return_beta`` is True.
    """
    if k < 2:
        raise ValueError("k must be at least 2")

    beta = np.array([[1]])
    gamma = np.array([[1]])

    rng = np.random.RandomState(seed)

    # Make sure that sqrt(n) || Pi_W || = h12 , sqrt(n) || Pi_X | = h11, and
    # < Pi_W, Pi_X> / (|| Pi_W || || Pi_X ||) = rho
    Pi_X = rng.normal(0, 1, (k, 1))
    Pi_X = Pi_X - Pi_X.mean(axis=0)
    Pi_X = Pi_X / np.linalg.norm(Pi_X)

    Pi_W = rng.normal(0, 1, (k, 1))
    Pi_W = Pi_W - Pi_W.mean(axis=0)
    Pi_W = oproj(Pi_X, Pi_W)
    Pi_W = Pi_W / np.linalg.norm(Pi_W)

    if rho >= 1:
        Pi_W = Pi_X
    elif rho <= -1:
        Pi_W = -Pi_X
    else:
        Pi_W = rho * Pi_X + np.sqrt(1 - rho**2) * Pi_W

    Pi_X = Pi_X / np.sqrt(n) * h11
    Pi_W = Pi_W / np.sqrt(n) * h12

    # Equal to Cov([eps, V_X, V_W]).
    if cov is None:
        cov = np.array([[1, 0, 0.95], [0, 1, 0.3], [0.95, 0.3, 1]])

    noise = scipy.stats.multivariate_normal.rvs(
        cov=cov,
        size=n,
        random_state=rng,
    )

    Z = rng.normal(0, 1, (n, k))

    Pi_ZD = rng.normal(0, 1, (k, md))
    D = rng.normal(0, 1, (n, md)) + Z @ Pi_ZD
    delta = rng.normal(0, 0.1, (md, 1))

    X = Z @ Pi_X + noise[:, 1:2]
    W = Z @ Pi_W + noise[:, 2:]
    y = X @ beta + W @ gamma + D @ delta + noise[:, 0:1]

    if return_beta:
        return Z, X, y.flatten(), None, W, D, np.concatenate([beta, delta]).flatten()
    else:
        return Z, X, y.flatten(), None, W, D


def simulate_gaussian_iv(
    n,
    *,
    mx,
    k,
    u=None,
    mw=0,
    mc=0,
    md=0,
    seed=0,
    include_intercept=True,
    return_beta=False,
    return_gamma=False,
):
    """
    Simulate a Gaussian IV dataset.

    Parameters
    ----------
    n : int
        Number of observations.
    mx : int
        Number of endogenous variables.
    k : int
        Number of instruments.
    u : int, optional
        Number of unobserved variables. If None, defaults to mx.
    mw : int, optional
        Number of endogenous variables not of interest.
    mc : int, optional
        Number of exogenous included variables.
    seed : int, optional
        Random seed.
    include_intercept : bool, optional
        Whether to include an intercept.
    return_beta : bool, optional
        Whether to return the true beta.
    return_gamma : bool, optional
        Whether to return the true gamma.

    Returns
    -------
    Z : np.ndarray of dimension (n, k)
        Instruments.
    X : np.ndarray of dimension (n, mx)
        Endogenous variables.
    y : np.ndarray of dimension (n,)
        Outcomes.
    C : np.ndarray of dimension (n, mc)
        Exogenous included variables.
    W : np.ndarray of dimension (n, mw)
        Endogenous variables not of interest.
    beta : np.ndarray of dimension (mx,)
        True beta. Only returned if ``return_beta`` is True.
    gamma : np.ndarray of dimension (mw,)
        True gamma. Only returned if ``return_gamma`` is True.
    """
    rng = np.random.RandomState(seed)

    m = mx + mw
    if u is None:
        u = m

    ux = rng.normal(0, 1, (u, m))
    uy = rng.normal(0, 1, (u, 1))

    alpha = rng.normal(0, 1, (mc + md, 1))
    beta = rng.normal(0, 1, (m, 1))

    Pi_ZX = rng.normal(0, 1, (k, m))
    Pi_CX = rng.normal(0, 1, (mc + md, m))
    Pi_CZ = rng.normal(0, 1, (mc + md, k))

    U = rng.normal(0, 1, (n, u))
    C = rng.normal(0, 1, (n, mc + md))
    if include_intercept:
        C += rng.normal(0, 1, (1, mc + md))

    Z = rng.normal(0, 1, (n, k)) + C @ Pi_CZ
    if include_intercept:
        Z += rng.normal(0, 1, (1, k))

    X = Z @ Pi_ZX + C @ Pi_CX + U @ ux
    X += rng.normal(0, 1, (n, m)) + include_intercept * rng.normal(0, 1, (1, m))
    y = C @ alpha + X @ beta + U @ uy
    y += rng.normal(0, 1, (n, 1)) + include_intercept * rng.normal(0, 1, (1, 1))

    X, W, C, D = X[:, :mx], X[:, mx:], C[:, :mc], C[:, mc:]

    gamma0 = beta[mx:]
    beta0 = np.concatenate([beta[:mx], alpha[mc:]])

    if return_beta and return_gamma:
        return Z, X, y.flatten(), C, W, D, beta0, gamma0
    elif return_beta:
        return Z, X, y.flatten(), C, W, D, beta0
    elif return_gamma:
        return Z, X, y.flatten(), C, W, D, gamma0
    else:
        return Z, X, y.flatten(), C, W, D
