import numpy as np
import scipy

from ivmodels.utils import oproj


def simulate_guggenberger12(n, *, k, seed=0, return_beta=False):
    """
    Generate data by process as proposed by :cite:t:`guggenberger2012asymptotic`.

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of instruments.
    seed : int, optional, default 0
        Random seed.
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
    C : np.ndarray of dimension (n, 0)
        Empty
    W : np.ndarray of dimension (n, 1)
        Endogenous variables not of interest.
    beta : np.ndarray of dimension (1,)
        True beta. Only returned if ``return_beta`` is True.
    """
    beta = np.array([[1]])
    gamma = np.array([[1]])

    rng = np.random.RandomState(seed)

    # Make sure that sqrt(n) || Pi_W | ~ 1 , sqrt(n) || Pi_X | ~ 100, and
    # n < Pi_W, Pi_X> ~ 0.95 * 100
    Pi_X = rng.normal(0, 1, (k, 1))
    Pi_X = Pi_X - Pi_X.mean(axis=0)
    Pi_X = Pi_X / np.linalg.norm(Pi_X)

    Pi_W = rng.normal(0, 1, (k, 1))
    Pi_W = Pi_W - Pi_W.mean(axis=0)
    Pi_W = oproj(Pi_X, Pi_W)
    Pi_W = Pi_W / np.linalg.norm(Pi_W)

    Pi_W = 0.95 * Pi_X + np.sqrt(1 - 0.95**2) * Pi_W

    Pi_X = Pi_X / np.sqrt(n) * 100
    Pi_W = Pi_W / np.sqrt(n) * 1

    # Equal to Cov([eps, V_X, V_W]).
    cov = np.array([[1, 0, 0.95], [0, 1, 0.3], [0.95, 0.3, 1]])
    noise = scipy.stats.multivariate_normal.rvs(
        cov=cov,
        size=n,
        random_state=rng,
    )

    Z = rng.normal(0, 1, (n, k))

    X = Z @ Pi_X + noise[:, 1:2]
    W = Z @ Pi_W + noise[:, 2:]
    y = X @ beta + W @ gamma + noise[:, 0:1]

    if return_beta:
        return Z, X, y.flatten(), np.array((n, 0)), W, beta.flatten()
    else:
        return Z, X, y.flatten(), np.array((n, 0)), W


def simulate_gaussian_iv(
    n, *, mx, k, u=None, mw=0, mc=0, seed=0, include_intercept=True, return_beta=False
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
    """
    rng = np.random.RandomState(seed)
    beta = rng.normal(0, 1, (mx, 1))

    if u is None:
        u = mx

    ux = rng.normal(0, 1, (u, mx))
    uy = rng.normal(0, 1, (u, 1))
    uw = rng.normal(0, 1, (u, mw))

    alpha = rng.normal(0, 1, (mc, 1))
    gamma = rng.normal(0, 1, (mw, 1))

    Pi_ZX = rng.normal(0, 1, (k, mx))
    Pi_ZW = rng.normal(0, 1, (k, mw))
    Pi_CX = rng.normal(0, 1, (mc, mx))
    Pi_CW = rng.normal(0, 1, (mc, mw))
    Pi_CZ = rng.normal(0, 1, (mc, k))

    U = rng.normal(0, 1, (n, u))
    C = rng.normal(0, 1, (n, mc)) + include_intercept * rng.normal(0, 1, (1, mc))

    Z = (
        rng.normal(0, 1, (n, k))
        + include_intercept * rng.normal(0, 1, (1, k))
        + C @ Pi_CZ
    )

    X = Z @ Pi_ZX + C @ Pi_CX + U @ ux
    X += rng.normal(0, 1, (n, mx)) + include_intercept * rng.normal(0, 1, (1, mx))
    W = Z @ Pi_ZW + C @ Pi_CW + U @ uw
    W += rng.normal(0, 1, (n, mw)) + include_intercept * rng.normal(0, 1, (1, mw))
    y = C @ alpha + X @ beta + W @ gamma + U @ uy
    y += rng.normal(0, 1, (n, 1)) + include_intercept * rng.normal(0, 1, (1, 1))

    if return_beta:
        return Z, X, y.flatten(), C, W, beta.flatten()
    else:
        return Z, X, y.flatten(), C, W
