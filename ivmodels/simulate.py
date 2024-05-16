import numpy as np


def simulate_gaussian_iv(
    n, *, mx, k, u=None, mw=0, mc=0, seed=0, include_intercept=True, return_beta=False
):
    """Simulate a Gaussian IV dataset."""
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
