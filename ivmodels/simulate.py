import numpy as np


def simulate_gaussian_iv(n, mx, k, u, mw=0, r=0, seed=0):
    """Simulate a Gaussian IV dataset."""
    rng = np.random.RandomState(seed)
    beta = rng.normal(0, 1, (mx, 1))

    ux = rng.normal(0, 1, (u, mx))
    uy = rng.normal(0, 1, (u, 1))
    uw = rng.normal(0, 1, (u, mw))

    alpha = rng.normal(0, 1, (r, 1))
    gamma = rng.normal(0, 1, (mw, 1))

    Pi_ZX = rng.normal(0, 1, (k, mx))
    Pi_ZW = rng.normal(0, 1, (k, mw))
    Pi_CX = rng.normal(0, 1, (r, mx))
    Pi_CW = rng.normal(0, 1, (r, mw))

    U = rng.normal(0, 1, (n, u))

    Z = rng.normal(0, 1, (n, k))
    C = rng.normal(0, 1, (n, r))

    X = Z @ Pi_ZX + C @ Pi_CX + U @ ux + rng.normal(0, 1, (n, mx))
    W = Z @ Pi_ZW + C @ Pi_CW + U @ uw + rng.normal(0, 1, (n, mw))
    y = C @ alpha + X @ beta + W @ gamma + U @ uy + rng.normal(0, 1, (n, 1))

    return Z, X, y.flatten(), C, W
