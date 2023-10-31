import numpy as np


def simulate_gaussian_iv(n, p, q, u, r=0, seed=0):
    """Simulate a Gaussian IV dataset."""
    rng = np.random.RandomState(seed)

    ux = rng.normal(0, 1, (u, p))
    uy = rng.normal(0, 1, (u, 1))
    uw = rng.normal(0, 1, (u, r))
    beta = rng.normal(0, 0.1, (p, 1))
    Pi_X = rng.normal(0, 1, (q, p))
    Pi_W = rng.normal(0, 1, (q, r))

    U = rng.normal(0, 1, (n, u))

    Z = rng.normal(0, 1, (n, q))
    X = Z @ Pi_X + U @ ux + rng.normal(0, 1, (n, p))
    W = Z @ Pi_W + U @ uw + rng.normal(0, 1, (n, r))
    y = X @ beta + U @ uy + rng.normal(0, 1, (n, 1))

    if r == 0:
        return Z, X, y
    else:
        return Z, X, y, W
