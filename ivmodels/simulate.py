import numpy as np


def simulate_gaussian_iv(n, p, q, u, seed=0):
    """Simulate a Gaussian IV dataset."""
    rng = np.random.RandomState(seed)

    delta = rng.normal(0, 1, (u, p))
    gamma = rng.normal(0, 1, (u, 1))

    beta = rng.normal(0, 0.1, (p, 1))
    Pi = rng.normal(0, 1, (q, p))

    U = rng.normal(0, 1, (n, u))

    Z = rng.normal(0, 1, (n, q))
    X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
    y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))

    return Z, X, y
