import numpy as np


def simulate_iv(n=100, discrete=False, p=1, dim_y=1, seed=0, shift=0):
    """Simulate an IV dataset."""
    rng = np.random.RandomState(seed)

    if discrete and shift == 0:
        A = 1 - 2 * rng.binomial(1, 0.5, size=(n, 1))
    elif discrete and shift != 0:
        A = np.ones((n, 1)) * shift
    else:
        A = rng.normal(size=(n, 1)) + shift

    H = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, p)) + H + A
    Y = X[:, 0:dim_y] + 2 * H

    return X, Y, A
