import numpy as np


def simulate_iv(discrete=False, p=1, dim_y=1, seed=0, shift=0):
    """Simulate an IV dataset."""
    rng = np.random.RandomState(seed)

    if discrete and shift == 0:
        A = 1 - 2 * rng.binomial(1, 0.5, size=(100, 1))
    elif discrete and shift != 0:
        A = np.ones((100, 1)) * shift
    else:
        A = rng.normal(size=(100, 1)) + shift

    H = rng.normal(size=(100, 1))
    X = rng.normal(size=(100, p)) + H + A
    Y = X[:, 0:dim_y] + 2 * H

    return X, Y, A
