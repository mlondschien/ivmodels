import numpy as np
import pytest


@pytest.fixture()
def simulate_iv():
    def _simulate_iv(n=100, discrete=False, p=1, seed=0, shift=0):
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
        Y = X[:, 0] + 2 * H[:, 0]

        return X, Y, A

    return _simulate_iv
