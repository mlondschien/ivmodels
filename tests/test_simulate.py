import pytest

from ivmodels.simulate import simulate_gaussian_iv


@pytest.mark.parametrize("n, p, q, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_simulate_gaussian_iv(n, p, q, u):
    Z, X, y = simulate_gaussian_iv(n, p, q, u)
    assert Z.shape == (n, q)
    assert X.shape == (n, p)
    assert y.shape == (n, 1)
