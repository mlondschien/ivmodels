import pytest

from ivmodels.simulate import simulate_gaussian_iv


@pytest.mark.parametrize(
    "n, mx, k, r, u, mw", [(100, 2, 2, 1, 1, 2), (100, 2, 5, 0, 2, 0)]
)
def test_simulate_gaussian_iv(n, mx, k, r, u, mw):
    Z, X, y, C, W = simulate_gaussian_iv(n, mx, k, u, mw, r)
    assert Z.shape == (n, k)
    assert X.shape == (n, mx)
    assert y.shape == (n,)
    assert C.shape == (n, r)
    assert W.shape == (n, mw)
