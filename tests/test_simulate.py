import pytest

from ivmodels.simulate import simulate_gaussian_iv


@pytest.mark.parametrize(
    "n, mx, k, mc, u, mw, md", [(100, 2, 2, 1, 1, 2, 1), (100, 2, 5, 0, 2, 0, 0)]
)
def test_simulate_gaussian_iv(n, mx, k, mc, u, mw, md):
    Z, X, y, C, W, D = simulate_gaussian_iv(n=n, mx=mx, k=k, mc=mc, u=u, mw=mw)
    assert Z.shape == (n, k)
    assert X.shape == (n, mx)
    assert y.shape == (n,)
    assert C.shape == (n, mc)
    assert W.shape == (n, mw)
    assert D.shape == (n, md)
