import pytest

from anchor_regression.linear_model import PULSE
from anchor_regression.testing import simulate_iv
from anchor_regression.utils import anderson_rubin_test


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rtol", [0.1, 0.01])
@pytest.mark.parametrize("p_value", [0.05, 0.01, 0.001])
def test_pulse(seed, p_value, rtol):
    X, Y, A = simulate_iv(discrete=False, p=2, seed=seed, shift=0, dim_y=1)
    pulse = PULSE(p_value=p_value, rtol=rtol, gamma_max=1000000).fit(X, Y, A)

    test_p_value = anderson_rubin_test(A, Y - pulse.predict(X))[1]
    assert test_p_value >= p_value
    assert test_p_value < p_value * (1 + 2 * rtol)
    assert anderson_rubin_test(A, Y - pulse.predict(X))[1] > p_value
