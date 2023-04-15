import pytest

from anchor_regression.linear_model import LinearAnchorRegression
from anchor_regression.pulse import PULSE
from anchor_regression.testing import simulate_iv
from anchor_regression.utils import pulse_test


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rtol", [0.1, 0.01])
@pytest.mark.parametrize("p_value", [0.05, 0.01, 0.001])
def test_pulse(seed, p_value, rtol):
    X, Y, A = simulate_iv(discrete=False, p=2, seed=seed, shift=0, dim_y=1)
    pulse = PULSE(p_value=p_value, rtol=rtol, gamma_max=1e4).fit(X, Y, A)

    # The PULSE selects the "smallest" gamma s.t. p_value(gamma) > p_value, where
    # "smallest" is defined up to rtol. I.e., p_value(gamma * (1 - rtol)) < p_value.
    test_p_value = pulse_test(A, Y - pulse.predict(X))[1]
    assert test_p_value >= p_value
    next_ar = LinearAnchorRegression(gamma=pulse.gamma * (1 - rtol)).fit(X, Y, A)
    assert pulse_test(A, Y - next_ar.predict(X))[1] < p_value
