import pytest

from anchor_regression.linear_model import AnchorRegression
from anchor_regression.pulse import PULSE
from anchor_regression.simulate import simulate_gaussian_iv
from anchor_regression.tests import pulse_test


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rtol", [0.1, 0.01])
@pytest.mark.parametrize("p_value", [0.05, 0.01, 0.001])
@pytest.mark.parametrize("n, p, q, u", [(100, 5, 2, 1), (100, 2, 1, 2)])
def test_pulse(seed, p_value, rtol, n, p, q, u):
    A, X, Y = simulate_gaussian_iv(n, p, q, u)
    pulse = PULSE(p_value=p_value, rtol=rtol, gamma_max=1e4).fit(X, Y, A)

    # The PULSE selects the "smallest" gamma s.t. p_value(gamma) > p_value, where
    # "smallest" is defined up to rtol. I.e., p_value(gamma * (1 - rtol)) < p_value.
    test_p_value = pulse_test(A, Y - pulse.predict(X))[1]
    assert test_p_value >= p_value
    next_ar = AnchorRegression(gamma=pulse.gamma * (1 - rtol)).fit(X, Y, A)
    assert pulse_test(A, Y - next_ar.predict(X))[1] < p_value
