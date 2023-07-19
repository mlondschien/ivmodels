import pytest

from ivmodels.linear_model import KClass
from ivmodels.pulse import PULSE
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import pulse_test


@pytest.mark.parametrize("rtol", [0.1, 0.01])
@pytest.mark.parametrize("p_min", [0.05, 0.01, 0.001])
@pytest.mark.parametrize("n, p, q, u", [(1000, 5, 1, 1), (1000, 2, 1, 2)])
def test_pulse(p_min, rtol, n, p, q, u):
    A, X, Y = simulate_gaussian_iv(n, p, q, u)
    pulse = PULSE(p_min=p_min, rtol=rtol).fit(X, Y.flatten(), A)

    # The PULSE selects the "largest" kappa s.t. p_value(kappa) >= p_min, where
    # "smallest" is defined up to rtol. I.e., p_value(kappa * (1 - rtol)) < p_value.
    test_p_value = pulse_test(A, Y.flatten() - pulse.predict(X))[1]
    assert test_p_value >= p_min
    next_kclass = KClass(kappa=pulse.kappa * (1 - rtol)).fit(X, Y.flatten(), A)
    assert pulse_test(A, Y.flatten() - next_kclass.predict(X))[1] < p_min
