import numpy as np
import pytest

from ivmodels.kclass import KClass
from ivmodels.pulse import PULSE
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import pulse_test


@pytest.mark.parametrize("rtol", [0.1, 0.01])
@pytest.mark.parametrize("p_min", [0.05, 0.01, 0.001])
@pytest.mark.parametrize("n, p, q, u", [(1000, 5, 1, 1), (1000, 2, 1, 2)])
def test_pulse(p_min, rtol, n, p, q, u):
    A, X, Y = simulate_gaussian_iv(n, p, q, u, seed=1)
    pulse = PULSE(p_min=p_min, rtol=rtol, kappa_max=0.999).fit(X, Y.flatten(), A)

    # The PULSE selects the "smallest" kappa s.t. p_value(kappa) >= p_min, where
    # "smallest" is defined up to rtol in p_value. I.e.,
    # p_value(kappa) / (1 + rtol) < p_min. If p_value(0) >= p_min, then the PULSE
    # selects kappa = 0.
    test_p_value = pulse_test(A, Y.flatten() - pulse.predict(X))[1]
    assert test_p_value >= p_min

    kclass = KClass(kappa=pulse.kappa_).fit(X, Y.flatten(), A)
    assert np.allclose(kclass.coef_, pulse.coef_)
    assert kclass.intercept_ == pulse.intercept_

    if not pulse.kappa == 0:
        test_p_value / (1 + rtol) < p_min
