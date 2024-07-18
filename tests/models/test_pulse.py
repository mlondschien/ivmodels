import numpy as np
import pytest

from ivmodels.models.kclass import KClass
from ivmodels.models.pulse import PULSE
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests.pulse import pulse_test


@pytest.mark.parametrize("rtol", [0.1, 0.01])
@pytest.mark.parametrize("p_min", [0.05, 0.01, 0.001])
@pytest.mark.parametrize("n, mx, k, u", [(1000, 5, 1, 1), (1000, 2, 1, 2)])
def test_pulse(p_min, rtol, n, mx, k, u):
    A, X, Y, _, _, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=u, seed=1)
    pulse = PULSE(p_min=p_min, rtol=rtol, kappa_max=0.999).fit(X, Y.flatten(), A)

    # The PULSE selects the "smallest" kappa s.t. p_value(kappa) >= p_min, where
    # "smallest" is defined up to rtol in p_value. I.e.,
    # p_value(kappa) / (1 + rtol) < p_min. If p_value(0) >= p_min, then the PULSE
    # selects kappa = 0.
    test_p_value = pulse_test(A, X, Y, pulse.coef_)[1]
    assert test_p_value >= p_min

    kclass = KClass(kappa=pulse.kappa_).fit(X, Y.flatten(), A)
    assert np.allclose(kclass.coef_, pulse.coef_)
    assert kclass.intercept_ == pulse.intercept_

    if not pulse.kappa == 0:
        test_p_value / (1 + rtol) < p_min
