import numpy as np
import pytest

from ivmodels.tests.anderson_rubin import (
    more_powerful_subvector_anderson_rubin_critical_value_function,
)


@pytest.mark.parametrize(
    "k, alpha, hat_kappa_1_cvs",
    [
        (1, 0.1, [[1.1, 0.7], [2.1, 1.2], [3.3, 1.6], [5.0, 2.0], [8.8, 2.4]]),
        (
            1,
            0.01,
            [[1.0, 0.9], [2.0, 1.8], [3.0, 2.6], [5.0, 3.9], [10.0, 5.6], [16.6, 6.2]],
        ),
        (
            5,
            0.1,
            [
                [1.0, 0.9],
                [2.0, 1.8],
                [3.0, 2.6],
                [6.0, 4.8],
                [11.0, 7.2],
                [15.6, 8.2],
                [34.8, 9.0],
            ],
        ),
    ],
)
def testmore_powerful_subvector_anderson_rubin_critical_value_function(
    k, alpha, hat_kappa_1_cvs
):
    """Compare to tables 3, 7 in Guggenberger (2019)."""
    for hat_kappa_1, cv in hat_kappa_1_cvs:
        # Through rounding to one decimal place
        assert (
            more_powerful_subvector_anderson_rubin_critical_value_function(
                cv, hat_kappa_1, k, mW=0
            )
            <= alpha
        )
        assert (
            more_powerful_subvector_anderson_rubin_critical_value_function(
                cv - 0.1, hat_kappa_1, k, mW=0
            )
            >= alpha
        )


@pytest.mark.parametrize("k", [1, 5, 20])
@pytest.mark.parametrize("hat_kappa_1", [0.1, 1, 5, 100])
def test_more_powerful_sAR_critical_value_function_integrates_to_one(k, hat_kappa_1):
    assert np.isclose(
        more_powerful_subvector_anderson_rubin_critical_value_function(
            hat_kappa_1, hat_kappa_1, k, mW=0
        ),
        0,
        atol=2e-4,
    )
