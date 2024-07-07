import numpy as np
import pytest

from ivmodels.confidence_set import ConfidenceSet


@pytest.mark.parametrize(
    "boundaries, values, finite, empty",
    [
        ([], {0: 1}, True, True),
        ([(-1, 1)], {-2: 1, 1: -1, 0: -1}, True, False),
        ([(-np.inf, np.inf)], {-7: -1}, False, False),
        ([(-np.inf, 0), (1, 20)], {0.5: 1, 12: -1}, False, False),
    ],
)
def test_confidence_set_call(boundaries, values, finite, empty):
    confidence_set = ConfidenceSet(boundaries)
    for value, expected in values.items():
        assert confidence_set(value) == expected
    assert confidence_set.is_finite() == finite
    assert confidence_set.is_empty() == empty
