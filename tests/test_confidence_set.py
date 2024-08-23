import numpy as np
import pytest

from ivmodels.confidence_set import ConfidenceSet


@pytest.mark.parametrize(
    "boundaries, values, finite, empty, length",
    [
        ([], {0: 1}, True, True, 0),
        ([(-1, 1)], {-2: 1, 1: -1, 0: -1}, True, False, 2),
        ([(-np.inf, np.inf)], {-7: -1}, False, False, np.inf),
        ([(-np.inf, 0), (1, 20)], {0.5: 1, 12: -1}, False, False, np.inf),
    ],
)
def test_confidence_set_call(boundaries, values, finite, empty, length):
    confidence_set = ConfidenceSet(boundaries)
    for value, expected in values.items():
        assert confidence_set(value) == expected
    assert confidence_set.is_finite() == finite
    assert confidence_set.is_empty() == empty
    assert confidence_set.length() == length
