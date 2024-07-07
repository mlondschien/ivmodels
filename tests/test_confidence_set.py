import numpy as np
import pytest

from ivmodels.confidence_set import ConfidenceSet


@pytest.mark.parametrize(
    "boundaries, values",
    [
        ([(-1, 1)], {-2: 1, 1: -1, 0: -1}),
        ([(-np.inf, np.inf)], {-7: -1}),
        ([(-np.inf, 0), (1, np.inf)], {0.5: 1, 12: -1}),
    ],
)
def test_confidence_set_call(boundaries, values):
    confidence_set = ConfidenceSet(boundaries)
    for value, expected in values.items():
        assert confidence_set(value) == expected
