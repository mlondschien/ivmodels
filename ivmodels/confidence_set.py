import numpy as np


class ConfidenceSet:
    """A class to represent a 1D confidence set.

    Parameters
    ----------
    boundaries: list of 2-tuples of floats.
        The boundaries of the confidence set. The confidence set is the union of
        the intervals defined by the boundaries.
    """

    def __init__(self, boundaries):
        self.boundaries = boundaries

    def __call__(self, x):  # noqa D
        """Return -1 if x is in the confidence set, 1 otherwise."""
        for left, right in self.boundaries:
            if left <= x <= right:
                return -1

        return 1

    def __format__(self, format_spec: str) -> str:  # noqa D
        if len(self.boundaries) == 0:
            return "\u2205"  # empty set symbol

        return " \u222A ".join(  # union symbol
            f"[{left.__format__(format_spec)}, {right.__format(format_spec)}]"
            for left, right in sorted(self.boundaries, key=lambda x: x[0])
        )

    def _boundary(self, error=True):
        return np.array(
            [x for b in self.boundaries for x in b if np.isfinite(x)]
        ).reshape(-1, 1)
