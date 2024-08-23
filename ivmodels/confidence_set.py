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

        return " U ".join(
            f"[{left.__format__(format_spec)}, {right.__format__(format_spec)}]"
            for left, right in sorted(self.boundaries, key=lambda x: x[0])
        )

    def __repr__(self):  # noqa D
        return f"{self}"

    def is_empty(self):
        """Return True if the confidence set is empty."""
        return len(self.boundaries) == 0

    def is_finite(self):
        """Return True if the confidence set is finite."""
        return all(np.isfinite(x) for b in self.boundaries for x in b)

    def _boundary(self):
        """Return array containing all finite boundary points of the confidence set."""
        return np.array(
            [x for b in self.boundaries for x in b if np.isfinite(x)]
        ).reshape(-1, 1)

    @staticmethod
    def from_quadric(quadric):
        """Create a 1D confidence set from a quadric."""
        if not quadric.dim() == 1:
            raise ValueError("Can only convert 1D Quadric to ConfidenceSet.")

        if quadric.is_empty():
            return ConfidenceSet([])

        boundary = sorted(quadric._boundary().flatten())
        if quadric.is_bounded():
            return ConfidenceSet([(boundary[0], boundary[1])])
        elif len(boundary) == 0:
            return ConfidenceSet([(-np.inf, np.inf)])
        else:
            return ConfidenceSet([(-np.inf, boundary[0]), (boundary[1], np.inf)])

    def length(self):
        """Return the length of the confidence set."""
        return sum(right - left for left, right in self.boundaries)
