import numpy as np


class ConfidenceSet:
    """A class to represent a confidence set."""

    def __init__(self, left, right, convex, empty=False, message=None):
        self.left = left
        self.right = right
        self.convex = convex
        self.empty = empty
        self.message = message

    def __call__(self, x):  # noqa D
        if self.empty:
            return 1

        between = self.left <= x <= self.right
        if self.convex and between:
            return -1
        elif not self.convex and not between:
            return -1
        else:
            return 1

    def __str__(self):  # noqa D
        if self.empty:
            return "[]"
        elif self.convex:
            return f"[{self.left}, {self.right}]"
        else:
            return f"(-infty, {self.left}] U [{self.right}, infty)"

    def _boundary(self, error=True):
        if self.empty:
            return np.zeros(shape=(0, 1))
        else:
            return np.array([[self.left], [self.right]])
