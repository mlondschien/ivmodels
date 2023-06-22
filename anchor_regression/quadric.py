# In part adapted from https://gitlab.com/Loicvh/quadproj/-/blob/master/src/quadproj/quadrics.py
import numpy as np


class Quadric:
    """
    A class to represent a quadric x^T A x + b^T x + c <= 0.

    Internally, works with a standardized form of the quadric. If `V^T D V = A` with
    D diagonal and V orthogonal, define `center=-A^{-1}b / 2` and
    `x_tilde=V^T (x - center)`. Then the standardized form is given by
    `x_tilde^T D x_tilde + c_standardized <= 0`.

    Parameters
    ----------
    A: np.ndarray of dimension (n, n)
        The matrix A of the quadratic form.
    b: np.ndarray of dimension (n,)
        The vector b of the quadratic form.
    c: float
        The constant c of the quadratic form.
    """

    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

        if not np.allclose(self.A, self.A.T):
            raise ValueError("Matrix `A` needs to be symmetric.")

        self.center = np.linalg.solve(A, -b / 2.0)
        self.c_standardized = c - self.center.T @ A @ self.center

        eigenvalues, eigenvectors = np.linalg.eig(A)
        argsort = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[argsort]
        eigenvectors = eigenvectors[:, argsort]
        assert np.allclose(A, eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)

        self.D = eigenvalues
        self.V = eigenvectors

    def __call__(self, x):
        """Evaluate the quadric at x."""
        return (x @ self.A * x).sum(axis=1) + self.b @ x.T + self.c

    def forward_map(self, x_tilde):
        """Map from the standardized space to the original space."""
        return x_tilde @ self.V.T + self.center

    def inverse_map(self, x):
        """Map from the original space to the standardized space."""
        return (x - self.center) @ self.V

    def _boundary(self, num=200, error=True):
        assert len(self.D) == 2

        if np.all(self.D > 0) and (self.c_standardized > 0):
            if error:
                raise ValueError("Quadric is empty.")
            else:
                return np.zeros(shape=(0, 2))
        if np.all(self.D < 0) and (self.c_standardized < 0):
            if error:
                raise ValueError("The quadric is the whole space.")
            else:
                return np.zeros(shape=(0, 2))

        # If both entries of D have the opposite sign as c_standardized, the
        # boundary of the quadric is an ellipse.
        if (np.all(self.D > 0) and (self.c_standardized < 0)) or (
            np.all(self.D < 0) and (self.c_standardized > 0)
        ):
            t = np.linspace(0, 2 * np.pi, num=num)
            circle = np.stack([np.cos(t), np.sin(t)]).T
            points_tilde = circle / np.sqrt(-self.D / self.c_standardized)
            points = self.forward_map(points_tilde)
            return points

        else:  # D has one positive and one negative entries
            t = np.linspace(-5, 5, num=num)
            assert self.D[0] > 0 and self.D[1] < 0  # D is sorted descending

            if self.c_standardized < 0:
                ellipsoid = np.concatenate(
                    [
                        np.stack([np.cosh(t), np.sinh(t)]).T,
                        np.stack([-np.cosh(t), -np.sinh(t)]).T,
                    ]
                )
                points_tilde = ellipsoid / np.sqrt(np.abs(self.D / self.c_standardized))
                points = self.forward_map(points_tilde)
                return points
            else:
                ellipsoid = np.concatenate(
                    [
                        np.stack([np.sinh(t), np.cosh(t)]).T,
                        np.stack([-np.sinh(t), -np.cosh(t)]).T,
                    ]
                )
                points_tilde = ellipsoid / np.sqrt(np.abs(self.D / self.c_standardized))
                points = self.forward_map(points_tilde)
                return points
