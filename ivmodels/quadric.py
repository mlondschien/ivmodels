# In part adapted from https://gitlab.com/Loicvh/quadproj/-/blob/master/src/quadproj/quadrics.py
import numpy as np
import scipy


class Quadric:
    """
    A class to represent a quadric :math:`x^T A x + b^T x + c \\leq 0`.

    Internally, works with a standardized form of the quadric. If :math:`V^T D V = A`
    with :math:`D` diagonal and :math:`V` orthonormal, define
    :math:`x_\\mathrm{center} := -A^{-1} b / 2`,
    :math:`\\tilde x = V^T (x - x_\\mathrm{center})` and
    :math:`\\tilde c = c - x_\\mathrm{center}^T A x_\\mathrm{center}`. Then, the
    standardized form is given by :math:`\\tilde x^T D \\tilde x + \\tilde c <= 0`.

    Parameters
    ----------
    A: np.ndarray of dimension (n, n)
        The matrix A of the quadratic form.
    b: np.ndarray of dimension (n,)
        The vector b of the quadratic form.
    c: float
        The constant c of the quadratic form.

    Attributes
    ----------
    center: np.ndarray of dimension (n,)
        The center of the quadric. Equal to :math:`-A^{-1} b / 2`.
    c_standardized: float
        The constant c of the standardized quadric. Equal to
        :math:`c - x_\\mathrm{center}^T A x_\\mathrm{center}`.
    D: np.ndarray of dimension (n,)
        The diagonal of the matrix :math:`D` in the eigenvalue decomposition
        :math:`V^T D V = A`.
    V: np.ndarray of dimension (n, n)
        The matrix :math:`V` in the eigenvalue decomposition :math:`V^T D V = A`.
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
        """Evaluate the quadric at :math:`x`.

        Parameters
        ----------
        x: np.ndarray of dimension (p,) or (n, p).
            The point(s) at which to evaluate the quadric.

        Returns
        -------
        np.ndarray of dimension (n,) or float
            The value(s) of the quadric at x. If x is a matrix, returns a vector of
            values. If x is a vector, returns a scalar.
        """
        if (
            (x.ndim == 1 and len(x) != self.A.shape[0])
            or (x.ndim == 2 and x.shape[1] != self.A.shape[0])
            or x.ndim > 2
        ):
            raise ValueError("x has the wrong dimension.")

        out = (x @ self.A * x).sum(axis=x.ndim - 1) + self.b.T @ x.T + self.c

        if x.ndim == 1:
            return out.item()

        return (x @ self.A * x).sum(axis=x.ndim - 1) + self.b.T @ x.T + self.c

    def forward_map(self, x_tilde):
        """Map from the standardized space to the original space."""
        return x_tilde @ self.V.T + self.center.T

    def inverse_map(self, x):
        """Map from the original space to the standardized space."""
        return (x - self.center.T) @ self.V

    def _boundary(self, num=200, error=True):
        assert len(self.D) <= 2

        if len(self.D) == 1:
            if self.D[0] * self.c_standardized > 0:
                raise ValueError("Quadric is empty.")
            else:
                return np.array(
                    [
                        self.center - np.sqrt(-self.c_standardized / self.D[0]),
                        self.center + np.sqrt(-self.c_standardized / self.D[0]),
                    ]
                ).T

        if np.all(self.D > 0) and (self.c_standardized > 0):
            if error:
                raise ValueError("Quadric is empty.")
            else:
                return np.zeros(shape=(0, 2))
        if np.all(self.D < 0) and (self.c_standardized <= 0):
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

    def volume(self):
        """Return the volume of the quadric."""
        if any(self.D <= 0):
            return np.inf
        elif self.c_standardized >= 0:
            return 0
        else:
            d = len(self.D)
            return (
                2
                / d
                * np.pi ** (d / 2)
                / scipy.special.gamma(d / 2)
                * np.sqrt(np.prod(-self.c_standardized / self.D))
            )

    def _projection(self, coordinate):
        """
        Return the projection of the quadric onto the coordinate.

        Solves argmin/max { x_coordinate | quadric(x) <= 0 }. The ``coordinate``-th
        coordinate of the solution can be seen as the boundary of the projection of the
        quadric onto the coordinate.

        Parameters
        ----------
        coordinate: int
            The coordinate onto which to project the quadric. Must be between 0 and
            p - 1.

        Returns
        -------
        (np.ndarray of dimension (p,), np.ndarray of dimension (p,))
            The lower and upper bounds of the projection of the quadric onto the
            coordinate.
        """
        one_hot = np.zeros_like(self.center)
        one_hot[coordinate] = 1
        solution = np.linalg.solve(self.A, one_hot)
        solution *= np.sqrt(-self.c_standardized / solution[coordinate])
        return (self.center - solution, self.center + solution)

    def project(self, coordinates):
        """
        Return the projection of the quadric onto ``coordinates``.

        For a quadric
        :math:`(x - x_\\mathrm{center})^T A (x - x_\\mathrm{center}) + c \\leq 0` and
        any matrix :math:`B \\in \\mathbb{R}^{q \\times p}` of rank :math:`q`, the
        projection of the quadric onto the coordinates given by the columns of :math:`B`
        is given by

        .. math::
           (Bx - Bx_\\mathrm{center})^T (B^T A^{-1} B)^{-1} (Bx - Bx_\\mathrm{center}) + c \\leq 0.

        Here, :math:`B` is given by ``coordinates``, with :math:`B_{i, j} = 1` if
        ``coordinates[i-1] == j`` and :math:`B_{i, j} = 0` otherwise for
        :math:`i = 1, \\ldots, q` and :math:`j = 1, \\ldots, p`.

        Parameters
        ----------
        coordinates: list of int
            The coordinates onto which to project the quadric. Entries must be unique
            and be between 0 and p - 1.

        Returns
        -------
        Quadric
            The projection of the quadric onto the coordinates.

        """
        if isinstance(coordinates, int):
            coordinates = [coordinates]

        if len(coordinates) == 0:
            raise ValueError("No coordinates specified.")

        if len(np.unique(coordinates)) != len(coordinates):
            raise ValueError("Coordinates must be unique.")

        if any([c < 0 or c >= self.A.shape[0] for c in coordinates]):
            raise ValueError("Coordinates must be between 0 and p - 1.")

        B = np.zeros((self.A.shape[0], len(coordinates)))
        for i, c in enumerate(coordinates):
            B[c, i] = 1

        A_inv = np.linalg.inv(self.A)
        A_new = np.linalg.inv(B.T @ A_inv @ B)
        b_new = A_new @ B.T @ A_inv @ self.b
        c_new = (
            self.c_standardized
            + 1.0 / 4.0 * self.b.T @ A_inv @ B @ A_new @ B.T @ A_inv @ self.b
        )

        return Quadric(A_new, b_new, c_new)
