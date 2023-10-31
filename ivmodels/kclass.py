import logging
import re

import numpy as np
from glum import GeneralizedLinearRegressor

from ivmodels.utils import proj

try:
    import pandas as pd

    _PANDAS_INSTALLED = True
except ImportError:
    _PANDAS_INSTALLED = False

logger = logging.getLogger(__name__)


class KClassMixin:
    """Mixin class for k-class estimators."""

    def __init__(
        self, kappa=1, instrument_names=None, instrument_regex=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.kappa = kappa

        if instrument_names is not None or instrument_regex is not None:
            if not _PANDAS_INSTALLED:
                raise ImportError(
                    "pandas is required to use instrument columns or regex."
                )

        self.instrument_names = instrument_names
        self.instrument_regex = instrument_regex

    def _X_Z(self, X, Z=None, check=True):
        """
        Extract instrument columns from X and Z.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data. Must be a pandas DataFrame if `instrument_names` or
            `instrument_regex` is not None.
        Z: array-like, shape (n_samples, n_instruments), optional
            The instrument data. If None, `instrument_names` or `instrument_regex` must be
            specified.
        check: bool, optional
            Whether to check the input data for consistency.

        Returns
        -------
        X: array-like, shape (n_samples, n_features - n_instrument)
            The input data with instrument columns removed.
        Z: array-like, shape (n_samples, n_instrument)
            The instrument data.

        Raises
        ------
        ValueError
            If `check` is True and `Z`, `instrument_names`, and `instrument_regex` are all
            None.
        ValueError
            If `check` is True and `Z` is not None and `instrument_names` or
            `instrument_regex` is not None.
        ValueError
            If `check` is True and `instrument_names` or `instrument_regex` is not None
            and `X` is not a pandas DataFrame.
        ValueError
            If `check` is True, `instrument_regex` is specified and no columns are
            matched.
        ValueError
            If `check` is True, `instrument_names` is specified, and some columns in
            `instrument_names` are missing in `X`.
        """
        if Z is not None:
            if (
                self.instrument_names is not None
                or self.instrument_regex is not None
                and check
            ):
                raise ValueError(
                    "If `instrument_names` or `instrument_regex` is specified, "
                    "then `Z` must be None."
                )
            else:
                return X, Z
        else:
            if self.instrument_names is None and self.instrument_regex is None:
                if check:
                    raise ValueError(
                        "If `instrument_names` and `instrument_regex` are None, "
                        "then `Z` must be specified."
                    )
                else:
                    return X, np.zeros(shape=(X.shape[0], 0))

            if not _PANDAS_INSTALLED:
                raise ImportError(
                    "pandas is required to use instrument_columns or regex."
                )

            if not isinstance(X, pd.DataFrame):
                if check:
                    raise ValueError(
                        "If `instrument_names` or `instrument_regex` is specified, "
                        "`X` must be a pandas DataFrame."
                    )
                else:
                    return X, None
            else:
                instrument_columns = pd.Index([])

                if self.instrument_regex is not None:
                    matched_columns = X.columns[
                        X.columns.str.contains(self.instrument_regex)
                    ]
                    if len(matched_columns) == 0 and check:
                        raise ValueError(
                            f"No columns in X matched the regex {self.instrument_regex}"
                        )
                    instrument_columns = instrument_columns.union(matched_columns)

                if self.instrument_names is not None:
                    included_columns = X.columns.intersection(self.instrument_names)
                    if len(included_columns) < len(self.instrument_names) and check:
                        raise ValueError(
                            "The following instrument columns were not found in X: "
                            f"{set(self.instrument_names) - set(included_columns)}"
                        )
                    instrument_columns = instrument_columns.union(included_columns)

                return X.drop(instrument_columns, axis=1), X[instrument_columns]

    def _fuller_alpha(self, kappa):
        """
        Extract the Fuller alpha parameter from the kappa parameter.

        Parameters
        ----------
        kappa: str
            The kappa parameter. Must be ``"fuller(a)"`` for some integer or float
            ``a``, ``"fuller"``, or ``"liml"``.

        Returns
        -------
        fuller_alpha: float
            The alpha parameter. If kappa is ``"fuller(a)"`` for some integer or float
            ``a``, then ``alpha = a``. If kappa is ``"fuller"``, then ``alpha = 1``.
            If kappa is ``"liml"``, then ``alpha = 0``.
        """
        if not isinstance(kappa, str):
            raise ValueError(f"Invalid kappa {kappa}. Must be a string.")

        fuller_match = re.match(r"fuller(\(\d+\.?\d*\))?", kappa, re.IGNORECASE)
        liml_match = re.match("liml", kappa, re.IGNORECASE)

        if fuller_match is None and liml_match is None:
            raise ValueError(
                f"Invalid kappa: {kappa}. Must be a float or 'fuller(a)' for "
                f"some float integer a or 'liml'."
            )

        if fuller_match is not None and fuller_match.group(1) is not None:
            try:
                return float(fuller_match.group(1)[1:-1])
            except ValueError:
                raise ValueError(
                    f"Invalid kappa: {kappa}. Must be a float or 'fuller(a)' for "
                    f"some float or integer a or 'liml'."
                )
        elif fuller_match is not None:
            return 1.0
        else:
            return 0.0

    def _kappa_liml(self, X, y, Z=None, X_proj=None, y_proj=None):
        """Compute the kappa parameter of the LIML estimator.

        Either ``Z`` or both ``X_proj`` and ``y_proj`` must be specified.

        Parameters
        ----------
        X: np.ndarray of dimension (n, k)
            Possibly endogenous regressors.
        y: np.ndarray of dimension (n,)
            Outcome.
        Z: np.ndarray of dimension (n, l), optional, default=None.
            Instruments.
        X_proj: np.ndarray of dimension (n, k), optional, default=None.
            Projection of X onto the subspace orthogonal to Z.
        y_proj: np.ndarray of dimension (n, 1), optional, default=None.
            Projection of y onto the subspace orthogonal to Z.

        Returns
        -------
        kappa_liml: float
            One plus the smallest eigenvalue of
            ``((X y)^T M_Z (X y))^{-1} (X y)^T P_Z (X y)``, where
            ``P_Z`` is the projection matrix onto the subspace spanned by Z.
        """
        if X_proj is None:
            X_proj = proj(Z, X)
        if y_proj is None:
            y_proj = proj(Z, y)

        Xy = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        Xy_proj = np.concatenate([X_proj, y_proj.reshape(-1, 1)], axis=1)
        W = np.linalg.solve((Xy - Xy_proj).T @ Xy, Xy.T @ Xy_proj)
        return 1 + min(np.linalg.eigvals(W))

    def _solve_normal_equations(self, X, y, X_proj, y_proj, alpha=0):
        if alpha != 0:
            raise NotImplementedError("alpha != 0 not yet implemented.")

        return np.linalg.solve(
            X.T @ (self.kappa_ * X_proj + (1 - self.kappa_) * X),
            X.T @ (self.kappa_ * y_proj + (1 - self.kappa_) * y),
        ).T

    def fit(self, X, y, Z=None, *args, **kwargs):
        """
        Fit a k-class or anchor regression estimator.

        If ``instrument_names`` or ``instrument_regex`` are specified, ``X`` must be a
        pandas DataFrame containing columns ``instrument_names`` and ``Z`` must be
        ``None``. At least one one of ``Z``, ``instrument_names``, and
        ``instrument_regex`` must be specified.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training input samples. If ``instrument_names`` or ``instrument_regex``
            are specified, ``X`` must be a pandas DataFrame containing columns
            ``instrument_names``.
        y: array-like, shape (n_samples,)
            The target values.
        Z: array-like, shape (n_samples, n_instruments), optional
            The instrument values. If ``instrument_names`` or ``instrument_regex`` are
            specified, ``Z`` must be ``None``. If ``Z`` is specified,
            ``instrument_names`` and ``instrument_regex`` must be ``None``.
        """
        X, Z = self._X_Z(X, Z)

        n, q = X.shape[0], Z.shape[1]

        x_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)

        X = X - x_mean
        y = y - y_mean

        X_proj = proj(Z, X)
        y_proj = proj(Z, y)

        if isinstance(self.kappa, str):
            if self.kappa.lower() in {"tsls", "2sls"}:
                self.kappa_ = 1
            elif self.kappa.lower() == "ols":
                self.kappa_ = 0
            else:
                self.fuller_alpha_ = self._fuller_alpha(self.kappa)
                self.kappa_liml_ = self._kappa_liml(X, y, X_proj=X_proj, y_proj=y_proj)
                self.kappa_ = self.kappa_liml_ - self.fuller_alpha_ / (n - q)

        else:
            self.kappa_ = self.kappa

        # If kappa <=1, the k-class estimator is an anchor regression estimator, i.e.,
        # sqrt( (1-kappa) * Id + kappa * P_Z) ) exists and we apply linear regression
        # to the transformed data.
        if self.kappa_ <= 1:
            X_tilde = (
                np.sqrt(1 - self.kappa_) * X + (1 - np.sqrt(1 - self.kappa_)) * X_proj
            )
            y_tilde = (
                np.sqrt(1 - self.kappa_) * y + (1 - np.sqrt(1 - self.kappa_)) * y_proj
            )
            super().fit(X_tilde, y_tilde, *args, **kwargs)

        else:
            if args or kwargs:
                raise ValueError(
                    f"Arguments {args} and {kwargs} are not supported for kappa > 1."
                )
            self.coef_ = self._solve_normal_equations(
                X, y, X_proj=X_proj, y_proj=y_proj, alpha=getattr(self, "alpha", 0)
            )

        self.intercept_ = -np.matmul(self.coef_, x_mean) + y_mean

        return self

    def predict(self, X, *args, **kwargs):  # noqa D
        X, _ = self._X_Z(X, Z=None, check=False)
        return super().predict(X, *args, **kwargs)


class KClass(KClassMixin, GeneralizedLinearRegressor):
    """K-class estimator for instrumental variable regression.

    The k-class estimator with parameter :math:`\\kappa` is defined as

    .. math::

       \\hat\\beta_\\mathrm{k-class}(\\kappa) &:= \\arg\\min_\\beta \\
       (1 - \\kappa) \\| y - X \\beta \\|_2^2 + \\kappa \\|P_Z (y - X \\beta) \\|_2^2
       \\\\
       &= (X^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X)^{-1} X^T
       (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X) y,

    where :math:`P_Z = Z (Z^T Z)^{-1} Z^T` is the projection matrix onto the subspace
    spanned by :math:`Z` and :math:`\\mathrm{Id}` is the identity matrix.
    This includes the the ordinary least-squares (OLS) estimator (:math:`\\kappa = 0`),
    the two-stage least-squares (2SLS) estimator
    (:math:`\\kappa = 1`), the limited information maximum likelihood (LIML) estimator
    (:math:`\\kappa = \\hat\\kappa_\\mathrm{LIML}`), and the Fuller estimator
    (:math:`\\kappa = \\hat\\kappa_\\mathrm{LIML} - \\alpha / (n - q)`) as special
    cases.

    Parameters
    ----------
    kappa: float or { "ols", "tsls", "2sls", "liml", "fuller", "fuller(a)"}
        The kappa parameter of the k-class estimator. If float, then kappa must be in
        :math:`[0, \\hat\\kappa_\\mathrm{LIML}]`, where
        :math:`\\kappa_\\mathrm{LIML} \\geq 1` is 1 plus the smallest eigenvalue of the
        matrix :math:`((X \\ \\ y)^T M_Z (X \\ \\ y))^{-1} (X \\ \\ y)^T P_Z (X \\ y)`,
        where :math:`P_Z` is the projection matrix onto the subspace spanned by :math:`Z`
        and :math:`M_Z = Id - P_Z`.
        If string, then must be one of ``"ols"``, ``"2sls"``, ``"tsls"``, ``"liml"``,
        ``"fuller"``, or ``"fuller(a)"``, where ``a`` is numeric. If ``kappa="ols"``,
        then ``kappa=0`` and the k-class estimator is the ordinary least squares
        estimator. If ``kappa="tsls"`` or ``kappa="2sls"``, then ``kappa=1`` and the
        k-class estimator is the two-stage least-squares estimator. If ``kappa="liml"``,
        then :math:`\\kappa = \\hat\\kappa_\\mathrm{LIML}` is used. If
        ``kappa="fuller(a)"``, then
        :math:`\\kappa = \\hat\\kappa_\\mathrm{LIML} - a / (n - q)`, where
        :math:`n` is the number of observations and :math:`q = \\mathrm{dim}(Z)` is the
        number of instruments. The string ``"fuller"`` is interpreted as
        ``"fuller(1.0)"``, yielding an estimator that is unbiased up to
        :math:`O(1/n)` :cite:p:`fuller1977some`.
    instrument_names: str or list of str, optional
        The names of the columns in ``X`` that should be used as instruments.
        Requires ``X`` to be a pandas DataFrame. If both ``instrument_names`` and
        ``instrument_regex`` are specified, the union of the two is used.
    instrument_regex: str, optional
        A regex that is used to select columns in ``X`` that should be used as
        instruments. Requires ``X`` to be a pandas DataFrame. If both
        ``instrument_names`` and ``instrument_regex`` are specified, the union of the
        two is used.
    alpha: float, optional, default=0
        Regularization parameter for elastic net regularization.
    l1_ratio: float, optional, default=0
        Ratio of L1 to L2 regularization for elastic net regularization. For
        ``l1_ratio=0`` the penalty is an L2 penalty. For ``l1_ratio=1`` it is an L1
        penalty.

    Attributes
    ----------
    coef_: array-like, shape (n_features,)
        The estimated coefficients for the linear regression problem.
    intercept_: float
        The estimated intercept for the linear regression problem.
    kappa_: float
        The kappa parameter of the k-class estimator.
    fuller_alpha_: float
        If ``kappa`` is one of ``{"fuller", "fuller(a)", "liml"}`` for some numeric
        value ``a``, the alpha parameter of the Fuller estimator.
    kappa_liml_: float
        If ``kappa`` is one of ``{"fuller", "fuller(a)", "liml"}`` for some numeric
        value ``a``, the kappa parameter of the LIML estimator.

    References
    ----------
    .. bibliography::
       :filter: False

       fuller1977some
    """

    def __init__(
        self, kappa=1, instrument_names=None, instrument_regex=None, alpha=0, l1_ratio=0
    ):
        super().__init__(
            kappa=kappa,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
            family="gaussian",
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
        )
