import logging
import re

import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

from anchor_regression.utils import proj

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
                    "pandas is required to use instrument columns or regex"
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
        Extract the alpha parameter from the kappa parameter.

        Parameters
        ----------
        kappa: float or str
            The kappa parameter. Must be a float, 'fuller(a)' for some integer or float
            a, 'fuller', or 'liml'.

        Returns
        -------
        fuller_alpha: float
            The alpha parameter. If kappa is a float, then alpha = kappa. If kappa is
            'fuller(a)' for some integer or float a, then alpha = a. If kappa is
            'fuller', then alpha = 1. If kappa is 'liml', then alpha = 0.
        """
        if not isinstance(kappa, str):
            return float(kappa)

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

    def _lambda_liml(self, X, y, Z=None, X_proj=None, y_proj=None):
        """Compute the lambda parameter of the LIML estimator.

        Either `Z` or both `X_proj` and `y_proj` must be specified.

        Parameters
        ----------
        X: np.ndarray of dimension (n, k).
            Possibly endogenous regressors.
        y: np.ndarray of dimension (n,).
            Outcome.
        Z: np.ndarray of dimension (n, l), optional, default=None.
            Instruments.
        X_proj: np.ndarray of dimension (n, k), optional, default=None.
            Projection of X onto the subspace orthogonal to Z.
        y_proj: np.ndarray of dimension (n, 1), optional, default=None.
            Projection of y onto the subspace orthogonal to Z.

        Returns
        -------
        lambda_liml: float
            Smallest eigenvalue of `((X y)^T (X y))^{-1} (X y)^T P_Z (X y)`, where P_Z
            is the projection matrix onto the subspace spanned by Z.
        """
        if X_proj is None:
            X_proj = proj(Z, X)
        if y_proj is None:
            y_proj = proj(Z, y)

        Xy = np.concatenate([X, y], axis=1)
        Xy_proj = np.concatenate([X_proj, y_proj], axis=1)
        W = np.linalg.solve(Xy.T @ Xy, Xy.T @ Xy_proj)
        return min(np.linalg.eigvals(W))

    def _solve_normal_equations(self, X, y, X_proj, y_proj, alpha=0):
        if alpha != 0:
            raise NotImplementedError("alpha != 0 not yet implemented.")
        return np.linalg.solve(
            X.T @ (self.kappa * X_proj - (1 - self.kappa) * X),
            X.T @ (self.kappa * y_proj - (1 - self.kappa) * y),
        )

    def fit(self, X, y, Z=None):
        """
        Fit a k-class estimator.

        If `instrument_names` or `instrument_regex` are specified, `X` must be a
        pandas DataFrame containing columns `instrument_names` and `a` must be
        `None`. At least one one of `a`, `instrument_names`, and `instrument_regex`
        must be specified.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training input samples. If `instrument_names` or `instrument_regex`
            are specified, `X` must be a pandas DataFrame containing columns
            `instrument_names`.
        y: array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values.
        Z: array-like, shape (n_samples, n_instruments), optional
            The instrument values. If `instrument_names` or `instrument_regex` are
            specified, `Z` must be `None`. If `Z` is specified, `instrument_names` and
            `instrument_regex` must be `None`.
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
            self.fuller_alpha_ = self._fuller_alpha(self.kappa)
            self.lambda_liml_ = self._lambda_liml(X, y, X_proj=X_proj, y_proj=y_proj)
            self.kappa_ = 1 / (1 - self.lambda_liml_) - self.fuller_alpha_ / (n - q)
        else:
            self.kappa_ = self.kappa

        # If kappa <=1, the k-class estimator is an anchor regression estimator, i.e.,
        # sqrt( (1-kappa) * Id + kappa * P_Z) ) exists and we apply linear regression
        # to the transformed data.
        if self.kappa <= 1:
            X_tilde = (
                np.sqrt(1 - self.kappa) * X + (1 - np.sqrt(1 - self.kappa)) * X_proj
            )
            y_tilde = (
                np.sqrt(1 - self.kappa) * y + (1 - np.sqrt(1 - self.kappa)) * y_proj
            )

            super().fit(X_tilde, y_tilde)

        else:
            self._solve_normal_equations(
                X, y, X_proj, y_proj, alpha=getattr(self, "alpha", 0)
            )

        self.intercept_ = -np.matmul(self.coef_, x_mean) + y_mean

        return self

    def predict(self, X):  # noqa D
        X, _ = self._X_Z(X, Z=None, check=False)
        return super().predict(X)


class KClass(KClassMixin, LinearRegression):
    """K-Class estimator for linear regression."""

    def __init__(self, kappa=1, instrument_names=None, instrument_regex=None):
        super().__init__(
            kappa=kappa,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
            fit_intercept=False,
        )


class AnchorMixin(KClassMixin):
    """Mixin class for anchor regression."""

    def __init__(
        self, gamma=1, instrument_names=None, instrument_regex=None, *args, **kwargs
    ):
        self.gamma_ = gamma
        super().__init__(
            kappa=(gamma - 1) / gamma,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
            *args,
            **kwargs,
        )

    @property
    def gamma(self):  # noqa D
        return self.gamma_

    @gamma.setter
    def gamma(self, value):
        self.gamma_ = value
        self.kappa = (value - 1) / value


class LinearAnchorRegression(AnchorMixin, LinearRegression):
    """
    Linear regression with anchor regularization.

    This is based on OLS after a data transformation. First standardizes `X` and `y`
    by subtracting the column means as proposed in [1]_. Consequently, no anchor
    regularization is applied to the intercept.

    Parameters
    ----------
    gamma: float
        The anchor regularization parameter. Gamma=1 corresponds to standard OLS.
    instrument_names: str or list of str, optional
        The names of the columns in `X` that should be used as instruments (anchors).
        Requires `X` to be a pandas DataFrame.
    instrument_regex: str, optional
        A regex that is used to select columns in `X` that should be used as instruments
        (anchors). Requires `X` to be a pandas DataFrame. If both `instrument_names` and
        `instrument_regex` are specified, the union of the two is used.

    References
    ----------
    .. [1] https://arxiv.org/abs/1801.06229
    """

    def __init__(self, gamma=1, instrument_names=None, instrument_regex=None):
        super().__init__(gamma, instrument_names, instrument_regex)


class AnchorRidge(AnchorMixin, Ridge):
    """
    Linear regression with l2 and anchor regularization.

    This is based on Ridge regression after a data transformation. First standardizes
    `X` and `y` by subtracting the column means as proposed in [1]_. Consequently, no
    anchor regularization is applied to the intercept. It is recommended to normalize
    the data to have unit variance before using ridge regression.

    Parameters
    ----------
    gamma: float
        The anchor regularization parameter. Gamma=1 corresponds to standard OLS.
    instrument_names: str or list of str, optional, default = None
        The names of the columns in `X` that should be used as instruments (anchors).
        Requires `X` to be a pandas DataFrame.
    instrument_regex: str, optional, default = None
        A regex that is used to select columns in `X` that should be used as instruments
        (anchors). Requires `X` to be a pandas DataFrame. If both `instrument_names` and
        `instrument_regex` are specified, the union of the two is used.
    alpha: float, optional, default=1.0
        The ridge regularization parameter. Higher values correspond to stronger
        regularization.

    References
    ----------
    .. [1] https://arxiv.org/abs/1801.06229
    """

    def __init__(self, gamma=1, instrument_names=None, instrument_regex=None, alpha=0):
        super().__init__(
            gamma=gamma,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
            fit_intercept=False,
            alpha=alpha,
        )


class AnchorElasticNet(AnchorMixin, ElasticNet):
    """
    Linear regression with l1, l2, and anchor regularization.

    This is based on Ridge regression after a data transformation. First standardizes
    `X` and `y` by subtracting the column means as proposed in [1]_. Consequently, no
    anchor regularization is applied to the intercept. It is recommended to normalize
    the data to have unit variance before using ridge regression.

    Parameters
    ----------
    gamma: float
        The anchor regularization parameter. Gamma=1 corresponds to standard OLS.
    instrument_names: str or list of str, optional, default = None
        The names of the columns in `X` that should be used as instruments (anchors).
        Requires `X` to be a pandas DataFrame.
    instrument_regex: str, optional, default = None
        A regex that is used to select columns in `X` that should be used as instruments
        (anchors). Requires `X` to be a pandas DataFrame. If both `instrument_names` and
        `instrument_regex` are specified, the union of the two is used.
    alpha: float, optional, default=1.0
        Constant that multiplies the l1 and l2 penalty terms.
    l1_ratio: float, optional, default=0.5
        The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`. For `l1_ratio = 0`
        the penalty is an L2 penalty. For `l1_ratio = 1` it is an L1 penalty. For
        `0 < l1_ratio < 1`, the penalty is a combination of L1 and L2.

    References
    ----------
    .. [1] https://arxiv.org/abs/1801.06229
    """

    def __init__(
        self,
        gamma,
        instrument_names=None,
        instrument_regex=None,
        alpha=1.0,
        l1_ratio=0.5,
    ):
        super().__init__(
            gamma=gamma,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
            fit_intercept=False,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )

    def predict(self, *args, **kwargs):  # noqa D
        # Make output shape consistent with LinearAnchorRegression and AnchorRidge, as
        # sklearn is inconsistent:
        # https://github.com/scikit-learn/scikit-learn/issues/5058
        output = super().predict(*args, **kwargs)
        if len(output.shape) == 1:
            output = output.reshape(-1, 1)
        return output
