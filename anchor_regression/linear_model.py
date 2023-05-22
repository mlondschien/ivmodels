import logging

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
        self, kappa, exogenous_names=None, exogenous_regex=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.kappa = kappa

        if exogenous_names is not None or exogenous_regex is not None:
            if not _PANDAS_INSTALLED:
                raise ImportError(
                    "pandas is required to use exogenous columns or regex"
                )

        self.exogenous_names = exogenous_names
        self.exogenous_regex = exogenous_regex

    def _X_a(self, X, a=None, check=True):
        """
        Extract exogenous columns from X and a.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data. Must be a pandas DataFrame if `exogenous_names` or
            `exogenous_regex` is not None.
        a: array-like, shape (n_samples, n_exogenouss), optional
            The exogenous data. If None, `exogenous_names` or `exogenous_regex` must be
            specified.
        check: bool, optional
            Whether to check the input data for consistency.

        Returns
        -------
        X: array-like, shape (n_samples, n_features - n_exogenous)
            The input data with exogenous columns removed.
        a: array-like, shape (n_samples, n_exogenous)
            The exogenous data.

        Raises
        ------
        ValueError
            If `check` is True and `a`, `exogenous_names`, and `exogenous_regex` are all
            None.
        ValueError
            If `check` is True and `a` is not None and `exogenous_names` or
            `exogenous_regex` is not None.
        ValueError
            If `check` is True and `exogenous_names` or `exogenous_regex` is not None
            and `X` is not a pandas DataFrame.
        ValueError
            If `check` is True, `exogenous_regex` is specified and no columns are
            matched.
        ValueError
            If `check` is True, `exogenous_names` is specified, and some columns in
            `exogenous_names` are missing in `X`.
        """
        if a is not None:
            if (
                self.exogenous_names is not None
                or self.exogenous_regex is not None
                and check
            ):
                raise ValueError(
                    "If `exogenous_names` or `exogenous_regex` is specified, "
                    "then `a` must be None."
                )
            else:
                return X, a
        else:
            if self.exogenous_names is None and self.exogenous_regex is None:
                if check:
                    raise ValueError(
                        "If `exogenous_names` and `exogenous_regex` are None, "
                        "then `a` must be specified."
                    )
                else:
                    return X, np.zeros(shape=(X.shape[0], 0))

            if not _PANDAS_INSTALLED:
                raise ImportError(
                    "pandas is required to use exogenous_columns or regex."
                )

            if not isinstance(X, pd.DataFrame):
                if check:
                    raise ValueError(
                        "If `exogenous_names` or `exogenous_regex` is specified, "
                        "`X` must be a pandas DataFrame."
                    )
                else:
                    return X, None
            else:
                exogenous_columns = pd.Index([])

                if self.exogenous_regex is not None:
                    matched_columns = X.columns[
                        X.columns.str.contains(self.exogenous_regex)
                    ]
                    if len(matched_columns) == 0 and check:
                        raise ValueError(
                            f"No columns in X matched the regex {self.exogenous_regex}"
                        )
                    exogenous_columns = exogenous_columns.union(matched_columns)

                if self.exogenous_names is not None:
                    included_columns = X.columns.intersection(self.exogenous_names)
                    if len(included_columns) < len(self.exogenous_names) and check:
                        raise ValueError(
                            "The following exogenous columns were not found in X: "
                            f"{set(self.exogenous_names) - set(included_columns)}"
                        )
                    exogenous_columns = exogenous_columns.union(included_columns)

                return X.drop(exogenous_columns, axis=1), X[exogenous_columns]

    def fit(self, X, y, a=None):
        """
        Fit a k-class estimator.

        If `exogenous_names` or `exogenous_regex` are specified, `X` must be a
        pandas DataFrame containing columns `exogenous_names` and `a` must be
        `None`. At least one one of `a`, `exogenous_names`, and `exogenous_regex`
        must be specified.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training input samples. If `exogenous_names` or `exogenous_regex`
            are specified, `X` must be a pandas DataFrame containing columns
            `exogenous_names`.
        y: array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values.
        a: array-like, shape (n_samples, n_exogenouss), optional
            The exogenous values. If `exogenous_names` or `exogenous_regex` are
            specified, `a` must be `None`. If `a` is specified, `exogenous_names` and
            `exogenous_regex` must be `None`.
        """
        X, a = self._X_a(X, a)

        x_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)

        X = X - x_mean
        y = y - y_mean

        X_proj = proj(a, X)
        y_proj = proj(a, y)

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

        # If kappa >1, we need to solve the normal equations explicitly.
        else:
            self.coef_ = np.linalg.solve(
                ((1 - self.kappa) * X + self.kappa * X_proj).T @ X,
                ((1 - self.kappa) * X + self.kappa * X_proj).T @ y,
            )

        self.intercept_ = -np.matmul(self.coef_, x_mean) + y_mean

        return self

    def predict(self, X):  # noqa D
        X, _ = self._X_a(X, a=None, check=False)
        return super().predict(X)


class LinearAnchorRegression(KClassMixin, LinearRegression):
    """
    Linear regression with anchor regularization.

    This is based on OLS after a data transformation. First standardizes `X` and `y`
    by subtracting the column means as proposed in [1]_. Consequently, no anchor
    regularization is applied to the intercept.

    Parameters
    ----------
    gamma: float
        The anchor regularization parameter. Gamma=1 corresponds to standard OLS.
    exogenous_names: str or list of str, optional
        The names of the columns in `X` that should be used as anchors. Requires `X` to
        be a pandas DataFrame.
    exogenous_regex: str, optional
        A regex that is used to select columns in `X` that should be used as anchors.
        Requires `X` to be a pandas DataFrame. If both `exogenous_names` and
        `exogenous_regex` are specified, the union of the two is used.

    References
    ----------
    .. [1] https://arxiv.org/abs/1801.06229
    """

    def __init__(self, gamma, exogenous_names=None, exogenous_regex=None):
        self.gamma = gamma
        super().__init__(
            kappa=(gamma - 1) / gamma,
            exogenous_names=exogenous_names,
            exogenous_regex=exogenous_regex,
            fit_intercept=False,
        )


class AnchorRidge(KClassMixin, Ridge):
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
    exogenous_names: str or list of str, optional, default = None
        The names of the columns in `X` that should be used as anchors. Requires `X` to
        be a pandas DataFrame.
    exogenous_regex: str, optional, default = None
        A regex that is used to select columns in `X` that should be used as anchors.
        Requires `X` to be a pandas DataFrame. If both `exogenous_names` and
        `exogenous_regex` are specified, the union of the two is used.
    alpha: float, optional, default=1.0
        The ridge regularization parameter. Higher values correspond to stronger
        regularization.

    References
    ----------
    .. [1] https://arxiv.org/abs/1801.06229
    """

    def __init__(self, gamma, exogenous_names=None, exogenous_regex=None, alpha=1.0):
        self.gamma = gamma
        super().__init__(
            kappa=(gamma - 1) / gamma,
            exogenous_names=exogenous_names,
            exogenous_regex=exogenous_regex,
            alpha=alpha,
            fit_intercept=False,
        )


class AnchorElasticNet(KClassMixin, ElasticNet):
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
    exogenous_names: str or list of str, optional, default = None
        The names of the columns in `X` that should be used as anchors. Requires `X` to
        be a pandas DataFrame.
    exogenous_regex: str, optional, default = None
        A regex that is used to select columns in `X` that should be used as anchors.
        Requires `X` to be a pandas DataFrame. If both `exogenous_names` and
        `exogenous_regex` are specified, the union of the two is used.
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
        self, gamma, exogenous_names=None, exogenous_regex=None, alpha=1.0, l1_ratio=0.5
    ):
        self.gamma = gamma
        super().__init__(
            kappa=(gamma - 1) / gamma,
            exogenous_names=exogenous_names,
            exogenous_regex=exogenous_regex,
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
