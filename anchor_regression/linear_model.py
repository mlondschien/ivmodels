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


class AnchorMixin:
    """Mixin class for anchor regression models."""

    def __init__(self, gamma, anchor_names=None, anchor_regex=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma = gamma

        if anchor_names is not None or anchor_regex is not None:
            if not _PANDAS_INSTALLED:
                raise ImportError("pandas is required to use anchor columns or regex")

        self.anchor_names = anchor_names
        self.anchor_regex = anchor_regex

    def _X_a(self, X, a=None, check=True):
        """
        Extract anchor columns from X and a.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data. Must be a pandas DataFrame if `anchor_names` or `anchor_regex`
            is not None.
        a: array-like, shape (n_samples, n_anchors), optional
            The anchor data. If None, `anchor_names` or `anchor_regex` must be specified.
        check: bool, optional
            Whether to check the input data for consistency.

        Returns
        -------
        X: array-like, shape (n_samples, n_features - n_anchors)
            The input data with anchor columns removed.
        a: array-like, shape (n_samples, n_anchors)
            The anchor data.

        Raises
        ------
        ValueError
            If `check` is True and `a`, `anchor_names`, and `anchor_regex` are all None.
        ValueError
            If `check` is True and `a` is not None and `anchor_names` or `anchor_regex`
            is not None.
        ValueError
            If `check` is True and `anchor_names` or `anchor_regex` is not None and
            `X` is not a pandas DataFrame.
        ValueError
            If `check` is True, `anchor_regex` is specified and no columns are matched.
        ValueError
            If `check` is True, `anchor_names` is specified, and some columns in
            `anchor_names` are missing in `X`.
        """
        if a is not None:
            if self.anchor_names is not None or self.anchor_regex is not None and check:
                raise ValueError(
                    "If `anchor_names` or `anchor_regex` is specified, "
                    "then `a` must be None."
                )
            else:
                return X, a
        else:
            if self.anchor_names is None and self.anchor_regex is None:
                if check:
                    raise ValueError(
                        "If `anchor_names` and `anchor_regex` are None, "
                        "then `a` must be specified."
                    )
                else:
                    return X, np.zeros(shape=(X.shape[0], 0))

            if not _PANDAS_INSTALLED:
                raise ImportError("pandas is required to use anchor_columns or regex.")

            if not isinstance(X, pd.DataFrame):
                if check:
                    raise ValueError(
                        "If `anchor_names` or `anchor_regex` is specified, "
                        "`X` must be a pandas DataFrame."
                    )
                else:
                    return X, None
            else:
                anchor_columns = pd.Index([])

                if self.anchor_regex is not None:
                    matched_columns = X.columns[
                        X.columns.str.contains(self.anchor_regex)
                    ]
                    if len(matched_columns) == 0 and check:
                        raise ValueError(
                            f"No columns in X matched the regex {self.anchor_regex}"
                        )
                    anchor_columns = anchor_columns.union(matched_columns)

                if self.anchor_names is not None:
                    included_columns = X.columns.intersection(self.anchor_names)
                    if len(included_columns) < len(self.anchor_names) and check:
                        raise ValueError(
                            "The following anchor columns were not found in X: "
                            f"{set(self.anchor_names) - set(included_columns)}"
                        )
                    anchor_columns = anchor_columns.union(included_columns)

                return X.drop(anchor_columns, axis=1), X[anchor_columns]

    def fit(self, X, y, a=None):
        """
        Fit an anchor regression model [1]_.

        If `anchor_names` or `anchor_regex` are specified, `X` must be a
        pandas DataFrame containing columns `anchor_names` and `a` must be
        `None`. At least one one of `a`, `anchor_names`, and `anchor_regex`
        must be specified.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training input samples. If `anchor_names` or `anchor_regex`
            are specified, `X` must be a pandas DataFrame containing columns
            `anchor_names`.
        y: array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values.
        a: array-like, shape (n_samples, n_anchors), optional
            The anchor values. If `anchor_names` or `anchor_regex` are
            specified, `a` must be `None`. If `a` is specified, `anchor_names` and
            `anchor_regex` must be `None`.
        """
        X, a = self._X_a(X, a)

        x_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)

        X = X - x_mean
        y = y - y_mean

        super().fit(
            X - (1 - np.sqrt(self.gamma)) * proj(a, X),
            y - (1 - np.sqrt(self.gamma)) * proj(a, y),
        )

        self.intercept_ = -np.matmul(self.coef_, x_mean) + y_mean

        return self

    def predict(self, X):  # noqa D
        X, _ = self._X_a(X, a=None, check=False)
        return super().predict(X)


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
    anchor_names: str or list of str, optional
        The names of the columns in `X` that should be used as anchors. Requires `X` to
        be a pandas DataFrame.
    anchor_regex: str, optional
        A regex that is used to select columns in `X` that should be used as anchors.
        Requires `X` to be a pandas DataFrame. If both `anchor_names` and
        `anchor_regex` are specified, the union of the two is used.

    References
    ----------
    .. [1] https://arxiv.org/abs/1801.06229
    """

    def __init__(self, gamma, anchor_names=None, anchor_regex=None):
        super().__init__(
            gamma=gamma,
            anchor_names=anchor_names,
            anchor_regex=anchor_regex,
            fit_intercept=False,
        )


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
    anchor_names: str or list of str, optional, default = None
        The names of the columns in `X` that should be used as anchors. Requires `X` to
        be a pandas DataFrame.
    anchor_regex: str, optional, default = None
        A regex that is used to select columns in `X` that should be used as anchors.
        Requires `X` to be a pandas DataFrame. If both `anchor_names` and
        `anchor_regex` are specified, the union of the two is used.
    alpha: float, optional, default=1.0
        The ridge regularization parameter. Higher values correspond to stronger
        regularization.

    References
    ----------
    .. [1] https://arxiv.org/abs/1801.06229
    """

    def __init__(self, gamma, anchor_names=None, anchor_regex=None, alpha=1.0):
        super().__init__(
            gamma=gamma,
            anchor_names=anchor_names,
            anchor_regex=anchor_regex,
            alpha=alpha,
            fit_intercept=False,
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
    anchor_names: str or list of str, optional, default = None
        The names of the columns in `X` that should be used as anchors. Requires `X` to
        be a pandas DataFrame.
    anchor_regex: str, optional, default = None
        A regex that is used to select columns in `X` that should be used as anchors.
        Requires `X` to be a pandas DataFrame. If both `anchor_names` and
        `anchor_regex` are specified, the union of the two is used.
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
        self, gamma, anchor_names=None, anchor_regex=None, alpha=1.0, l1_ratio=0.5
    ):
        super().__init__(
            gamma=gamma,
            anchor_names=anchor_names,
            anchor_regex=anchor_regex,
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
