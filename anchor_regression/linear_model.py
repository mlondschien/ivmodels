import logging

import numpy as np
from sklearn.linear_model import LinearRegression

from anchor_regression.utils import anderson_rubin_test, proj

try:
    import pandas as pd

    _PANDAS_INSTALLED = True
except ImportError:
    _PANDAS_INSTALLED = False

logger = logging.getLogger(__name__)


class LinearAnchorRegression(LinearRegression):
    """
    Linear regression with anchor regularization.

    This is based on OLS after a data transformation. First standardizes `X` and `y`,
    as proposed in [1]_.

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
        self.gamma = gamma
        super().__init__(fit_intercept=False)

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
            if self.anchor_names is None and self.anchor_regex is None and check:
                raise ValueError(
                    "If `anchor_names` and `anchor_regex` are None, "
                    "then `a` must be specified."
                )
            elif not isinstance(X, pd.DataFrame):
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
        Fit a linear anchor regression model [1]_.

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


class PULSE(LinearAnchorRegression):
    """
    p-uncorrelated least squares estimator (PULSE) [1]_.

    Perform (linear) anchor regression with regularization parameter `gamma` chosen s.t.
    the Anderson-Rubin test of correlation between the anchor and the residual is not
    significant at level `p_value`.

    Parameters
    ----------
    anchor_names: str or list of str, optional
        The names of the columns in `X` that should be used as anchors. Requires `X` to
        be a pandas DataFrame.
    anchor_regex: str, optional
        A regex that is used to select columns in `X` that should be used as anchors.
        Requires `X` to be a pandas DataFrame. If both `anchor_names` and
        `anchor_regex` are specified, the union of the two is used.
    p_value: float, optional, default = 0.05
        The p-value of the Anderson-Rubin test that is used to determine the regularization
        parameter `gamma`. The PULSE will search for the smallest `gamma` that makes the
        test not significant at level `p_value` with binary search.
    gamma_max: float, optional, default = 1e4
        The maximum value of `gamma` that is used in the binary search. If anchor
        regression with gamma = `gamma_max` is still significant at level `p_value`, an
        error is raised.
    rtol: float, optional, default = 0.1
        The relative tolerance of the binary search.

    References
    ----------
    .. [1] https://arxiv.org/abs/2005.03353
    """

    def __init__(
        self,
        anchor_names=None,
        anchor_regex=None,
        p_value=0.05,
        gamma_max=1e4,
        rtol=0.1,
    ):
        super().__init__(
            gamma=None, anchor_names=anchor_names, anchor_regex=anchor_regex
        )
        self.p_value = p_value
        self.gamma_max = gamma_max
        self.rtol = rtol

    def fit(self, X, y, a=None):
        """Fit a p-uncorrelated least squares estimator (PULSE) [1]_.

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
        _, a_ = self._X_a(X, a, check=False)

        high = self.gamma_max
        low = 1

        # We first check that the "gamma_hat" lies somewhere between 1 and gamma_max.
        # This is equivalent to p_value(1) < self.p_value < p_value(gamma_max).
        self.gamma = self.gamma_max
        super().fit(X, y, a)
        p_value = anderson_rubin_test(a_, y - self.predict(X))[1]
        if p_value < self.p_value:
            raise ValueError(
                f"The Anderson Rubin test is significant at significance level "
                f"{p_value} < {self.p_value} with maximal gamma={self.gamma_max}. "
                "Consider increasing `gamma_max`."
            )

        self.gamma = 1
        super().fit(X, y, a)
        p_value = anderson_rubin_test(a_, y - self.predict(X))[1]
        if p_value > self.p_value:
            raise ValueError(
                f"The Anderson Rubin test is not significant at significance level "
                f"{p_value} > {self.p_value} with gamma=1."
            )

        # We then perform a binary search to find the smallest gamma that satisfies
        # p_value(gamma) >= self.p_value. Throughout, we enforce that
        # p_value(low) < self.p_value < p_value(high).
        while high - low > self.rtol * high:
            mid = (high + low) / 2
            self.gamma = mid
            super().fit(X, y, a)
            p_value = anderson_rubin_test(a_, y - self.predict(X))[1]
            logger.debug(
                f"Anderson-Rubin test with gamma={mid} yields p_value={p_value}."
            )

            if p_value < self.p_value:
                low = mid
            else:
                high = mid

        if low == mid:
            self.gamma = high
            super().fit(X, y, a)

        return self
