import numpy as np
from sklearn.linear_model import LinearRegression

try:
    import pandas as pd

    _PANDAS_INSTALLED = True
except ImportError:
    _PANDAS_INSTALLED = False


class LinearAnchorRegression(LinearRegression):
    """
    Linear regression with anchor regularization.

    This is based on OLS after a data transformation. First standardizes `X` and `y`,
    as proposed in [1]_.

    Parameters
    ----------
    gamma: float
        The anchor regularization parameter. Gamma=1 corresponds to standard OLS.
    anchor_column_names: str or list of str, optional
        The names of the columns in `X` that should be used as anchors. Requires `X` to be a
        pandas DataFrame.
    anchor_column_regex: str, optional
        A regex that is used to select columns in `X` that should be used as anchors. Requires
        `X` to be a pandas DataFrame. If both `anchor_column_names` and `anchor_column_regex`
        are specified, the union of the two is used.

    References
    ----------
    .. [1] https://arxiv.org/abs/1801.06229
    """

    def __init__(self, gamma, anchor_column_names=None, anchor_column_regex=None):
        self.gamma = gamma
        super().__init__(fit_intercept=False)

        if anchor_column_names is not None or anchor_column_regex is not None:
            if not _PANDAS_INSTALLED:
                raise ImportError("pandas is required to use anchor columns or regex")

        self.anchor_column_names = anchor_column_names
        self.anchor_column_regex = anchor_column_regex

    def _anchor_columns(self, X):
        anchor_columns = pd.Index([])
        if self.anchor_column_regex is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "anchor_column_regex can only be used with pandas DataFrames"
                )
            anchor_columns = anchor_columns.union(
                X.columns[X.columns.str.contains(self.anchor_column_regex)]
            )
        if self.anchor_column_names is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "anchor_column_names can only be used with pandas DataFrames"
                )
            anchor_columns = anchor_columns.union(self.anchor_column_names)

        return anchor_columns

    def fit(self, X, y):  # noqa D
        anchor_columns = self._anchor_columns(X)
        a = X[anchor_columns]
        X = X[X.columns.difference(anchor_columns)]

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
        anchor_columns = self._anchor_columns(X)

        # if X is a numpy array and anchor columns are specified, _anchor_column raises
        if isinstance(X, pd.DataFrame):
            X = X[X.columns.difference(anchor_columns)]

        return super().predict(X)


def proj(anchor, f):
    """Project f onto the subspace spanned by anchor.

    Parameters
    ----------
    anchor: np.ndarray of dimension (n, d_anchor).
        The anchor matrix.
    f: np.ndarray of dimension (n, d_f) or (n,).
        The vector to project.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of f onto the subspace spanned by anchor. Same dimension as f.
    """
    anchor = anchor - anchor.mean(axis=0)
    f = f - f.mean(axis=0)

    return np.dot(anchor, np.linalg.lstsq(anchor, f, rcond=None)[0])
