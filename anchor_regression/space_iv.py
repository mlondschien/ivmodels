import itertools

import numpy as np

from anchor_regression.linear_model import KClass
from anchor_regression.tests import anderson_rubin_test


class SpaceIV:
    """
    Run the space IV algorithm.

    Returns `argmin ||beta||_0` subject to `AR(beta) <= alpha-quantile`.

    Parameters
    ----------
    s_max : int, optional
        Maximum number of variables to consider.
    alpha : float, optional
        Confidence level.
    """

    def __init__(self, s_max=None, alpha=0.05):
        self.alpha = alpha
        self.s_max = s_max

    def fit(self, X, y, Z=None):  # noqa D
        liml = KClass(kappa="liml")
        if self.s_max is None:
            self.s_max = min(Z.shape[1], X.shape[1])

        for s in range(1, self.s_max + 1):
            best_p_value = 0
            best_S = np.array([])
            best_coef_ = None
            best_intercept_ = None

            for S in itertools.combinations(range(X.shape[1]), s):
                S = np.array(S)
                if len(S) == 0:
                    p_val = anderson_rubin_test(Z, y)[1]
                else:
                    liml.fit(X[:, S], y, Z=Z)
                    p_val = anderson_rubin_test(Z, y - liml.predict(X[:, S]))[1]

                print(f"{S} {p_val}")

                if p_val > best_p_value:
                    best_p_value = p_val
                    best_S = S
                    best_coef_ = liml.coef_
                    best_intercept_ = liml.intercept_

            if best_p_value > self.alpha:
                self.coef_ = np.zeros(X.shape[1])
                self.coef_[best_S] = best_coef_
                self.intercept_ = best_intercept_
                self.S_ = best_S
                self.s_ = s
                break

        return self
