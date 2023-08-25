import itertools

import numpy as np

from ivmodels.kclass import KClass
from ivmodels.tests import anderson_rubin_test


class SpaceIV:
    """
    Run the space IV algorithm from :cite:t:`pfister2022identifiability`.

    Returns :math:`\\arg\\min \\| \\beta \\|_0` subject to
    :math:`\\mathrm{AR}(\\beta) \\leq q_{1 - \\alpha}`, where :math:`q_{1 - \\alpha}`
    is the :math:`1 - \\alpha` quantile of the F distribution with :math:`q` and
    :math:`n-q` degrees of freedom.

    Parameters
    ----------
    s_max : int, optional, default = None
        Maximum number of variables to consider. If ``None``, set to ``X.shape[1]``.
    p_min : float, optional, default = 0.05
        Confidence level (:math:`\\alpha` above).

    Attributes
    ----------
    coef_ : array-like, shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    S_ : array-like, shape (s,)
        Indices of the selected variables.
    s_ : int
        Number of selected variables.
    kappa_ : float
        Equal to :math:`\\hat\\kappa_\\mathrm{LIML}` for the selected model.

    References
    ----------
    .. bibliography::
       :filter: False

       pfister2022identifiability
    """

    def __init__(self, s_max=None, p_min=0.05):
        self.p_min = p_min
        self.s_max = s_max

    def fit(self, X, y, Z=None):
        """
        Fit a SpaceIV model.

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

                if p_val >= best_p_value:
                    best_p_value = p_val
                    best_S = S
                    best_coef_ = liml.coef_
                    best_intercept_ = liml.intercept_

            if best_p_value > self.p_min:
                self.coef_ = np.zeros(X.shape[1])
                self.coef_[best_S] = best_coef_
                self.intercept_ = best_intercept_
                self.S_ = best_S
                self.s_ = s
                break

        return self
