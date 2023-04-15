import logging

from anchor_regression.linear_model import LinearAnchorRegression
from anchor_regression.utils import pulse_test

logger = logging.getLogger(__name__)


class PULSEMixin:
    """Mixin class for PULSE estimators."""

    def __init__(
        self,
        p_value=0.05,
        gamma_max=1e4,
        rtol=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        p_value = pulse_test(a_, y - self.predict(X))[1]
        if p_value < self.p_value:
            raise ValueError(
                f"The Anderson Rubin test is significant at significance level "
                f"{p_value} < {self.p_value} with maximal gamma={self.gamma_max}. "
                "Consider increasing `gamma_max`."
            )

        self.gamma = 1
        super().fit(X, y, a)
        p_value = pulse_test(a_, y - self.predict(X))[1]
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
            p_value = pulse_test(a_, y - self.predict(X))[1]
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


class PULSE(PULSEMixin, LinearAnchorRegression):
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
