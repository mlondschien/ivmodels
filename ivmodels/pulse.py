import logging

from ivmodels.linear_model import AnchorRegression
from ivmodels.tests import pulse_test

logger = logging.getLogger(__name__)


class PULSEMixin:
    """Mixin class for PULSE estimators."""

    def __init__(
        self,
        p_min=0.05,
        rtol=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p_min = p_min
        self.rtol = rtol

    def fit(self, X, y, Z=None, *args, **kwargs):
        """Fit a p-uncorrelated least squares estimator (PULSE) [1]_.

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
        Z: array-like, shape (n_samples, n_anchors), optional
            The instrument (anchor) values. If `instrument_names` or `instrument_regex`
            are specified, `Z` must be `None`. If `Z` is specified, `instrument_names`
            and `instrument_regex` must be `None`.
        """
        _, Z_ = self._X_Z(X, Z, check=False)

        high = 1
        low = 0

        # We first check that the PULSE test is not rejected with significance level
        # self.p_min for two-stage-least-squares (kappa = 1).
        # TODO: Evaluate which estimator minimizes the pulse_test statistic, use that.
        self.kappa = 1
        super().fit(X, y, Z, *args, **kwargs)
        p_value = pulse_test(Z_, y.flatten() - self.predict(X))[1]

        if p_value < self.p_min:
            raise ValueError(
                f"The PULSE test is rejected at significance level {p_value} < "
                f"{self.p_min} at the two-stage-least-squares estimate."
            )

        self.kappa = 0
        super().fit(X, y, Z, *args, **kwargs)
        p_value = pulse_test(Z_, y.flatten() - self.predict(X))[1]

        if p_value > self.p_min:
            raise ValueError(
                f"The PULSE test is significant at significance level {p_value} < "
                f"{self.p_min} only at the ordinary least squares estimate."
            )

        # We then perform a binary search to find the smallest kappa that satisfies
        # p_value(kappa) >= self.p_min. Throughout, we enforce that
        # p_value(low) <= self.p_min <= p_value(high).
        while high - low > self.rtol * high:
            mid = (high + low) / 2
            self.kappa = mid
            super().fit(X, y, Z, *args, **kwargs)
            p_value = pulse_test(Z_, y - self.predict(X))[1]
            logger.debug(
                f"The PULSE test with kappa={mid} yields p_value={p_value}."
            )
            if p_value < self.p_min:
                low = mid
            else:
                high = mid

        # If, in the last search step of binary search, we had p_value < self.p_min at
        # mid (and thus set low <- mid), we set kappa to high, where p_value >= self.p_min.
        if low == mid:
            self.kappa = high
            super().fit(X, y, Z, *args, **kwargs)

        return self


class PULSE(PULSEMixin, AnchorRegression):
    """
    p-uncorrelated least squares estimator (PULSE) [1]_.

    Perform (linear) k-class estimation parameter `kappa` chosen s.t. the PULSE test of
    correlation between the anchor and the residual is not significant at level
    `p_value`.

    Parameters
    ----------
    instrument_names: str or list of str, optional
        The names of the columns in `X` that should be used as anchors. Requires `X` to
        be a pandas DataFrame.
    instrument_regex: str, optional
        A regex that is used to select columns in `X` that should be used as anchors.
        Requires `X` to be a pandas DataFrame. If both `instrument_names` and
        `instrument_regex` are specified, the union of the two is used.
    p_min: float, optional, default = 0.05
        The p-value of the Anderson-Rubin test that is used to determine the regularization
        parameter `gamma`. The PULSE will search for the smallest `gamma` that makes the
        test not significant at level `p_min` with binary search.
    rtol: float, optional, default = 0.1
        The relative tolerance of the binary search.

    References
    ----------
    .. [1] https://arxiv.org/abs/2005.03353
    """

    def __init__(
        self,
        instrument_names=None,
        instrument_regex=None,
        p_min=0.05,
        rtol=0.1,
    ):
        super().__init__(
            gamma=1,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
        )
        self.p_min = p_min
        self.rtol = rtol
