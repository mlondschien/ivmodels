import logging

from ivmodels.kclass import KClass
from ivmodels.tests import pulse_test

logger = logging.getLogger(__name__)


class PULSEMixin:
    """Mixin class for PULSE estimators."""

    def __init__(
        self,
        p_min=0.05,
        rtol=0.01,
        kappa_max=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p_min = p_min
        self.rtol = rtol
        self.kappa_max = kappa_max

    def fit(self, X, y, Z=None, *args, **kwargs):
        """Fit a p-uncorrelated least squares estimator (PULSE) [1].

        If ``instrument_names`` or ``instrument_regex`` are specified, ``X`` must be a
        pandas DataFrame containing columns ``instrument_names`` and ``a`` must be
        ``None``. At least one one of ``a``, ``instrument_names``, and
        ``instrument_regex`` must be specified.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training input samples. If ``instrument_names`` or ``instrument_regex``
            are specified, ``X`` must be a pandas DataFrame containing columns
            ``instrument_names``.
        y: array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values.
        Z: array-like, shape (n_samples, n_anchors), optional
            The instrument (anchor) values. If ``instrument_names`` or
            ``instrument_regex`` are specified, ``Z`` must be ``None``. If ``Z`` is
            specified, ``instrument_names`` and ``instrument_regex`` must be ``None``.
        """
        _, Z_ = self._X_Z(X, Z, check=False)

        if self.kappa_max == 1 and Z_.shape[1] < X.shape[1] and self.alpha == 0:
            raise ValueError(
                "For an underidentified model, that is, Z.shape[1] < X.shape[1], either"
                " kappa_max must be < 1 or alpha must be > 0."
            )

        high = self.kappa_max
        low = 0

        # We first check that the PULSE test is not rejected with significance level
        # self.p_min for two-stage-least-squares (kappa = 1).
        # TODO: Evaluate which estimator minimizes the pulse_test statistic, use that.
        # If, in an underidentified setting, this is given by kappa < 1, use this as the
        # default kappa_max.
        self.kappa = high
        super().fit(X, y, Z, *args, **kwargs)
        p_value_high = pulse_test(Z_, X, y.flatten(), self.coef_)[1]

        if p_value_high < self.p_min:
            raise ValueError(
                f"The PULSE test is rejected at significance level {p_value_high} < "
                f"{self.p_min} at the KClass({high}) estimate."
            )

        self.kappa = 0
        super().fit(X, y, Z, *args, **kwargs)
        p_value_low = pulse_test(Z_, X, y.flatten(), self.coef_)[1]

        if p_value_low > self.p_min:
            return self

        # We then perform a binary search to find the smallest kappa that satisfies
        # p_value(kappa) >= self.p_min. Throughout, we enforce that
        # p_value(low) <= self.p_min <= p_value(high).
        while p_value_high - p_value_low > self.rtol * self.p_min:
            mid = (high + low) / 2
            self.kappa = mid
            super().fit(X, y, Z, *args, **kwargs)
            p_value_mid = pulse_test(Z_, X, y.flatten(), self.coef_)[1]
            logger.debug(
                f"The PULSE test with kappa={mid} yields p_value={p_value_mid}."
            )
            if p_value_mid < self.p_min:
                low = mid
                p_value_low = p_value_mid
            else:
                high = mid
                p_value_high = p_value_mid

        # If, in the last search step of binary search, we had p_value_mid < self.p_min
        # (and thus set low <- mid), we set kappa to high, where p_value >= self.p_min.
        if low == mid:
            self.kappa = high
            super().fit(X, y, Z, *args, **kwargs)

        return self


class PULSE(PULSEMixin, KClass):
    """
    p-uncorrelated least squares estimator (PULSE) :cite:p:`jakobsen2022distributional`.

    Perform k-class estimation with k-class parameter
    :math:`\\kappa \\in [0, \\kappa_\\mathrm{max}]` chosen minimally such that the PULSE
    test of correlation between the instruments and the residuals is not significant at
    level ``p_min``.

    Parameters
    ----------
    instrument_names: str or list of str, optional
        The names of the columns in ``X`` that should be used as anchors. Requires ``X``
        to be a pandas DataFrame.
    instrument_regex: str, optional
        A regex that is used to select columns in ``X`` that should be used as anchors.
        Requires ``X`` to be a pandas DataFrame. If both ``instrument_names`` and
        ``instrument_regex`` are specified, the union of the two is used.
    p_min: float, optional, default = 0.05
        The p-value of the PULSE test that is used to determine the
        k-class parameter :math:`\\kappa`. The PULSE will search for the smallest
        :math:`\\kappa` that makes the test not significant at level ``p_min`` with
        binary search.
    rtol: float, optional, default = 0.01
        The relative tolerance of the binary search. The PULSE will search for a
        :math:`\\kappa` such that the PULSE test is not significant at level ``p_min`
        with binary search but is significant at level ``p_min * (1 + rtol)``.
    kappa_max: float, optional, default = 1
        The maximum value of ``kappa`` to consider. The PULSE will search for the
        smallest ``kappa`` that makes the test not significant at level ``p_min`` with
        binary search. If ``kappa_max = 1``, the PULSE will run a regression equivalent
        to two-stage-least-squares. If ``alpha = 0`` and ``Z.shape[1] < X.shape[1]``,
        this is not well-defined and the PULSE will raise an exception.
    alpha: float, optional, default = 0
        The regularization parameter for elastic net. If ``alpha`` is 0, the estimator
        is unregularized.
    l1_ratio: float, optional, default = 0
        The ratio of L1 to L2 regularization for elastic net. If ``l1_ratio`` is 1, the
        estimator is Lasso. If ``l1_ratio`` is 0, the estimator is Ridge.

    Attributes
    ----------
    coef_: array-like, shape (n_features,)
        The estimated coefficients.
    intercept_: float
        The estimated intercept.
    kappa_: float
        The estimated kappa.

    References
    ----------
    .. bibliography::
       :filter: False

       jakobsen2022distributional
    """

    def __init__(
        self,
        instrument_names=None,
        instrument_regex=None,
        p_min=0.05,
        rtol=0.01,
        kappa_max=1,
        alpha=0,
        l1_ratio=0,
    ):
        super().__init__(
            kappa=kappa_max,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )
        self.p_min = p_min
        self.rtol = rtol
        self.kappa_max = kappa_max
