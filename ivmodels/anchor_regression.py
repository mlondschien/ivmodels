from glum import GeneralizedLinearRegressor

from ivmodels.kclass import KClassMixin


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


class AnchorRegression(AnchorMixin, GeneralizedLinearRegressor):
    """
    Linear regression with anchor regularization :cite:p:`rothenhausler2021anchor`.

    The anchor regression estimator with parameter :math:`\\gamma` is defined as

    .. math:: \\hat\\beta_\\mathrm{anchor}(\\gamma) := \\arg\\min_\\beta \\
       \\| y - X \\beta \\|_2^2 + (\\gamma - 1) \\|P_Z (y - X \\beta) \\|_2^2.

    If :math:`\\gamma \\geq 0`, then :math:`\\hat\\beta_\\mathrm{anchor}(\\gamma) =
    \\hat\\beta_\\mathrm{k-class}((\\gamma - 1) / \\gamma)`.

    The optimization is based on OLS after a data transformation. First standardizes
    ``X`` and ``y`` by subtracting the column means as proposed by
    :cite:t:`rothenhausler2021anchor`. Consequently, no anchor regularization is applied
    to the intercept.

    Parameters
    ----------
    gamma: float
        The anchor regularization parameter. ``gamma=1`` corresponds to OLS.
    instrument_names: str or list of str, optional
        The names of the columns in ``X`` that should be used as instruments (anchors).
        Requires ``X`` to be a pandas DataFrame. If both ``instrument_names`` and
        ``instrument_regex`` are specified, the union of the two is used.
    instrument_regex: str, optional
        A regex that is used to select columns in ``X`` that should be used as instruments
        (anchors). Requires ``X`` to be a pandas DataFrame. If both ``instrument_names``
        and ``instrument_regex`` are specified, the union of the two is used.
    alpha: float, optional, default=0
        Regularization parameter for elastic net regularization.
    l1_ratio: float, optional, default=0
        Ratio of L1 to L2 regularization for elastic net regularization. For
        ``l1_ratio=0`` the penalty is an L2 penalty. For ``l1_ratio=1`` it is an L1
        penalty.

    Attributes
    ----------
    coef_: array-like, shape (n_features,)
        The estimated coefficients for the linear regression problem.
    intercept_: float
        The estimated intercept for the linear regression problem.
    kappa_: float
        The kappa parameter of the corresponding k-class estimator.

    References
    ----------
    .. bibliography::
       :filter: False

       rothenhausler2021anchor
    """

    def __init__(
        self, gamma=1, instrument_names=None, instrument_regex=None, alpha=0, l1_ratio=0
    ):
        super().__init__(
            gamma=gamma,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
            alpha=alpha,
            l1_ratio=l1_ratio,
            family="gaussian",
            fit_intercept=False,
        )
