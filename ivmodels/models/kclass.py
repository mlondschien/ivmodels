import logging
import re

import numpy as np
import scipy
from glum import GeneralizedLinearRegressor

from ivmodels.utils import proj, to_numpy

try:
    import pandas as pd

    _PANDAS_INSTALLED = True
except ImportError:
    _PANDAS_INSTALLED = False

logger = logging.getLogger(__name__)


class KClassMixin:
    """Mixin class for k-class estimators."""

    def __init__(
        self,
        kappa=1,
        instrument_names=None,
        instrument_regex=None,
        exogenous_names=None,
        exogenous_regex=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.kappa = kappa

        if (
            instrument_names is not None
            or instrument_regex is not None
            or exogenous_names is not None
            or exogenous_regex is not None
        ):
            if not _PANDAS_INSTALLED:
                raise ImportError(
                    "pandas is required to use instrument columns or regex."
                )

        self.instrument_names = instrument_names
        self.instrument_regex = instrument_regex
        self.exogenous_names = exogenous_names
        self.exogenous_regex = exogenous_regex

    def _X_Z_C(self, X, Z=None, C=None, predict=False):
        """
        Extract instrument and exogenous columns from X and Z.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data. Must be a pandas DataFrame if any of `instrument_names`,
            `instrument_regex`, `exogenous_names` or `exogenous_names` is not None.
        Z: array-like, shape (n_samples, n_instruments), optional
            The instrument data.
        C: array-like, shape (n_samples, n_exogenous), optional
            The exogenous regressors.
        predict: bool, optional, default=False
            Whether the method is called from the predict method. If True, instruments
            are not necessary.

        Returns
        -------
        X: np.array, shape (n_samples, n_features - n_instrument - n_exogenous)
            The input data with instrument and exogenous columns removed.
        Z: np.array, shape (n_samples, n_instrument)
            The instrument data.
        C: np.array, shape (n_samples, n_exogenous)
            The exogenous data.

        Raises
        ------
        ValueError
            If `Z` is not None and `instrument_names` or `instrument_regex` is not None.
        ValueError
            If `C` is not None and `exogenous_names` or `exogenous_regex` is not None.
        ValueError
            If `instrument_names`, `instrument_regex`, `exogenous_names` or
            `exogenous_regex` is not None and `X` is not a pandas DataFrame.
        ValueError
            If `predict` is False, `instrument_regex` is specified and no columns are
            matched.
        ValueError
            If `predict` is False, `instrument_names` is specified and some columns in
            `instrument_names` are missing in `X`.
        ValueError
            If `exogenous_regex` is specified and no columns are matched.
        ValueError
            If `exogenous_names` is specified and some columns in `exogenous_names` are
            missing in `X`.
        ValueError
            If and the columns selected by `instrument_names` and `instrument_regex` and
            `exogenous_names` and `exogenous_regex` are not disjoint.

        """
        if (
            self.exogenous_names is not None
            or self.exogenous_regex is not None
            or self.instrument_names is not None
            or self.instrument_regex is not None
        ):
            if not _PANDAS_INSTALLED:
                raise ImportError(
                    "pandas is required to use `exogenous_names`, "
                    "`exogenous_regex`, `instrument_names` of `instrument_regex`."
                )

            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "If `instrument_names`, `instrument_regex`, `exogenous_names`, "
                    "or `exogenous_regex` is specified, `X` must be a pandas "
                    "DataFrame."
                )

        if self.exogenous_names is not None or self.exogenous_regex is not None:
            if C is not None:
                raise ValueError(
                    "If `exogenous_names` or `exogenous_regex` is specified, "
                    "`C` must be None."
                )

        if self.instrument_names is not None or self.instrument_regex is not None:
            if Z is not None:
                raise ValueError(
                    "If `instrument_names` or `instrument_regex` is specified, "
                    "`Z` must be None."
                )

        if self.instrument_names is None:
            instrument_names = pd.Index([])
        else:
            instrument_names = pd.Index(self.instrument_names)

            if not instrument_names.isin(X.columns).all():
                if predict:
                    instrument_names = instrument_names.intersection(X.columns)
                else:
                    raise ValueError(
                        "The following instrument columns were not found in X: "
                        f"{set(self.instrument_names) - set(X.columns)}"
                    )

        if self.instrument_regex is not None:
            matched_columns = X.columns[X.columns.str.contains(self.instrument_regex)]
            if not predict and len(matched_columns) == 0:
                raise ValueError(
                    f"No columns in X matched the regex {self.instrument_regex}"
                )
            instrument_names = instrument_names.union(matched_columns)

        if len(instrument_names) > 0:
            Z = X[instrument_names]

        if self.exogenous_names is None:
            exogenous_names = pd.Index([])
        else:
            exogenous_names = pd.Index(self.exogenous_names)

            if not exogenous_names.isin(X.columns).all():
                raise ValueError(
                    "The following exogenous columns were not found in X: "
                    f"{set(self.exogenous_names) - set(X.columns)}"
                )

        if self.exogenous_regex is not None:
            matched_columns = X.columns[X.columns.str.contains(self.exogenous_regex)]
            if len(matched_columns) == 0:
                raise ValueError(
                    f"No columns in X matched the regex {self.exogenous_regex}"
                )
            exogenous_names = exogenous_names.union(matched_columns)

        intersection_names = instrument_names.intersection(exogenous_names)
        if len(intersection_names) > 0:
            raise ValueError(
                f"The columns selected by `instrument_names` and `instrument_regex` "
                f"and `exogenous_names` and `exogenous_regex` must be disjoint. "
                f"The following columns are in both `instrument_names` and "
                f"`exogenous_names`: {intersection_names}"
            )
        if len(exogenous_names) > 0:
            C = X[exogenous_names]

        non_endogenous_names = instrument_names.union(exogenous_names)
        if len(non_endogenous_names) > 0:
            X = X.drop(columns=non_endogenous_names)

        if C is None:
            C = np.zeros((X.shape[0], 0))
        if Z is None:
            Z = np.zeros((X.shape[0], 0))

        return to_numpy(X), to_numpy(Z), to_numpy(C)

    def _fuller_alpha(self, kappa):
        """
        Extract the Fuller alpha parameter from the kappa parameter.

        Parameters
        ----------
        kappa: str
            The kappa parameter. Must be ``"fuller(a)"`` for some integer or float
            ``a``, ``"fuller"``, or ``"liml"``.

        Returns
        -------
        fuller_alpha: float
            The alpha parameter. If kappa is ``"fuller(a)"`` for some integer or float
            ``a``, then ``alpha = a``. If kappa is ``"fuller"``, then ``alpha = 1``.
            If kappa is ``"liml"``, then ``alpha = 0``.
        """
        if not isinstance(kappa, str):
            raise ValueError(f"Invalid kappa {kappa}. Must be a string.")

        fuller_match = re.match(r"fuller(\(\d+\.?\d*\))?", kappa, re.IGNORECASE)
        liml_match = re.match("liml", kappa, re.IGNORECASE)

        if fuller_match is None and liml_match is None:
            raise ValueError(
                f"Invalid kappa: {kappa}. Must be a float or 'fuller(a)' for "
                f"some float integer a or 'liml'."
            )

        if fuller_match is not None and fuller_match.group(1) is not None:
            try:
                return float(fuller_match.group(1)[1:-1])
            except ValueError:
                raise ValueError(
                    f"Invalid kappa: {kappa}. Must be a float or 'fuller(a)' for "
                    f"some float or integer a or 'liml'."
                )
        elif fuller_match is not None:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def _spectrum(X, y, Z=None, X_proj=None, y_proj=None, subset_by_index=None):
        if (y_proj is None or X_proj is None) and Z is None:
            raise ValueError("Either Z or both X_proj and y_proj must be specified.")
        if X_proj is None:
            X_proj = proj(Z, X)
        if y_proj is None:
            y_proj = proj(Z, y)

        Xy = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        Xy_proj = np.concatenate([X_proj, y_proj.reshape(-1, 1)], axis=1)
        return scipy.linalg.eigvalsh(
            a=Xy.T @ Xy_proj, b=(Xy - Xy_proj).T @ Xy, subset_by_index=subset_by_index
        )

    @staticmethod
    def ar_min(X, y, Z=None, X_proj=None, y_proj=None):
        """
        Compute the minimum of the unnormalized Anderson Rubin statistic.

        Computes

        .. math::

           &\\min_{\\beta} \\frac{(y - X \\beta)^T P_Z (y - X \\beta)}{(y - X \\beta)^T M_Z (y - X \\beta)} \\\\
           &=\\lambda_\\mathrm{min}(((X y)^T M_Z (X y))^{-1} (X y)^T P_Z (X y)),

        where :math:`P_Z` is the projection matrix onto the subspace spanned by
        :math:`Z` and :math:`M_Z = \\mathrm{Id} - P_Z`.

        Either ``Z`` or both ``X_proj`` and ``y_proj`` must be specified.

        Parameters
        ----------
        X: np.ndarray of dimension (n, mx)
            Possibly endogenous regressors.
        y: np.ndarray of dimension (n,)
            Outcome.
        Z: np.ndarray of dimension (n, k), optional, default=None.
            Instruments.
        X_proj: np.ndarray of dimension (n, mx), optional, default=None.
            Projection of X onto the subspace orthogonal to Z.
        y_proj: np.ndarray of dimension (n,), optional, default=None.
            Projection of y onto the subspace orthogonal to Z.

        Returns
        -------
        ar_min: float
            The smallest eigenvalue of
            :math:`((X y)^T M_Z (X y))^{-1} (X y)^T P_Z (X y)`,
            where :math:`P_Z` is the projection matrix onto the subspace spanned by `Z`.
        """
        return KClassMixin()._spectrum(
            X=X, y=y, Z=Z, X_proj=X_proj, y_proj=y_proj, subset_by_index=[0, 0]
        )[0]

    def _solve_normal_equations(self, X, y, X_proj, alpha=0):
        if alpha != 0:
            raise NotImplementedError("alpha != 0 not yet implemented for kappa>1.")

        X_kappa = self.kappa_ * X_proj + (1 - self.kappa_) * X
        return np.linalg.solve(X_kappa.T @ X, X_kappa.T @ y).flatten()

    def fit(self, X, y, Z=None, C=None, *args, **kwargs):
        """
        Fit a k-class or anchor regression estimator.

        If ``instrument_names`` or ``instrument_regex`` are specified, ``X`` must be a
        pandas DataFrame containing columns ``instrument_names`` and ``Z`` must be
        ``None``. At least one one of ``Z``, ``instrument_names``, and
        ``instrument_regex`` must be specified.
        If ``exogenous_names`` or ``exogenous_regex`` are specified, ``X`` must be a
        pandas DataFrame containing columns ``exogenous_names`` and ``C`` must be
        ``None``.

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
        C: array-like, shape (n_samples, n_exogenous), optional
            The exogenous regressors. If ``exogenous_names`` or ``exogenous_regex`` are
            specified, ``C`` must be ``None``. If ``C`` is specified,
            ``exogenous_names`` and ``exogenous_regex`` must be ``None``.
        """
        X, Z, C = self._X_Z_C(X, Z, C, predict=False)

        n, mx = X.shape

        # Including an intercept is equivalent to replacing y <- M_1 y, X <- M_1 X,
        # C <- M_1 C, Z <- M_1 Z, where M_1 = Id - 1/n 1 1^T is the projection
        # orthogonal to np.ones(n). I.e., center the columns of all design matrices.
        # We thus subtract the means here and work with the centered data to compute
        # kappa and when fitting the model via solving the normal equations (if
        # kappa > 1). If we fit the model via the normal equation, we manually compute
        # the intercept (for this we need x_mean and y_mean). When fitting the model via
        # OLS + a data transformation (if kappa <= 1), we let
        # glum.GeneralizedLinearRegressor figure out the intercept by adding it back to
        # the data.
        if self.fit_intercept:
            x_mean = X.mean(axis=0)
            y_mean = y.mean(axis=0)

            X = X - x_mean
            y = y - y_mean
            Z = Z - Z.mean(axis=0)

            if C.shape[1] > 0:
                c_mean = C.mean(axis=0)
                C = C - c_mean
            else:
                c_mean = np.zeros(0)
        else:
            x_mean, y_mean, c_mean = np.zeros(mx), 0, np.zeros(C.shape[1])

        # Including exogenous covariates C can be done by the following approaches:
        #
        # - Replace Z <- M_C Z, X <- M_C X, y <- M_C y, where M_C = Id - P_C is the
        #   projection orthogonal to the column space of C. Then apply the k-class
        #   estimator to the transformed data. For the coefficients of C, fit a linear
        #   model y - X \hat\beta_k ~ C. See also the test
        #   test_equivalence_exogenous_covariates_and_fitting_on_residuals
        #
        # - Include C in both the instruments and the endogenous regressors, that is,
        #   replace Z <- [Z C], X <- [X C], y <- y, and apply the k-class estimator to
        #   the augmented data. This is the approach taken here. Care needs to be taken
        #   to compute kappa. For this, we follow the first approach. See below.
        if C.shape[1] > 0:
            Z = np.hstack([Z, C])
            # save some compute here, as proj([Z, C], C) = C
            X_proj = np.hstack([proj(Z, X), C])
            X = np.hstack([X, C])
            x_mean = np.hstack([x_mean, c_mean])

        else:
            X_proj = proj(Z, X)

        y_proj = proj(Z, y)

        if isinstance(self.kappa, str):
            if self.kappa.lower() in {"tsls", "2sls"}:
                self.kappa_ = 1
            elif self.kappa.lower() == "ols":
                self.kappa_ = 0
            else:
                self.fuller_alpha_ = self._fuller_alpha(self.kappa)
                # If C!=None, we compute the ar_min as if we removed C from all design
                # matrices. I.e., we replace Z <- M_C Z, X <- M_C X, y <- M_C y.
                # We also exclude the columns in X coming from C.
                self.ar_min_ = self.ar_min(
                    X[:, :mx] - proj(C, X[:, :mx]),
                    y - proj(C, y),
                    X_proj=X_proj[:, :mx],
                    y_proj=y_proj,
                )
                self.kappa_liml_ = 1 + self.ar_min_
                self.kappa_ = self.kappa_liml_ - self.fuller_alpha_ / (n - Z.shape[1])
        else:
            self.kappa_ = self.kappa

        # If kappa <=1, the k-class estimator is an anchor regression estimator and
        # sqrt( (1-kappa) * Id + kappa * P_Z) ) = sqrt(1-kappa) * Id + sqrt(kappa) * P_Z
        # and we apply linear regression to the transformed data.
        if self.kappa_ <= 1:
            X_tilde = (
                np.sqrt(1 - self.kappa_) * X + (1 - np.sqrt(1 - self.kappa_)) * X_proj
            ) + x_mean  # Add means so glum.GLM can figure out the intercept.
            y_tilde = (
                np.sqrt(1 - self.kappa_) * y + (1 - np.sqrt(1 - self.kappa_)) * y_proj
            ) + y_mean
            super().fit(X_tilde, y_tilde, *args, **kwargs)

        else:
            if args or kwargs:
                raise ValueError(
                    f"Arguments {args} and {kwargs} are not supported for kappa > 1."
                )
            self.coef_ = self._solve_normal_equations(
                X, y, X_proj=X_proj, alpha=getattr(self, "alpha", 0)
            )

            if self.fit_intercept:
                self.intercept_ = -self.coef_.T @ x_mean + y_mean
            else:
                # needed for glum.GeneralizedLinearRegressor.predict
                self.intercept_ = 0

        return self

    def predict(self, X, C=None, *args, **kwargs):  # noqa D
        X, _, C = self._X_Z_C(X, C=C, Z=None, predict=True)
        return super().predict(np.hstack([X, C]), *args, **kwargs)


class KClass(KClassMixin, GeneralizedLinearRegressor):
    """K-class estimator for instrumental variable regression.

    The k-class estimator with parameter :math:`\\kappa` is defined as

    .. math::

       \\hat\\beta_\\mathrm{k-class}(\\kappa) &:= \\arg\\min_\\beta \\
       (1 - \\kappa) \\| y - X \\beta \\|_2^2 + \\kappa \\|P_Z (y - X \\beta) \\|_2^2
       \\\\
       &= (X^T (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X)^{-1} X^T
       (\\kappa P_Z + (1 - \\kappa) \\mathrm{Id}) X) y,

    where :math:`P_Z = Z (Z^T Z)^{-1} Z^T` is the projection matrix onto the subspace
    spanned by :math:`Z` and :math:`\\mathrm{Id}` is the identity matrix.
    This includes the the ordinary least-squares (OLS) estimator (:math:`\\kappa = 0`),
    the two-stage least-squares (2SLS) estimator
    (:math:`\\kappa = 1`), the limited information maximum likelihood (LIML) estimator
    (:math:`\\kappa = \\hat\\kappa_\\mathrm{LIML}`), and the Fuller estimator
    (:math:`\\kappa = \\hat\\kappa_\\mathrm{LIML} - \\alpha / (n - k)`) as special
    cases.

    Specifying exogenous included regressors :math:`C` is equivalent to including them
    into both :math:`Z` and :math:`X`.

    Parameters
    ----------
    kappa: float or { "ols", "tsls", "2sls", "liml", "fuller", "fuller(a)"}
        The kappa parameter of the k-class estimator. If float, then kappa must be in
        :math:`[0, \\hat\\kappa_\\mathrm{LIML}]`, where
        :math:`\\kappa_\\mathrm{LIML} \\geq 1` is the smallest eigenvalue of the
        matrix :math:`((X \\ \\ y)^T M_Z (X \\ \\ y))^{-1} (X \\ \\ y)^T (X \\ y)`,
        where :math:`P_Z` is the projection matrix onto the subspace spanned by :math:`Z`
        and :math:`M_Z = Id - P_Z`.
        If exogenous included regressors math:`C` are specified, then
        :math:`\\kappa_\\mathrm{LIML}` is the smallest eigenvalue of the matrix
        :math:`((X \\ \\ y)^T M_{[Z, C]} (X \\ \\ y))^{-1} (X \\ \\ y)^T M_C (X \\ y)`.
        If string, then must be one of ``"ols"``, ``"2sls"``, ``"tsls"``, ``"liml"``,
        ``"fuller"``, or ``"fuller(a)"``, where ``a`` is numeric. If ``kappa="ols"``,
        then ``kappa=0`` and the k-class estimator is the ordinary least squares
        estimator. If ``kappa="tsls"`` or ``kappa="2sls"``, then ``kappa=1`` and the
        k-class estimator is the two-stage least-squares estimator. If ``kappa="liml"``,
        then :math:`\\kappa = \\hat\\kappa_\\mathrm{LIML}` is used. If
        ``kappa="fuller(a)"``, then
        :math:`\\kappa = \\hat\\kappa_\\mathrm{LIML} - a / (n - k)`, where
        :math:`n` is the number of observations and :math:`q = \\mathrm{dim}(Z)` is the
        number of instruments. The string ``"fuller"`` is interpreted as
        ``"fuller(1.0)"``, yielding an estimator that is unbiased up to
        :math:`O(1/n)` :cite:p:`fuller1977some`.
    instrument_names: str or list of str, optional
        The names of the columns in ``X`` that should be used as instruments.
        Requires ``X`` argument of ``fit`` method to be a pandas DataFrame. If both
        ``instrument_names`` and ``instrument_regex`` are specified, the union of the
        two is used.
    instrument_regex: str, optional
        A regex that is used to select columns in ``X`` that should be used as
        instruments. Requires ``X`` argument of ``fit`` method to be a pandas DataFrame.
        If both ``instrument_names`` and ``instrument_regex`` are specified, the union
        of the two is used.
    exogenous_names: str or list of str, optional
        The names of the columns in ``X`` that should be used as exogenous regressors.
        Requires ``X`` argument of ``fit`` method to be a pandas DataFrame. If both
        ``exogenous_names`` and ``exogenous_regex`` are specified, the union of the two
        is used.
    exogenous_regex: str, optional
        A regex that is used to select columns in ``X`` that should be used as exogenous
        regressors. Requires ``X`` argument of ``fit`` method to be a pandas DataFrame.
        If both ``exogenous_names`` and ``exogenous_regex`` are specified, the union of
        the two is used.
    alpha: float, optional, default=0
        Regularization parameter for elastic net regularization. Only implemented for
        :math:`\\kappa \\leq 1`.
    l1_ratio: float, optional, default=0
        Ratio of L1 to L2 regularization for elastic net regularization. For
        ``l1_ratio=0`` the penalty is an L2 penalty. For ``l1_ratio=1`` it is an L1
        penalty. Only implemented for :math:`\\kappa \\leq 1`.

    Attributes
    ----------
    coef_: array-like, shape (n_features,)
        The estimated coefficients for the linear regression problem.
    intercept_: float
        The estimated intercept for the linear regression problem.
    kappa_: float
        The kappa parameter of the k-class estimator.
    fuller_alpha_: float
        If ``kappa`` is one of ``{"fuller", "fuller(a)", "liml"}`` for some numeric
        value ``a``, the alpha parameter of the Fuller estimator.
    ar_min_: float
        If ``kappa`` is one of ``{"fuller", "fuller(a)", "liml"}`` for some numeric
        value ``a``, the minimum of the unnormalized Anderson Rubin statistic.
    kappa_liml_: float
        If ``kappa`` is one of ``{"fuller", "fuller(a)", "liml"}`` for some numeric
        value ``a``, the kappa parameter of the LIML estimator, equal to
        ``1 + ar_min_``.

    References
    ----------
    .. bibliography::
       :filter: False

       fuller1977some
    """

    def __init__(
        self,
        kappa=1,
        instrument_names=None,
        instrument_regex=None,
        exogenous_names=None,
        exogenous_regex=None,
        alpha=0,
        l1_ratio=0,
        fit_intercept=True,
    ):
        super().__init__(
            kappa=kappa,
            instrument_names=instrument_names,
            instrument_regex=instrument_regex,
            exogenous_names=exogenous_names,
            exogenous_regex=exogenous_regex,
            family="gaussian",
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
        )
